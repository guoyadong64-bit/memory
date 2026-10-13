"""Fresh construction/indexing and all 24 answers from fixed shared observations."""
import argparse
import hashlib
import shutil
from pathlib import Path

from benchmarks.findingdory_mem0 import ComparisonClients, code_hashes
from benchmarks.findingdory_pilot import load, save
from benchmarks.findingdory_sequence import compare
from benchmarks.findingdory_structured import freeze, fingerprint
from benchmarks.findingdory_visual_answer import VISUAL_ANSWER_SYSTEM
from benchmarks.run_budget import RunBudget
from memory_agent.settings import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=Path('runtime/findingdory_run_20261013_mem0_v16'))
    parser.add_argument('--run-dir',type=Path,default=Path('runtime/findingdory_run_20261013_full_compare'))
    parser.add_argument('--data-dir',type=Path,default=Path('runtime/findingdory'))
    args = parser.parse_args()
    source,run,data = args.source_run,args.run_dir,args.data_dir
    if run.resolve() in {source.resolve(),data.resolve()}:
        raise ValueError('Use an isolated output directory')
    settings = Settings.load().model_copy(update={'timeout':120,'retries':0})
    model_keys = ('llm_model','vision_model','answer_model','embedding_model')
    models = {key:getattr(settings,key) for key in model_keys}
    raw = {ep:load(source/'observations'/f'{ep}.json') for ep in ('ep_1','ep_33')}
    hashes = code_hashes() | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'benchmarks/findingdory_sequence.py','benchmarks/findingdory_visual_answer.py','benchmarks/answer_normalization.py','benchmarks/run_budget.py')}
    freeze(run/'evaluation_config.json',{'models':models,'source_run':str(source.resolve()),
        'task_sha256':fingerprint(load(data/'selected_tasks.json')),
        'observation_sha256':{ep:fingerprint(value) for ep,value in raw.items()},
        'answer_prompt_sha256':fingerprint(VISUAL_ANSWER_SYSTEM),'code_sha256':hashes,
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol':'fresh hypergraph, fresh Mem0 SDK extraction/index for BOTH episodes, all 24 new answers',
        'perception':'fixed shared 192 observations; no new video extraction',
        'retrieval_k':20,'max_context_utf8_bytes':40000,'image_limit':12,
        'historical_answers_reused':0})
    prior = load(source/'api_budget.json')
    budget = RunBudget(run,max_requests=500,max_reported_tokens=prior['limits']['max_reported_tokens'])
    if 'carried_usage' not in budget.state:
        budget.state['reported_tokens'] = prior['reported_tokens']
        budget.state['carried_usage'] = {'source_run':str(source.resolve()),'reported_tokens':prior['reported_tokens']}
        budget.state['historical_usage'] = prior['historical_usage']
        budget.save()
    for ep in raw:
        path = run/'observations'/f'{ep}.json'; path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():shutil.copy2(source/'observations'/f'{ep}.json',path)
        if fingerprint(load(path)) != fingerprint(raw[ep]):raise ValueError('Shared inputs changed')
    dimensions_path = run/'embedding_dimensions.json'
    if not dimensions_path.exists():
        client = ComparisonClients(settings,budget)
        try:dimensions = len(client.embed(['Embedding dimension probe for fresh memory indexing'])[0])
        finally:client.close()
        save(dimensions_path,{'model':settings.embedding_model,'dimensions':dimensions})
    dimensions = load(dimensions_path)['dimensions']
    config_path = run/'mem0_config.json'
    if not config_path.exists():
        # Record the current SDK configuration, never copy old facts or indices.
        freeze(config_path,{'sdk_version':'2.2.1','infer':True,'chunk_frames':8,
            'llm':settings.llm_model,'embedding':settings.embedding_model,'dimensions':dimensions,
            'storage':'fresh local Qdrant + SQLite per episode','nlp':'SDK default regex fallback',
            'observation_hashes':{ep:fingerprint(value) for ep,value in raw.items()},
            'new_input_ep_33_sha256':fingerprint(raw['ep_33']),
            'ep_1':'fresh official SDK extraction and indexing','ep_33':'fresh official SDK extraction and indexing'})
    print({'started':True,'models':models,'dimensions':dimensions,'historical_answers_reused':0},flush=True)
    compare(settings,source,run,data,run,budget,fresh_all_indexes=True)


if __name__ == '__main__':main()
