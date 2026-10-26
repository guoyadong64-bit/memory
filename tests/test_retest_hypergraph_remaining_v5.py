import json

import httpx
import pytest

from benchmarks import retest_hypergraph_remaining_v5 as runner
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_structured import fingerprint
from memory_agent.settings import Settings


def questions():
    return [{'row_idx':i,'ep_id':'ep_1','task_id':f'task_{i}','question':f'question {i}',
             'answer':json.dumps([[i]]),'high_level_category':'Spatial','low_level_category':'Recall'} for i in range(4)]


def test_selection_preserves_both_initial_and_v4_correct_answers(tmp_path,monkeypatch):
    source,v4,shared=(tmp_path/name for name in ('source','v4','shared'))
    tasks=questions()
    settings=Settings(llm_model='fake')
    baseline={i:{'correct':i==0,'prediction':{'goal_frames':[0]}} for i in range(4)}
    monkeypatch.setattr(runner.previous,'baseline_errors',lambda *args:(tasks[1:],baseline))
    atomic_json(source/'config.json',{'models':{'llm_model':'fake'}})
    atomic_json(shared/'observations_manifest.json',{'sealed':True})
    atomic_json(v4/'config.json',{'method':'hypergraph_recall_v4','models':{'llm_model':'fake'},
        'tasks_sha256':fingerprint(tasks[1:]),'observations_manifest_sha256':runner.common.checksum(shared/'observations_manifest.json'),
        'source_run':str(source),'source_config_sha256':runner.common.checksum(source/'config.json'),
        'source_answers_sha256':fingerprint(baseline)})
    for task in tasks[1:]:
        i=task['row_idx']; n=1 if i==1 else 0
        atomic_json(v4/'answers'/f'{i}.json',{'row_idx':i,'method':'hypergraph_recall_v4',
            'task_sha256':fingerprint(task),'prediction':{'goal_frames':[n]},'evidence':[{'frame_indices':[n]}],
            'correct':i==1,'candidate_frame_covered':float(i==1)})
    atomic_json(v4/'summary.json',{'complete':True,'completed':3,'correct':1})
    selected,current=runner.remaining_errors(source,v4,tasks,shared,settings)
    assert [t['row_idx'] for t in selected] == [2,3]
    assert sum(r['correct'] for r in current.values()) == 2
    with pytest.raises(RuntimeError,match='differ'):
        runner.remaining_errors(source,v4,tasks,shared,settings.model_copy(update={'llm_model':'changed'}))


def test_remaining_only_resume_does_not_repeat_paid_calls_or_claim_full_accuracy(tmp_path):
    tasks=questions()[2:]
    current={i:{'correct':i<2} for i in range(4)}
    shared=tmp_path/'shared'
    atomic_json(shared/'perception/api_budget.json',{'requests_started':0,'reported_tokens':0})
    sent=[]; crash=[True]
    def transport(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200,json={'model':'fake','usage':{'total_tokens':10}})
    def clients(settings,budget,cache,**kwargs):
        return runner.previous.RetestClients(settings,budget,cache,transport=httpx.MockTransport(transport),**kwargs)
    class Method:
        def __init__(self,run,shared,observations,client,progress):self.client=client
        def build(self):pass
        def close(self):pass
        def retrieve(self,episode,question):
            self.client.post('https://mock/api','fake',{'model':'fake','question':question},'embedding')
            return [{'id':'fact','frame_indices':[int(question[-1])]}],None,{'read_projection_v5':{}}
    def decoder(client,question,*args):
        client.post('https://mock/api','fake',{'model':'fake','question':question},'answer')
        if question=='question 3' and crash[0]:
            crash[0]=False
            raise RuntimeError('interrupted after response')
        return {'goal_frames':[int(question[-1])]}
    args=(tmp_path/'run',tmp_path/'source',shared,tasks,{'ep_1':[]},Settings(llm_key='fake'),current)
    kwargs={'method_factory':Method,'client_factory':clients,'decoder':decoder}
    with pytest.raises(RuntimeError,match='interrupted'):
        runner.evaluate(*args,**kwargs)
    assert len(sent)==4
    result=runner.evaluate(*args,**kwargs)
    assert result['complete'] and result['correct']==result['total']==2
    assert result['historical_correct_not_retested']==2 and result['new_overall_accuracy'] is None
    assert result['api_usage']['requests_started']==4==len(sent)
    runner.evaluate(*args,**kwargs)
    assert len(sent)==4
    assert len(list((tmp_path/'run/answers').glob('*.json')))==2
