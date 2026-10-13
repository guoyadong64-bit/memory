import importlib.util
import json
import pytest
from benchmarks.amem_adapter import create_memory,save_state,evidence,AMemProviderStop,CompletionAdapter
from memory_agent.settings import Settings

class Client:
    settings=Settings(llm_model='test',embedding_model='test',llm_key='test')
    def __init__(self):self.calls=[]
    def chat_options(self,model):return {}
    def embed(self,texts):
        self.calls.append('embedding')
        return [[1.0,float('second' in text),.5] for text in texts]
    def post(self,url,key,payload,role):
        self.calls.append(role)
        out=({'keywords':['cup','robot','table'],'context':'robot transfers a cup','tags':['robot','action','location']}
             if role=='amem_metadata' else {'should_evolve':True,'actions':['strengthen'],
                'suggested_connections':['note_first'],'tags_to_update':['cup','related','action'],
                'new_context_neighborhood':[],'new_tags_neighborhood':[]})
        return {'choices':[{'message':{'content':json.dumps(out)}}]}

@pytest.mark.skipif(importlib.util.find_spec('chromadb') is None,reason='A-Mem isolated environment required')
def test_official_metadata_linking_provenance_and_cached_resume(tmp_path):
    client=Client();memory=create_memory(tmp_path,client)
    memory.add_note('At frame_index 1, the first cup is on the table.',id='note_first')
    memory.add_note('At frame_index 2, the second robot view shows a cup.',id='note_second')
    assert type(memory).__module__=='agentic_memory.memory_system'
    assert client.calls.count('amem_metadata')==2 and client.calls.count('amem_evolution')==1
    assert memory.read('note_second').links==['note_first']
    found=memory.search_agentic('cup',k=2)
    mapped=evidence(memory,found,{'note_first':[1],'note_second':[2]})
    assert len(mapped)==2 and {f for x in mapped for f in x['frame_indices']}=={1,2}
    indexed_context=memory.retriever.collection.get(ids=['note_first'],include=['metadatas'])['metadatas'][0]['context']
    memory.read('note_first').context='updated context before official consolidation'
    save_state(memory,tmp_path)
    before=len(client.calls);restored=create_memory(tmp_path,client)
    assert len(restored.memories)==2 and restored.read('note_second').links==['note_first']
    assert restored.read('note_first').context=='updated context before official consolidation'
    assert restored.retriever.collection.get(ids=['note_first'],include=['metadatas'])['metadatas'][0]['context']==indexed_context
    assert len(client.calls)==before

def test_provider_failure_crosses_official_exception_fallbacks(tmp_path):
    class Broken(Client):
        def post(self,*args):raise RuntimeError('simulated')
    with pytest.raises(AMemProviderStop):
        CompletionAdapter(Broken(),tmp_path).get_completion('test',{'type':'json_object'})
