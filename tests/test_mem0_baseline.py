import importlib.util
import json

import pytest

from benchmarks.findingdory_mem0 import decode
from memory_agent.settings import Settings
from benchmarks.mem0_adapter import memory_frame_indices


def test_mem0_fact_cannot_claim_every_frame_of_its_ingestion_batch():
    assert memory_frame_indices({"memory":"At frame_index 3 the cup rests.",
        "metadata":{"frame_indices":list(range(8))}})==[3]
    assert memory_frame_indices({"memory":"The cup rests.",
        "metadata":{"frame_indices":list(range(8))}})==[]


def test_common_decoder_rejects_uncited_or_out_of_provenance_frames():
    class Client:
        def generate(self,*args):return {"goal_frames":[90],"evidence_ids":["fact"]}
    result=decode(Client(),"Where?",[{"id":"fact","content":"cup at frame 3","frame_indices":[3]}])
    assert result["goal_frames"] == []


@pytest.mark.skipif(importlib.util.find_spec("mem0") is None, reason="Optional official SDK environment")
def test_official_mem0_extracts_and_searches_with_budgeted_transport(tmp_path):
    from benchmarks.mem0_adapter import create_memory
    class Client:
        settings=Settings(llm_model="test",llm_key="test",embedding_model="test")
        def __init__(self):self.prompts=[]
        def chat_options(self,model):return {}
        def post(self,url,key,payload,role):
            self.prompts.append((role,payload["messages"]))
            return {"choices":[{"message":{"content":json.dumps({"memory":[
                {"text":"At frame_index 3 the cup is on the desk."}]})}}]}
        def embed(self,texts):return [[1.0,0.2,0.1] for _ in texts]
    client=Client()
    memory=create_memory(tmp_path,client,3)
    try:
        result=memory.add([{"role":"user","content":"At frame_index 3 the cup is on the desk."}],
                          user_id="episode",metadata={"frame_indices":[3]},infer=True)
        assert result["results"] and client.prompts[0][0]=="mem0_extract"
        assert type(memory).__module__=="mem0.memory.main"
        found=memory.search("cup",top_k=5,filters={"user_id":"episode"})["results"]
        assert found and found[0]["metadata"]["frame_indices"]==[3]
    finally:memory.close()
