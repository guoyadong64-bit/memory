import json
from benchmarks.findingdory_visual_answer import answer_errors,select_images,decode_visual
from benchmarks.findingdory_memory import structured_records,history_query
from memory_agent.recall import graph_plan,parse_plan
from test_findingdory_structured import frame
from memory_agent.settings import Settings


def test_unresolved_category_recalls_identity_views_without_aliasing():
    records = structured_records('sample',[frame()])
    query = history_query('Navigate to a previously unnamed category.','sample',records)
    query.recall_plan = parse_plan(query.question)
    result = graph_plan(records,query)
    assert result['record_ids'] and not result['targets'] and not result['complete']
    assert any('Category unresolved' in warning for warning in result['warnings'])


def test_image_budget_keeps_one_witness_per_memory_episode_before_seconds():
    evidence=[{'id':f'event_{i}','frame_indices':[i*3,i*3+1],
               'content':{'trajectory':[{'phase':'transport'}]}} for i in range(6)]
    chosen=select_images(evidence,[],limit=6)
    assert chosen==[0,3,6,9,12,15]
    assert select_images(evidence,[],limit=1)==[0]
    assert select_images(evidence,[],limit=0)==[]


def test_constraints_check_question_count_and_target_identity_without_gold():
    evidence=[{'id':'proof','frame_indices':[0,1,2]}]
    question='Revisit in order. The order to revisit them is: second, first.'
    assert 'requested_target_count_not_met' in answer_errors({'goal_frames':[0],'evidence_ids':['proof']},evidence,question)
    output={'goal_frames':[0,1],'evidence_ids':['proof'],'target_bindings':[
        {'frame_index':0,'entity_id':'same_support','evidence_ids':['proof']},
        {'frame_index':1,'entity_id':'same_support','evidence_ids':['proof']}]}
    assert 'duplicate_or_invalid_physical_target_identity' in answer_errors(output,evidence,question)
    output['target_bindings'][1]['entity_id']='invented_support'
    assert 'undeclared_physical_target_identity' in answer_errors(output,evidence,question,identity_ids={'same_support'})


def test_answer_repair_is_bounded_and_does_not_add_evidence(tmp_path):
    directory=tmp_path/'frames'/'sample';directory.mkdir(parents=True)
    (directory/'000.jpg').write_bytes(b'test image')
    class Client:
        settings=Settings(llm_model='test')
        requests=[]
        def chat_options(self,model):return {}
        def post(self,url,key,payload,role):
            self.requests.append(payload)
            # Both attempts remain incomplete; do not guess a second target.
            return {'choices':[{'message':{'content':json.dumps({'goal_frames':[0],'evidence_ids':['proof']})}}]}
    client=Client()
    result=decode_visual(client,'The order to revisit them is: second, first.',
                         [{'id':'proof','content':'observation','frame_indices':[0]}],[],tmp_path,'sample')
    assert len(client.requests)==2 and not result['goal_frames']
    assert result['visual_frames']==[0]
    assert 'requested_target_count_not_met' in result['validation']
