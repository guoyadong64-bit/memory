from benchmarks.answer_normalization import select_any_target

def test_any_target_uses_first_citable_candidate_without_gold():
    output={'goal_frames':[99,4,8],'evidence_ids':['proof'],'target_bindings':[
        {'frame_index':99,'entity_id':'a'},{'frame_index':4,'entity_id':'b'},{'frame_index':8,'entity_id':'c'}]}
    chosen,audit=select_any_target(output,'Navigate to any object you did not interact with.',
                                  [{'id':'proof','frame_indices':[4,8]}])
    assert chosen['goal_frames']==[4] and chosen['target_bindings'][0]['entity_id']=='b'
    assert audit['original_output']==output and output['goal_frames']==[99,4,8]

def test_all_and_ordered_queries_are_never_truncated():
    output={'goal_frames':[4,8],'evidence_ids':['proof']}
    evidence=[{'id':'proof','frame_indices':[4,8]}]
    for question in ('Revisit all objects, any order.','Any suggestion? The order to revisit them is: second, first.',
                     'Navigate to a tape.'):
        assert select_any_target(output,question,evidence)==(output,None)

def test_normalization_cannot_make_uncited_output_valid():
    output={'goal_frames':[4,8],'evidence_ids':['invented']}
    assert select_any_target(output,'Navigate to any object.',[{'id':'proof','frame_indices':[4,8]}])==(output,None)
