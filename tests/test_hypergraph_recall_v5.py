import inspect
import json

import pytest

from benchmarks.hypergraph_recall_v5 import WitnessIndex, project_retrieval, select_images_v5, LIMITS
from memory_agent.models import MemoryEntity, RoleBinding, Evolution, Stage
from test_hypergraph_recall_v4 import frame_record


def test_named_observation_outside_semantic_seeds_can_be_recalled_with_provenance():
    records = [frame_record('ep', n, identity='ball') for n in range(40)]
    records += [frame_record('ep', 70, identity='blue_cup')]
    before = [r.model_dump(mode='json') for r in records]
    evidence, _, audit = project_retrieval(records, [r.id for r in records[:20]], 'Navigate to a blue cup.')
    frames = {n for e in evidence for n in e['frame_indices']}
    assert 70 in frames
    assert len(frames) <= 32 and len(evidence) <= 20
    assert len(json.dumps(evidence, ensure_ascii=False).encode()) <= 40000
    assert audit['certified_event_membership'] is False
    assert before == [r.model_dump(mode='json') for r in records]
    rows = {row[0] for e in evidence for row in e['content']['frame_rows']}
    assert frames == rows
    assert all(not e['content'].get('trajectory') for e in evidence)


def test_cross_episode_identity_is_not_an_expansion_edge():
    local = [frame_record('ep_a', n, identity='cup') for n in range(5)]
    foreign = [frame_record('ep_b', 80, identity='cup')]
    evidence, _, _ = project_retrieval(local+foreign, [local[0].id], 'Navigate to a cup.')
    assert 80 not in {n for e in evidence for n in e['frame_indices']}
    with pytest.raises(ValueError, match='namespace'):
        WitnessIndex(local+foreign)


def test_unknown_plan_references_do_not_crash_and_valid_targets_are_remapped():
    records = [frame_record('ep', n) for n in range(15)]
    resolution = {'complete': False, 'record_ids': ['missing',records[0].id],
        'anchor_record_ids': ['missing'], 'target_record_ids': [records[0].id],
        'targets': [{'record_id': records[0].id, 'frame_index': 0, 'entity_id': 'cup'}]}
    evidence, revised, _ = project_retrieval(records, [records[0].id], 'Navigate to a cup.', resolution)
    assert revised['targets'][0]['record_id'] in {e['id'] for e in evidence}
    assert revised['targets'][0]['frame_index'] == 0


def test_views_keep_original_observation_objects_and_validate():
    records = [frame_record('ep', n, identity=f'object_{n}') for n in range(12)]
    index = WitnessIndex(records)
    evidence, _, _ = index.project([records[0].id], 'Navigate to the second object you interacted with.')
    original = {o.id: o for r in records for o in r.observations}
    assert len(original) == 12
    # Serialized rows remain original facts, not invented future observations.
    for e in evidence:
        for row in e['content']['frame_rows']:
            source = json.loads(index.originals[row[0]][0].value)
            assert row[1] == source['clock']
            assert source['description'].startswith(row[2].rstrip(' …'))
    assert len({n for e in evidence for n in e['frame_indices']}) == 12


def test_image_selector_uses_only_retrieved_originals_and_is_capped():
    records = [frame_record('ep', n, identity=f'item_{n}') for n in range(50)]
    index = WitnessIndex(records)
    evidence, resolution, audit = index.project([records[0].id], 'Navigate to a cup.')
    observations = list(index.frames.values())
    images = select_images_v5(evidence, observations, resolution)
    assert len(images) == len(set(images)) == 12
    assert set(images) <= {n for e in evidence for n in e['frame_indices']}
    assert images == audit['image_priority'][:12]
    assert select_images_v5(evidence, observations, resolution, limit=0) == []


def test_retrieval_interface_accepts_no_labels_categories_or_previous_answers():
    for fn in (project_retrieval, WitnessIndex.rank, WitnessIndex.project):
        assert not any(word in name for name in inspect.signature(fn).parameters
                       for word in ('gold','category','row_idx','answer','correct'))


def test_rank_is_deterministic_for_the_same_question_and_history():
    records = [frame_record('ep', n) for n in range(50)]
    index = WitnessIndex(records)
    first = index.rank([records[0].id], 'Navigate to a cup.')
    assert first == index.rank([records[0].id], 'Navigate to a cup.')
    assert len(first[0]) == LIMITS['candidate_frames']


def test_condition_location_is_not_mistaken_for_the_requested_object():
    root = frame_record('ep', 10, identity='cup')
    observation = root.observations[0]
    event = root.model_copy(update={
        'entities': [*root.entities, MemoryEntity(id='cabinet',name='cabinet',kind='place',namespace='ep')],
        'role_bindings': [*root.role_bindings,
            RoleBinding(role='object',entity_id='cup',observation_ids=[observation.id],confidence=.95),
            RoleBinding(role='source',entity_id='cabinet',observation_ids=[observation.id],confidence=.95)],
        'evolution': Evolution(namespace='ep',case_id='case',stages=[Stage(id='stage',phase='grasp',
            at=observation.observed_at,state={'object_id':'cup'},observation_ids=[observation.id])])})
    index = WitnessIndex([event,frame_record('ep',11,identity='cup')])
    resolution = {'complete':True,'targets':[{'entity_id':'cabinet','record_id':event.id,'frame_index':10}]}
    _, revised, audit = index.project([event.id], 'Navigate to the object you picked from the cabinet.', resolution)
    assert audit['query']['answer_role'] == 'object'
    assert audit['query']['linked_role_entity_ids'] == ['cup']
    assert revised['complete'] is False and revised['targets'] == []


def test_answerer_preserves_common_payload_and_validation_when_image_order_is_the_same(tmp_path):
    from benchmarks import hypergraph_visual_answer_v5 as v5
    from benchmarks import findingdory_visual_answer as original
    from memory_agent.settings import Settings
    observations = [json.loads(frame_record('ep',0).observations[0].value)]
    folder = tmp_path/'frames/ep'
    folder.mkdir(parents=True)
    (folder/'000.jpg').write_bytes(b'mocked-image')
    evidence = [{'id':'fact','content':'cup is visible','frame_indices':[0]}]
    class Client:
        settings = Settings(llm_key='fake',llm_model='mock')
        def __init__(self):self.payloads=[]
        def chat_options(self,model):return {}
        def post(self,url,key,payload,role):
            self.payloads.append(payload)
            return {'choices':[{'message':{'content':json.dumps({'goal_frames':[0],'evidence_ids':['fact']})}}]}
    a,b=Client(),Client()
    assert v5.decode_visual(a,'Navigate to a cup.',evidence,observations,tmp_path,'ep') == original.decode_visual(
        b,'Navigate to a cup.',evidence,observations,tmp_path,'ep')
    assert a.payloads == b.payloads


def test_explicit_observed_clock_is_indexed_without_synthesizing_missing_times():
    records = [frame_record('ep',n) for n in range(60)]
    root = frame_record('ep',90)
    observation = root.observations[0]
    frame = json.loads(observation.value)
    frame['clock'] = '07:17'
    root = root.model_copy(update={'observations':[observation.model_copy(update={'value':json.dumps(frame)})]})
    index = WitnessIndex([*records,root])
    ranking,_ = index.rank([r.id for r in records[:20]], 'Navigate to the object you interacted with at 7:17 yesterday.')
    assert 90 in ranking[:12]
    assert index.clock_incidence['07:17'] == {90}
