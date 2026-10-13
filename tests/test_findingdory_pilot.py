from benchmarks.findingdory_pilot import goal_indices, observation_records, score_frames, task_query
from memory_agent.memory import HypergraphMemory


def test_frame_membership_requires_correct_goal_count_and_order():
    assert score_frames([2, 7], [[1, 2, 3], [7, 8]])
    assert not score_frames([7, 2], [[1, 2, 3], [7, 8]])
    assert not score_frames([2], [[1, 2, 3], [7, 8]])
    assert not score_frames([], [[-1]])


def test_unordered_goals_use_one_to_one_matching():
    assert score_frames([7, 2], [[1, 2, 3], [7, 8]], unordered=True)
    assert not score_frames([2, 2], [[1, 2, 3], [7, 8]], unordered=True)
    # Greedy assignment would fail, but a valid perfect matching exists.
    assert score_frames([2, 3], [[2, 3], [2]], unordered=True)


def test_output_parser_rejects_invalid_or_out_of_range_frames():
    assert goal_indices('{"goal_frames":[0,95]}') == [0, 95]
    assert goal_indices('```json\n{"goal_frames":[4]}\n```') == [4]
    for text in ['{"goal_frames":[true]}', '{"goal_frames":[96]}',
                 '{"goal_frames":[-1]}', '{"goal_frames":[1.5]}', 'unsupported']:
        assert goal_indices(text) == []


def test_observation_mapping_does_not_invent_actions_or_wall_clock():
    records = observation_records('ep_1', [{"frame_index": 9,
        "description": "A cup is visible on a table.", "entities": ["cup", "table"],
        "event": "scene_observation", "action": None, "outcome": None, "clock": None}])
    record = records[0]
    assert record.action is None and record.outcome is None
    assert record.time.event_at.year == 2000
    assert record.time.applicable_until is None
    assert record.space.namespace == 'ep_1'
    assert record.id == 'ep_1_frame_9'


def test_adapter_supplies_scope_required_by_existing_memory_gates(tmp_path):
    records = observation_records('ep_1', [{"frame_index": 9,
        "description": "A cup is visible on a table.", "entities": ["cup", "table"],
        "event": "scene_observation", "action": None, "outcome": None, "clock": None}])
    store = HypergraphMemory(tmp_path / 'memory.sqlite3')
    store.add(records[0])
    query = task_query('Navigate to a cup.', 'ep_1')
    for strategy in ('flat', 'hypergraph'):
        rows, report = store.search(query, strategy=strategy)
        assert report['eligible_count'] == 1
        assert [r.id for r in rows] == ['ep_1_frame_9']
    wrong = task_query('Navigate to a cup.', 'ep_2')
    assert not store.search(wrong)[0]
