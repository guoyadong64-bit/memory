"""Task-defined answer normalization, independent of reference answers."""
import re
from memory_agent.recall import parse_plan


def select_any_target(output,question,evidence):
    """An explicit ANY-one question accepts one witnessed candidate, not a list.

    Choose the first citable candidate in model order. No reference frames,
    categories, scoring or correctness flags are consulted here.
    """
    plan=parse_plan(question)
    goals=output.get('goal_frames')
    if (plan.all_targets or plan.visit_order or not re.search(r'\bany\b|任意',question.lower())
            or not isinstance(goals,list) or len(goals)<2):
        return output,None
    known={item['id']:set(item['frame_indices']) for item in evidence}
    ids=output.get('evidence_ids',[])
    if not isinstance(ids,list) or any(not isinstance(i,str) or i not in known for i in ids):
        return output,None
    supported=set().union(*(known[i] for i in ids)) if ids else set()
    chosen=next((f for f in goals if type(f) is int and f in supported),None)
    if chosen is None:return output,None
    bindings=output.get('target_bindings',[])
    if not isinstance(bindings,list):return output,None
    normalized=output | {'goal_frames':[chosen]}
    if bindings:
        binding=next((b for b in bindings if isinstance(b,dict) and b.get('frame_index')==chosen),None)
        if binding is None:return output,None
        normalized['target_bindings']=[binding]
    return normalized,{'policy':'explicit ANY-one: first citable model candidate',
        'original_output':output,'selected_frame':chosen}
