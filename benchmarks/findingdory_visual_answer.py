"""Common, evidence-bounded visual answer verification for both memory methods."""
import base64
import json
from pathlib import Path

from benchmarks.findingdory_mem0 import ANSWER_SYSTEM, answer_request
from benchmarks.findingdory_pilot import goal_indices
from benchmarks.answer_normalization import select_any_target
from memory_agent.clients import ModelError
from memory_agent.recall import parse_plan


VISUAL_ANSWER_SYSTEM = ANSWER_SYSTEM + """
Retrieved names, phases and resolution warnings are model hypotheses, not authority.
Use the attached original images to verify or reject them. The question can help
identify an ambiguous category but never establishes an unseen action by itself.
Only attached/retrieved frames are eligible, with memory evidence IDs supporting them.
A robot's mechanical cylinders, blocks and attached finger plates are its own body,
not external objects. Look between the fingers for a manipulated item.
For an interaction receptacle, distinguish the surface where the robot approached,
lifted or released the item from background furniture. If the release is occluded,
use the visible before/after sequence to assess the location, preserving uncertainty;
do not demand a fully visible final placement when the interaction surface is clear.
A woven supporting seat and an open storage container have different roles; verify
the picture rather than trusting a guessed 'box' or 'stool' label.
For any untouched object, select a clear, identifiable object view, avoiding tiny
fragments, empty floors and the robot's own parts. A before/after object match needs
visual appearance evidence; do not merge all rectangles or all green toys.
Do not invent metric distances or current pose from uncalibrated pictures.
JSON must contain goal_frames and evidence_ids. Extra fields are ignored by scoring.
Before selecting frames, build an observed event inventory: manipulated item,
source, destination, first/last witnessed frame, and uncertainty. Category names
can be revised from the images; never change an instance ID merely to match a noun.
For temporal offsets count interactions, not frames or approach-only fragments.
For ALL queries enumerate distinct physical targets, then apply the requested
permutation. Several placements onto the same support are ONE target.
Also return target_bindings: one {frame_index, entity_id, evidence_ids} per goal.
Use an existing entity ID when known; use null when identity is genuinely unknown.
Do not invent a new entity ID to evade deduplication. If a requested permutation
has five positions, a one-target list is incomplete and must be reconsidered.
"""


def select_images(evidence, observations, resolution=None, limit=12):
    """Pick witnesses only from the method's available frame set, without labels."""
    allowed = {n for item in evidence for n in item["frame_indices"]}
    by_frame = {frame["frame_index"]:frame for frame in observations}
    required = [t["frame_index"] for t in (resolution or {}).get("targets", [])
                if t.get("frame_index") in allowed]
    if limit < 1:
        return []
    # Group by memory episode whenever available. Model frame-level interaction
    # IDs frequently change between grasp/transport/release of the SAME episode.
    groups = {}
    grouped = set()
    for item in evidence:
        content = item.get("content")
        if isinstance(content,dict) and content.get("trajectory"):
            indices = sorted(set(item["frame_indices"]) & allowed)
            if indices:
                groups[item["id"]] = indices
                grouped.update(indices)
    for index in sorted(allowed):
        frame = by_frame.get(index,{})
        interaction = frame.get("interaction") or {}
        if interaction and index not in grouped:
            groups.setdefault(interaction["id"], []).append(index)
    # Reserve one witness per event FIRST, then second witnesses. Global uniform
    # thinning used to drop a whole event or the identifiable pre-release object.
    ordered = list(dict.fromkeys(required))[:limit]
    for boundary in (0,-1):
        for indices in groups.values():
            if indices[boundary] not in ordered and len(ordered) < limit:
                ordered.append(indices[boundary])
    remaining = [index for index in sorted(allowed) if index not in ordered]
    # Spread otherwise unstructured evidence through time, not just the last view.
    if remaining:
        slots = max(0,limit-len(ordered))
        ordered.extend(remaining[round(k*(len(remaining)-1)/max(1,slots-1))] for k in range(slots))
    # Many incomplete events must not crowd out all later interactions.
    return list(dict.fromkeys(ordered))[:limit]


def answer_errors(output, evidence, question, resolution=None, identity_ids=None):
    """Check task constraints without consulting evaluation annotations."""
    predicted = goal_indices(json.dumps(output))
    known = {item["id"]:item for item in evidence}
    ids = output.get("evidence_ids",[])
    errors = []
    if predicted and (not isinstance(ids,list) or not ids or
            any(not isinstance(i,str) or i not in known for i in ids) or
            not set(predicted) <= {f for i in ids if i in known for f in known[i]["frame_indices"]}):
        errors.append("missing_or_invalid_frame_provenance")
    plan = parse_plan(question)
    if plan.visit_order and len(predicted) != len(plan.visit_order):
        errors.append("requested_target_count_not_met")
    if (not plan.all_targets and not plan.visit_order and len(predicted)>1 and
            (plan.goal in {"entity","source","destination","uninteracted"})):
        errors.append("single_target_query_returned_multiple_goals")
    bindings = output.get("target_bindings",[])
    if bindings:
        if not isinstance(bindings,list) or any(not isinstance(b,dict) for b in bindings):
            errors.append("invalid_target_bindings")
        elif [b.get("frame_index") for b in bindings] != predicted:
            errors.append("target_bindings_do_not_match_goal_frames")
        else:
            identities = [b.get("entity_id") for b in bindings if b.get("entity_id") is not None]
            if any(not isinstance(i,str) for i in identities) or len(set(identities)) != len(identities):
                errors.append("duplicate_or_invalid_physical_target_identity")
            if identity_ids is not None and any(i not in identity_ids for i in identities if isinstance(i,str)):
                errors.append("undeclared_physical_target_identity")
            for b in bindings:
                refs = b.get("evidence_ids",[])
                if not isinstance(refs,list) or not refs or any(not isinstance(i,str) or i not in known for i in refs):
                    errors.append("invalid_target_binding_provenance")
                    break
                if b.get("frame_index") not in {f for i in refs for f in known[i]["frame_indices"]}:
                    errors.append("invalid_target_binding_provenance")
                    break
    if plan.all_targets and not predicted:
        errors.append("no_supported_targets_returned")
    if (plan.entity_hint or plan.anchor_hint) and not predicted:
        errors.append("identity_or_temporal_anchor_unresolved")
    return list(dict.fromkeys(errors))


def decode_visual(client, question, evidence, observations, data, episode, resolution=None):
    if not evidence:
        return {"goal_frames":[],"evidence_ids":[],"visual_frames":[]}
    indices = select_images(evidence,observations,resolution)
    state = answer_request(question,evidence,resolution) | {"attached_visual_frames":indices}
    catalog = {}
    for frame in observations:
        if frame["frame_index"] in indices:
            for entity in frame.get("entities",[]):
                catalog.setdefault(entity["id"],entity)
    state["observed_entity_catalog"] = list(catalog.values())
    content = [{"type":"text","text":json.dumps(state,ensure_ascii=False)}]
    for index in indices:
        path = Path(data)/"frames"/episode/f"{index:03d}.jpg"
        content.extend([{"type":"text","text":f"Original evidence image: frame_index={index}"},
            {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+
                base64.b64encode(path.read_bytes()).decode(),"detail":"high"}}])
    model = client.settings.answer_model or client.settings.llm_model
    repairs = []
    normalizations = []
    for attempt in range(2):
        request_content = content
        if repairs:
            request_content = content + [{"type":"text","text":json.dumps({
                "repair_required":repairs[-1]["errors"],"previous_output":repairs[-1]["output"],
                "instruction":"Recheck the same evidence. Return a complete supported set; do not invent missing goals or identities."})}]
        response = client.post(client.settings.llm_url.rstrip("/")+"/chat/completions",client.settings.llm_key,
            {"model":model,"temperature":0,"max_tokens":3072,"response_format":{"type":"json_object"},
             **client.chat_options(model),"messages":[{"role":"system","content":VISUAL_ANSWER_SYSTEM},
                                                     {"role":"user","content":request_content}]},"answer")
        try:
            output = json.loads(response["choices"][0]["message"]["content"])
            if not isinstance(output,dict):raise ValueError()
        except (ValueError,KeyError,IndexError,TypeError):
            raise ModelError("answer: invalid visual structured answer") from None
        output,normalization = select_any_target(output,question,evidence)
        if normalization is not None:normalizations.append(normalization)
        errors = answer_errors(output,evidence,question,resolution,set(catalog))
        if not errors:break
        repairs.append({"errors":errors,"output":output})
    predicted = goal_indices(json.dumps(output))
    ids = output.get("evidence_ids",[])
    known = {item["id"]:item for item in evidence}
    valid = (predicted and isinstance(ids,list) and ids and
        all(isinstance(identity,str) and identity in known for identity in ids) and
        set(predicted) <= {n for identity in ids for n in known[identity]["frame_indices"]})
    if not valid or errors:
        return {"goal_frames":[],"evidence_ids":[],"visual_frames":indices,
                "validation":errors or ["no_supported_answer"],"raw_output":output,"repairs":repairs,
                "normalizations":normalizations}
    return {"goal_frames":predicted,"evidence_ids":ids,"visual_frames":indices,
            "target_bindings":output.get("target_bindings",[]),"repairs":repairs,
            "normalizations":normalizations}
