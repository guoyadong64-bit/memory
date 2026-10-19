"""Historical constraints over evidence, independent of dataset labels and provider APIs."""
import json
import re
from datetime import timedelta

from .models import RecallPlan
from .identity import direct_lift


ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}


def parse_plan(question):
    """Conservative language fallback, not a lookup table of task/answer pairs."""
    q = question.lower().strip().rstrip(".")
    goal = "source" if re.search(r"pick(?:ed)? .*?from|picked an object from|取出|拿起.*从", q) else "entity"
    if re.search(r"plac(?:ed|e) .*?(?:on|in)|放置|放到", q):
        goal = "destination"
    if "did not interact" in q or "未交互" in q:
        goal = "uninteracted"
    plan = RecallPlan(goal=goal, all_targets=bool(re.search(r"\ball\b|所有", q)),
        previous_day="yesterday" in q or "昨天" in q,
        requires_geometry=bool(re.search(r"farthest|nearest|closest|最远|最近", q)))
    match = re.search(r"(?:after|before)(?: ending| you finished| you finish)?(?: the)? interaction(?: with)?\s+(.+)", q)
    if not match:
        match = re.search(r"(?:after|before).*?interacting with\s+(.+)", q)
    if match:
        plan.anchor_hint = match.group(1).strip()
        plan.relation = "before" if "before" in match.group(0) else "after"
        offset = re.search(r"\b(\d+) interactions? (?:after|before)", q)
        if offset:
            plan.offset = int(offset.group(1))
    if plan.relation == "any":
        for word, n in ORDINALS.items():
            if re.search(r"\bthe " + word + r"\b", q):
                plan.ordinal = n
                break
    order = re.search(r"order to revisit them is:\s*(.+)", q)
    if order:
        plan.visit_order = [ORDINALS[w] if w in ORDINALS else int(w) for w in
            re.findall(r"\b(?:" + "|".join(ORDINALS) + r"|\d+)\b", order.group(1))]
    if goal == "entity" and not plan.anchor_hint:
        obj = re.search(r"navigate to (?:a |an |the )?(.+)", q)
        if obj and not re.search(r"object.*interact|receptacle", obj.group(1)):
            plan.entity_hint = obj.group(1)
    location = re.search(r"(?:all the|to a|to the)\s+([a-z]+)\s+(?:you picked|you placed|that you picked|that you placed)", q)
    if location and location.group(1) not in {"object", "objects", "receptacle", "receptacles"}:
        plan.location_hint = location.group(1)
    return plan


def payload(record):
    try:
        value = json.loads(record.answer)
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def frames(record):
    value = payload(record)
    if "frame_index" in value:
        return [value]
    return value.get("observations", [])


def words(value):
    result = {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
              if w not in {"the", "a", "an", "object", "objects", "you", "robot"}}
    # A tabletop is a surface on a table; a desk is a distinct furniture type.
    # Treating every desk as a table creates false source/destination matches.
    if "tabletop" in result:
        result.add("table")
    for run in re.findall(r"[\u4e00-\u9fff]+", value):
        result.update(run[i:i+2] for i in range(max(1, len(run)-1)))
    return result


def hint_score(hint, identity, name="", description=""):
    if not hint:
        return 1.0
    wanted = words(hint)
    if not wanted:
        return 0.0
    named = words(identity + " " + name)
    exact = len(wanted & named) / len(wanted)
    # Descriptions are only supplied for the explicitly manipulated object.
    described = .75 * len(wanted & words(description)) / len(wanted) if description else 0
    return max(exact, described)


def catalog(records):
    entities = {}
    for record in records:
        for f in frames(record):
            interaction = f.get("interaction") or {}
            for e in f.get("entities", []):
                entry = entities.setdefault(e["id"], {"id": e["id"], "name": e["name"], "kind": e["kind"], "examples": []})
                if interaction.get("object_id") == e["id"] and len(entry["examples"]) < 2:
                    entry["examples"].append(f.get("description", "")[:220])
    return list(entities.values())


def compact_record(record):
    """Project a role/evidence subgraph without repeating each frame's identity catalog."""
    data=payload(record)
    visual=frames(record)
    if visual:
        observation_ids = {}
        for observation in record.observations:
            value = observation_frame(observation)
            if value:
                observation_ids.setdefault(value["frame_index"], []).append(observation.id)
        data={k:v for k,v in data.items() if k not in {"observations","entities","description","relations","interaction"}}
        data["observations"]=[{k:f.get(k) for k in ("frame_index","clock","description","relations","interaction","uncertainty")}
                              | {"visible_entity_ids":[e["id"] for e in f.get("entities",[])],
                                 "observation_ids":observation_ids.get(f["frame_index"], [])} for f in visual]
    roles={}
    for binding in record.role_bindings:
        key=(binding.role,binding.entity_id)
        item=roles.setdefault(key,{"role":binding.role,"entity_id":binding.entity_id,
            "confidence":binding.confidence,"basis":binding.basis,
            "observation_ids":[],"stage_ids":[]})
        item["confidence"]=min(item["confidence"],binding.confidence)
        if binding.basis == "linked":item["basis"] = "linked"
        item["observation_ids"].extend(binding.observation_ids)
        if binding.stage_id:item["stage_ids"].append(binding.stage_id)
    for item in roles.values():
        item["observation_ids"]=list(dict.fromkeys(item["observation_ids"]))
        item["stage_ids"]=list(dict.fromkeys(item["stage_ids"]))
    return {"id": record.id, "source": record.source, "event_at":
        record.time.event_at.isoformat() if record.time and record.time.event_at else None,
        "evidence": data or record.answer,
        "trajectory": [{"at": s.at.isoformat(), "phase": s.phase, "state": s.state,
                        "action": s.action, "outcome": s.outcome,
                        "observation_ids": s.observation_ids} for s in record.evolution.stages]
                       if record.evolution else [],
        "entities": [e.model_dump(mode="json") for e in record.entities],
        "roles": list(roles.values()),
        "interpretation": "model-derived evidence, not independent visual truth"}


def focused_evidence(item, targets):
    """Project the smallest witnessed stages needed for a resolved target.

    No target is derived here: targets already come from the memory resolver.
    Raw observations and IDs remain available for checking the selected subgraph.
    """
    content = item.get("content")
    if not isinstance(content, dict) or not targets:
        return item
    data = content.get("evidence")
    if not isinstance(data, dict) or not isinstance(data.get("observations"), list):
        return item
    required = {oid for target in targets for oid in target.get("observation_ids", [])}
    target_frames = {target["frame_index"] for target in targets if target.get("frame_index") is not None}
    trajectory = content.get("trajectory", [])
    # Retain local source evidence plus the first available movement witness.
    if any(t["role"] == "source" for t in targets):
        for stage in trajectory:
            if stage.get("phase") in {"transport", "release", "settled"}:
                required.update(stage.get("observation_ids", []))
                break
    selected = [f for f in data["observations"] if f.get("frame_index") in target_frames or
                set(f.get("observation_ids", [])) & required]
    if not selected:
        return item
    selected_frames = {f["frame_index"] for f in selected}
    available_ids = {oid for frame in selected for oid in frame.get("observation_ids", [])}
    roles = [r | {"observation_ids": [oid for oid in r.get("observation_ids", []) if oid in available_ids]}
             for r in content.get("roles", []) if set(r.get("observation_ids", [])) & available_ids]
    projected = content | {
        "evidence": data | {"observations": selected, "frame_indices": sorted(selected_frames)},
        "trajectory": [s for s in trajectory if set(s.get("observation_ids", [])) & available_ids],
        "roles": roles,
        "projection": {"kind": "target_evidence_subgraph", "original_frame_count": len(item["frame_indices"]),
                       "retained_frame_count": len(selected_frames)}}
    # The legacy identity-name map is duplicated by the typed entity catalog.
    projected["evidence"] = {k:v for k,v in projected["evidence"].items()
                             if k not in {"entities","grouping_policy","raw_interaction_ids","interaction_id"}}
    return item | {"content": projected, "frame_indices": sorted(selected_frames)}


def pack_evidence(evidence, resolution=None, max_bytes=40000, require_all_targets=True):
    """Reserve proof for resolved targets before filling remaining context.

    Oversized records never stop consideration of later, smaller evidence. The
    same byte cap is used for fact-text baselines. Missing target proofs make the
    returned resolution explicitly incomplete instead of silently dropping them.
    """
    if max_bytes < 2:
        raise ValueError("Context budget must fit a JSON list")
    targets = (resolution or {}).get("targets", [])
    by_record = {}
    for target in targets:
        by_record.setdefault(target["record_id"], []).append(target)
    priority_ids = list(dict.fromkeys([*by_record, *((resolution or {}).get("anchor_record_ids", [])),
                                     *((resolution or {}).get("record_ids", []))]))
    priority = {identity:index for index, identity in enumerate(priority_ids)}
    ordered = sorted(enumerate(evidence), key=lambda pair:(priority.get(pair[1]["id"], len(priority)), pair[0]))
    bounded, dropped = [], []
    for _, item in ordered:
        projected = focused_evidence(item, by_record[item["id"]]) if item["id"] in by_record else item
        if item["id"] not in by_record and item["id"] in (resolution or {}).get("record_ids", []):
            content = item.get("content")
            trajectory = content.get("trajectory", []) if isinstance(content,dict) else []
            if trajectory:
                # Preserve both sides of every observed phase, not just the whole
                # event's endpoints. A grasp or release in the middle is essential
                # to reconstruct roles and must survive context compression.
                phases = {}
                for stage in trajectory:
                    phases.setdefault(stage["phase"], []).append(stage)
                boundaries = [stage for group in phases.values() for stage in (group[0],group[-1])]
                ids = {oid for stage in boundaries for oid in stage.get("observation_ids", [])}
                projected = focused_evidence(item,[{"record_id":item["id"],"role":"object",
                    "observation_ids":list(ids),"frame_index":None}])
                if projected is not item:
                    view = projected["content"]
                    focus = {role["entity_id"] for role in view.get("roles", [])
                             if role["role"] in {"object","actor","source","destination"}}
                    observations = view["evidence"]["observations"]
                    # Keep direct participant-to-location edges; unrelated scenery
                    # belongs in the original frame, not every event's name catalog.
                    neighbors = {rel["target"] for frame in observations for rel in frame.get("relations", [])
                                 if rel.get("source") in focus}
                    focus |= neighbors
                    if focus:
                        observations = [frame | {
                            "relations":[rel for rel in frame.get("relations", [])
                                         if rel.get("source") in focus and rel.get("target") in focus],
                            "visible_entity_ids":[eid for eid in frame.get("visible_entity_ids", []) if eid in focus]}
                            for frame in observations]
                        projected = projected | {"content":view | {
                            "entities":[entity for entity in view.get("entities", []) if entity["id"] in focus],
                            "roles":[role for role in view.get("roles", []) if role["entity_id"] in focus],
                            "evidence":view["evidence"] | {"observations":observations}}}
        if item["id"] in by_record and projected is not item and projected["content"].get("trajectory"):
            view = projected["content"]
            focus = {role["entity_id"] for role in view.get("roles", [])
                     if role["role"] in {"object","actor","source","destination"}}
            focus.update(t["entity_id"] for t in by_record[item["id"]] if t.get("entity_id"))
            focus |= {rel["target"] for frame in view["evidence"]["observations"] for rel in frame.get("relations", [])
                      if rel.get("source") in focus}
            if focus:
                projected = projected | {"content":view | {
                    "entities":[entity for entity in view.get("entities", []) if entity["id"] in focus]}}
        if projected is not item:
            view = projected["content"]
            namespaces = {entity.get("namespace") for entity in view.get("entities", [])}
            if len(namespaces) == 1 and None not in namespaces:
                # One namespace applies to the whole projected identity catalog.
                projected = projected | {"content":view | {"entity_namespace":next(iter(namespaces)),
                    "entities":[{k:v for k,v in entity.items() if k != "namespace"}
                                for entity in view["entities"]]}}
            # Compact label provenance without throwing away alternate category
            # hypotheses or witnessed roles. Detailed observation data stays in
            # SQLite; this is only the answering projection of the same subgraph.
            view = projected["content"]
            projected = projected | {"content":view | {
                "entities":[{k:v for k,v in entity.items() if k in {"id","name","kind","labels","attributes"}}
                            for entity in view.get("entities", [])]}}
        if len(json.dumps(bounded + [projected], ensure_ascii=False).encode()) <= max_bytes:
            bounded.append(projected)
        else:
            dropped.append(item["id"])
    revised = None
    if resolution is not None:
        revised = resolution.copy()
        kept = {item["id"]:item for item in bounded}
        retained = [t for t in targets if t["record_id"] in kept and
                    t.get("frame_index") in kept[t["record_id"]]["frame_indices"]]
        revised["targets"] = retained
        missing = ((len(retained) != len(targets) if require_all_targets else bool(targets) and not retained)
                   or any(i not in kept for i in revised.get("anchor_record_ids", [])))
        if missing:
            revised["complete"] = False
            revised["warnings"] = [*resolution.get("warnings", []), "Context budget omitted required target or anchor evidence"]
        revised["context_dropped_record_ids"] = dropped
    return bounded, revised


def entity_name(record, identity):
    entity = next((e for e in record.entities if e.id == identity), None)
    if entity:
        return " ".join([entity.name, *[x.text for x in entity.labels if x.confidence >= .8]])
    names = payload(record).get("entities", {})
    if isinstance(names, dict):
        return names.get(identity, "")
    return next((e["name"] for e in names if e["id"] == identity), "")


def observation_frame(observation):
    if not isinstance(observation.value,str):
        return None
    try:
        value=json.loads(observation.value)
    except (ValueError,TypeError):
        return None
    return value if isinstance(value,dict) and type(value.get("frame_index")) is int else None


def pickup_departure(record, stage, source_id):
    """Certify local lifting or later movement, independently of final placement."""
    subject=stage.state.get("object_id")
    if stage.state.get("pickup_evidence") == "direct_lift":
        for observation in record.observations:
            if observation.id not in stage.observation_ids:
                continue
            frame = observation_frame(observation)
            if frame and (frame.get("interaction") or {}).get("object_id") == subject and direct_lift(frame):
                return True
    return any(s.at>=stage.at and s.state.get("object_id")==subject and (
        (s.phase=="transport" and s.state.get("held_by")) or
        (s.phase in {"release","settled"} and s.state.get("target_id") and
         s.state["target_id"]!=source_id and s.state["target_id"] in
         {s.state.get("on"),s.state.get("inside")})) for s in record.evolution.stages)


def target_groundings(record, goal, location_hint=None):
    """One explicitly bound role per event/physical target, not nearby mentions."""
    role_name = goal if goal in {"source", "destination"} else "object"
    result = []
    seen = set()
    roles = record.role_bindings
    if role_name == "object" and record.evolution:
        stage_index = {s.id:s for s in record.evolution.stages}
        roles = sorted(roles, key=lambda b:(
            stage_index[b.stage_id].phase in {"release", "settled"}, stage_index[b.stage_id].at)
            if b.stage_id in stage_index else (False,record.created_at), reverse=True)
    for role in roles:
        if role.role != role_name or role.confidence < .8 or role.entity_id in seen:
            continue
        if location_hint and hint_score(location_hint, role.entity_id, entity_name(record,role.entity_id)) < .5:
            continue
        stage = next((s for s in record.evolution.stages if s.id == role.stage_id), None) if record.evolution else None
        if role_name == "source" and (not stage or stage.phase not in {"grasp", "transport"}
                or not stage.state.get("held_by") or stage.state.get("source_id") != role.entity_id):
            continue
        if role_name == "source" and not pickup_departure(record,stage,role.entity_id):
            continue
        if role_name == "destination" and (not stage or stage.phase not in {"release", "settled"}
                or stage.state.get("target_id") != role.entity_id
                or role.entity_id not in {stage.state.get("on"),stage.state.get("inside")}):
            continue
        stage_observations = stage.observation_ids if stage else role.observation_ids
        available = {o.id:o for o in record.observations}
        index = next((payload_frame.get("frame_index") for oid in stage_observations
            if oid in available and (payload_frame := observation_frame(available[oid])) is not None), None)
        result.append({"record_id":record.id,"entity_id":role.entity_id,"role":role_name,
                       "frame_index":index,"observation_ids":role.observation_ids})
        seen.add(role.entity_id)
    return result


def visible_targets(records, plan, untouched=False):
    touched = {b.entity_id for r in records for b in r.role_bindings if b.role == "object" and b.confidence >= .8}
    attempted = {i.get("object_id") for r in records for f in frames(r)
                 if (i := f.get("interaction")) and i.get("confidence",0) >= .8}
    targets = {}
    for record in records:
        entities = {e.id:e for e in record.entities}
        for role in record.role_bindings:
            entity = entities[role.entity_id]
            if role.role != "visible" or entity.kind != "object" or role.confidence < .8:
                continue
            if entity.attributes.get("role") == "robot_component":
                continue
            partial = entity.attributes.get("visibility") in {"partial","occluded"} or "partial" in words(entity.id)
            if untouched and (entity.id in touched or entity.id in attempted or partial):
                continue
            score = hint_score(plan.entity_hint, entity.id, entity_name(record,entity.id))
            if plan.entity_hint and score < .5:
                continue
            label_score = max((l.confidence for l in entity.labels if l.confidence >= .8), default=.8)
            value = payload(record)
            index = value.get("frame_index")
            if type(index) is not int:
                continue
            candidate = {"record_id":record.id,"entity_id":entity.id,"role":"visible",
                         "frame_index":index,"observation_ids":role.observation_ids}
            # Prefer explicit identity evidence and a relation-rich view; tie by time.
            # A small item resting on another object is more useful to navigate to
            # than flooring/furniture merely resting on the ground. Visibility and
            # an explicit affordance take precedence over "latest frame wins".
            resting = any(rel.get("source")==entity.id and rel.get("predicate") in {"on","inside"}
                          and rel.get("confidence",0)>=.8 and rel.get("target") in entities
                          and (entities[rel["target"]].kind == "object" or
                               entities[rel["target"]].attributes.get("role") == "support")
                          for rel in value.get("relations",[]))
            movable = entity.attributes.get("role") == "movable"
            clarity = 0 if partial else 2 if entity.attributes.get("visibility") == "clear" else 1
            quality = (score, movable, clarity, resting, label_score, index)
            if entity.id not in targets or quality > targets[entity.id][0]:
                targets[entity.id] = (quality, candidate)
    return [v[1] for v in sorted(targets.values(),key=lambda value:value[0],reverse=True)]


def record_match(record, plan):
    if record.role_bindings and plan.goal in {"source", "destination"}:
        return max((hint_score(plan.location_hint, t["entity_id"], entity_name(record,t["entity_id"]))
                    for t in target_groundings(record, plan.goal, plan.location_hint)), default=0)
    checks = []
    data = frames(record)
    if plan.entity_hint:
        checks.append(max((hint_score(plan.entity_hint, e["id"], entity_name(record,e["id"]) or e["name"])
            for f in data for e in f.get("entities", []) if e["kind"] == "object"), default=0))
    if plan.goal in {"source", "destination"}:
        valid = []
        for f in data:
            i = f.get("interaction") or {}
            if i.get("confidence", 0) < .8:
                continue
            target = i.get("source_id" if plan.goal == "source" else "target_id")
            if target == i.get("object_id"):
                target = None
            phase = i.get("phase")
            action_ok = phase in {"grasp", "transport"} if plan.goal == "source" else phase in {"release", "settled"}
            if plan.goal == "source":
                # A nearby gripper or a model-inferred source does not prove a pickup.
                # Require visible object -> actor held_by evidence in that grasp frame.
                actor = i.get("actor_id")
                held = any(rel.get("source") == i.get("object_id") and
                           rel.get("predicate") == "held_by" and
                           rel.get("confidence", 0) >= .8 and
                           (not actor or rel.get("target") == actor)
                           for rel in f.get("relations", []))
                action_ok = action_ok and held
            if action_ok and target:
                named = next((e["name"] for e in f.get("entities", []) if e["id"] == target), "")
                valid.append(hint_score(plan.location_hint, target, named))
        # Linked source/destination evidence can be expressed in stage state.
        if record.evolution:
            field = "source_id" if plan.goal == "source" else "target_id"
            names = payload(record).get("entities", {})
            for s in record.evolution.stages:
                valid_phase = s.phase in {"grasp", "transport"} if plan.goal == "source" else s.phase in {"release", "settled"}
                if plan.goal == "source":
                    valid_phase = valid_phase and bool(s.state.get("held_by"))
                if valid_phase and (target := s.state.get(field)) and target != s.state.get("object_id"):
                    valid.append(hint_score(plan.location_hint, target, names.get(target, "")))
        checks.append(max(valid, default=0))
    return sum(checks) / len(checks) if checks else 0.0


def graph_plan(records, query):
    """Resolve ordinal/relative operators over persisted, evidence-bound process records.

    An observed timeline can be incomplete. Resolution supplies retrieval evidence,
    never guarantees an unobserved real-world interaction or a navigation outcome.
    """
    p = query.recall_plan
    result = {"record_ids": [], "target_record_ids": [], "anchor_record_ids": [],
              "entity_ids": [], "warnings": [], "targets": [], "complete": False}
    if not p:
        return result
    if p.requires_geometry:
        result["warnings"].append("Metric geometry/current pose unavailable; do not guess distance")
        return result
    records = [r for r in records if (not query.space or
        (r.space and r.space.namespace == query.space.namespace)) and
        (not query.at or not r.time or not r.time.event_at or r.time.event_at <= query.at)]
    if p.previous_day and query.at:
        day = (query.at - timedelta(days=1)).date()
        records = [r for r in records if r.time and r.time.event_at and r.time.event_at.date() == day]
    if p.goal == "uninteracted" or (p.goal == "entity" and p.entity_hint and not p.anchor_hint):
        targets = visible_targets(records, p, untouched=p.goal == "uninteracted")
        if not targets and p.goal == "entity":
            # Retrieval repair uses representative identity views, not a guessed
            # category alias. These are candidates, never certified targets.
            views = visible_targets(records,p.model_copy(update={"entity_hint":None}))
            result["record_ids"] = list(dict.fromkeys(t["record_id"] for t in views))
            result["warnings"].append("Category unresolved; representative entity views supplied for visual verification")
            return result
        result["targets"] = targets
        result["target_record_ids"] = list(dict.fromkeys(t["record_id"] for t in targets))
        result["record_ids"] = result["target_record_ids"]
        result["entity_ids"] = list(dict.fromkeys(t["entity_id"] for t in targets))
        result["complete"] = bool(targets)
        if p.goal == "uninteracted":
            result["warnings"].append("Untouched means absent from the full observed interaction inventory; unseen actions remain unknown")
        return result
    all_events = sorted((r for r in records if r.evolution), key=lambda r: r.evolution.stages[0].at)
    episodes = all_events
    unresolved_records = []
    episodes = [r for r in episodes if any(s.phase in {"grasp", "transport", "release"}
                or (s.phase == "settled" and s.outcome) for s in r.evolution.stages)]
    if query.space:
        episodes = [r for r in episodes if r.evolution.namespace == query.space.namespace]
    if query.at:
        episodes = [r for r in episodes if r.evolution.stages[-1].at <= query.at]
    if p.previous_day and query.at:
        day = (query.at - timedelta(days=1)).date()
        episodes = [r for r in episodes if r.evolution.stages[0].at.date() == day]
    if p.anchor_hint:
        matches = []
        for r in episodes:
            obj = r.evolution.stages[0].state.get("object_id", "")
            name = payload(r).get("entities", {}).get(obj, "")
            # Background mentions cannot identify the manipulated object.
            score = hint_score(p.anchor_hint, obj, entity_name(r,obj) or name)
            if score >= .5:
                matches.append((score, r))
        if not matches:
            result["warnings"].append("Anchor entity is unresolved")
            # Text similarity tends to retrieve the unnamed anchor itself and miss
            # the requested AFTER/BEFORE target. Return an uncommitted chronological
            # inventory for evidence review, never pretend its ordinal is resolved.
            result["record_ids"] = [r.id for r in episodes]
            result["warnings"].append("Candidate event inventory supplied for anchor verification; no offset target certified")
            return result
        best = max(s for s, r in matches)
        anchors = [r for s, r in matches if s == best]
        if len(anchors) != 1:
            result["warnings"].append("Multiple anchor interactions; cannot establish a unique offset")
            return result
        anchor = anchors[0]
        candidates = ([r for r in episodes if r.evolution.stages[0].at > anchor.evolution.stages[-1].at]
            if p.relation == "after" else [r for r in episodes if r.evolution.stages[-1].at < anchor.evolution.stages[0].at][::-1])
        episodes = candidates[p.offset-1:p.offset]
        result["anchor_record_ids"].append(anchor.id)
        result["warnings"].append("Offset is over observed interactions, not a certified complete real-world history")
    elif p.ordinal:
        if p.goal in {"source", "destination"}:
            episodes = [r for r in episodes if record_match(r, p) > 0]
        episodes = episodes[p.ordinal-1:p.ordinal]
    elif p.latest:
        episodes = episodes[-1:]
    else:
        if (p.all_targets and p.location_hint and p.goal in {"source", "destination"}
                and not any(record_match(r,p) > 0 for r in episodes)):
            # A category mismatch is not evidence that no event occurred. Keep
            # witnessed actions for visual category/role verification; do not
            # relabel furniture or certify additional targets by proximity.
            uncertain = [r.id for r in episodes if record_match(r,p) == 0 and any(
                s.phase in ({"grasp","transport"} if p.goal == "source" else {"release","settled"})
                for s in r.evolution.stages)]
            if uncertain:
                unresolved_records.extend(uncertain)
                result["warnings"].append("Candidate actions retained for location category/role verification; inventory incomplete")
        if p.all_targets and p.goal in {"source", "destination"} and not p.location_hint and any(
                record_match(r,p) == 0 and any(s.phase in (
                    {"grasp","transport"} if p.goal == "source" else {"release","settled"})
                    for s in r.evolution.stages) for r in episodes):
            result["warnings"].append("Some observed interactions lack grounded " + p.goal + " roles; multi-target inventory is incomplete")
            # Keep incomplete event witnesses for evidence review. They are NOT
            # counted as confirmed targets and cannot satisfy a requested ordinal.
            unresolved_records = [r.id for r in episodes if record_match(r,p) == 0]
            # Sparse video can omit the grasp but still show an approach. Preserve
            # those boundary witnesses without counting them as completed pickups.
            unresolved_records.extend(r.id for r in all_events if all(s.phase == "approach" for s in r.evolution.stages))
        episodes = [r for r in episodes if record_match(r, p) > 0]
    for r in episodes:
        result["target_record_ids"].append(r.id)
        field = "source_id" if p.goal == "source" else "target_id" if p.goal == "destination" else "object_id"
        result["entity_ids"].extend(s.state[field] for s in r.evolution.stages if s.state.get(field) and
            (p.goal not in {"source", "destination"} or
             ((s.phase in {"grasp", "transport"} and s.state.get("held_by")) if p.goal == "source"
              else s.phase in {"release", "settled"})))
        result["targets"].extend(target_groundings(r, p.goal, p.location_hint))
    result["record_ids"] = list(dict.fromkeys(result["anchor_record_ids"] + result["target_record_ids"] + unresolved_records))
    result["entity_ids"] = sorted(set(result["entity_ids"]))
    if p.all_targets or p.visit_order:
        unique={}
        for target in result["targets"]:
            unique.setdefault(target["entity_id"],target)
        result["targets"]=list(unique.values())
    # Selecting a record is insufficient when its requested role has no frame evidence.
    result["complete"] = bool(result["targets"]) and not any("incomplete" in w for w in result["warnings"])
    if p.visit_order:
        count = len({t["entity_id"] for t in result["targets"]})
        if any(n > count for n in p.visit_order):
            result["warnings"].append("Requested visit order exceeds grounded target inventory; never skip missing ordinals")
            result["complete"] = False
    return result
