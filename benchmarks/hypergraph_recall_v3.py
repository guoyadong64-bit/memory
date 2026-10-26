"""Bounded, namespace-local entity incidence over witnessed hypergraph records.

No labels, task IDs or category tables enter retrieval. A trace is a projection
of original observations, never a newly asserted continuous physical event.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re

from benchmarks.findingdory_mem0 import RETRIEVAL_K
from benchmarks.findingdory_memory import history_query
from benchmarks.hypergraph_evidence_subgraph import pack_subgraph, record_frames
from benchmarks.hypergraph_recall_v2 import HypergraphRecallV2, query_subgraph
from memory_agent.models import EntityLabel, MemoryEntity, Record, RoleBinding, SpatialContext, TemporalContext
from memory_agent.recall import hint_score, observation_frame, parse_plan

MAX_ENTITIES = 4
MAX_TRACE_FRAMES = 4
MAX_ADDED_FRAMES = 8
MAX_CANDIDATE_FRAMES = 32


def _namespace(record):
    if record.space:
        return record.space.namespace
    values = {observation.namespace for observation in record.observations}
    return next(iter(values)) if len(values) == 1 else None


def _hints(question, plan):
    hints = [plan.anchor_hint, plan.location_hint]
    if plan.entity_hint and not re.search(r"\b(?:which|that|time|duration|interacted)\b", plan.entity_hint):
        hints.append(plan.entity_hint)
    # Preserve multiword nouns and the manipulated object in room/source queries.
    for pattern in (r"(?:navigate to|revisit)\s+(?:any |an? |the |all the )?(.+?)\s+(?:that )?you\s+(?:did not interact|picked|placed)",
                    r"(?:picked|placed)\s+(?:an? |the )?(.+?)\s+(?:from|on|in)\b"):
        match = re.search(pattern, question, re.I)
        if match and not re.fullmatch(r"(?:objects?|receptacles?|room)", match[1].strip(), re.I):
            hints.append(match[1].strip())
    return [hint for hint in hints if hint]


def _entity_trace(entity_id, namespace, frames, source_records, candidates, seed_frames, new_allowance, plan):
    seen = sorted(n for n in candidates if any(e["id"] == entity_id for e in frames[n][0].get("entities", [])))
    if not seen or new_allowance <= 0:
        return None

    def quality(n):
        frame = frames[n][0]
        entity = next(e for e in frame["entities"] if e["id"] == entity_id)
        partial = entity.get("attributes", {}).get("visibility") in {"partial", "occluded"}
        supported = any(r["source"] == entity_id and r["predicate"] in {"on", "inside"}
                        and r.get("confidence", 0) >= .8 for r in frame.get("relations", []))
        held = any(r["source"] == entity_id and r["predicate"] == "held_by"
                   and r.get("confidence", 0) >= .8 for r in frame.get("relations", []))
        return (not partial, supported and not held, n)

    present = [n for n in seen if n in seed_frames]
    selected = [max(present, key=quality)] if present else []
    role_frames = []
    for n in seen:
        action = frames[n][0].get("interaction") or {}
        field = "source_id" if plan.goal == "source" else "target_id" if plan.goal == "destination" else "object_id"
        if action.get(field) == entity_id and action.get("confidence", 0) >= .8:
            role_frames.append(n)
    choices = [*sorted(role_frames, key=quality, reverse=True), max(seen, key=quality), seen[-1], seen[0],
               *sorted(seen, key=quality, reverse=True)]
    added = 0
    for n in choices:
        if n in selected:
            continue
        if n not in seed_frames:
            if added >= new_allowance:
                continue
            added += 1
        selected.append(n)
        if len(selected) == MAX_TRACE_FRAMES:
            break
    if not any(n not in seed_frames for n in selected):
        return None
    selected = sorted(selected)
    observations = [frames[n][1] for n in selected]
    observation_ids = {o.id for o in observations}
    entities = {}
    for n in selected:
        for raw in frames[n][0].get("entities", []):
            entry = entities.setdefault(raw["id"], {"id": raw["id"], "namespace": namespace,
                "name": raw["name"], "kind": raw["kind"], "labels": [], "attributes": {}})
            # Dynamic visibility/support stays in frame-bound rows, never a
            # permanent property of an identity spanning several observations.
            if raw.get("attributes", {}).get("role"):
                entry["attributes"]["role"] = raw["attributes"]["role"]
            for label in raw.get("labels", []):
                if label.get("confidence", 0) >= .8:
                    entry["labels"].append(EntityLabel(text=label["text"], confidence=label["confidence"],
                                                        observation_ids=[frames[n][1].id]))
    roles = [RoleBinding(role="visible", entity_id=entity_id, observation_ids=sorted(observation_ids), confidence=.8)]
    for record in source_records:
        for role in record.role_bindings:
            ids = sorted(set(role.observation_ids) & observation_ids)
            if role.entity_id == entity_id and role.role != "visible" and ids:
                roles.append(role.model_copy(update={"observation_ids": ids, "stage_id": None}))
    fragment_ids = [record.id for record in source_records if record.evolution and
                    set(record_frames(record)) & set(selected)]
    provenance = {"kind": "same_entity_observation_trace", "entity_id": entity_id, "namespace": namespace,
        "continuous_event": False, "source_observations": [{"frame_index": n,
            "observation_id": frames[n][1].id, "source_record_id": frames[n][2]} for n in selected],
        "event_fragment_ids": fragment_ids,
        "interpretation": "One-hop same-ID visibility evidence; gaps do not imply continuous action or duration."}
    digest = hashlib.sha256((namespace + "|" + entity_id).encode()).hexdigest()[:18]
    record = Record(id=f"entity_trace_{digest}", key=source_records[0].key,
        question=f"Observed locations of {entity_id}",
        answer=json.dumps(provenance | {"frame_indices": selected}, ensure_ascii=False),
        source="Derived query-time subgraph projection; original observations retained", verified=True,
        kind="experience", space=SpatialContext(namespace=namespace, entities=[entity_id]),
        time=TemporalContext(event_at=max(o.observed_at for o in observations)),
        observations=observations, entities=[MemoryEntity(**entity) for entity in entities.values()],
        role_bindings=roles)
    return record


def query_subgraph_v3(records, seed_ids, question, resolution):
    index = {record.id: record for record in records}
    namespaces = {_namespace(index[identity]) for identity in seed_ids if identity in index}
    namespaces.discard(None)
    if len(namespaces) > 1:
        raise ValueError("A retrieval seed set must belong to one namespace")
    namespace = next(iter(namespaces)) if namespaces else None
    local = [record for record in records if namespace is not None and _namespace(record) == namespace]
    if not local:
        return [], {"trace_expansion": {"added_unique_frames": 0, "promoted_entities": 0,
            "input_unique_frames": 0, "output_unique_frames": 0}, "candidate_sources": {}}
    chosen, context = query_subgraph(local, seed_ids, question, resolution)
    context["seed_projection_record_ids"] = [record.id for record in chosen]
    baseline_frames = set().union(*(set(record_frames(record)) for record in chosen)) if chosen else set()
    plan = parse_plan(question)
    hints = _hints(question, plan)
    seed_visible = {entity["id"] for record in chosen for frame in record_frames(record).values()
                    for entity in frame.get("entities", [])}
    by_frame, catalog, appearances = {}, {}, defaultdict(set)
    witnessed_names = defaultdict(set)
    roles = defaultdict(set)
    participants = defaultdict(set)
    entity_records = defaultdict(list)
    for record in sorted(local, key=lambda r: (bool(r.evolution), r.id)):
        for observation in record.observations:
            frame = observation_frame(observation)
            if frame is None:
                continue
            n = frame["frame_index"]
            by_frame.setdefault(n, (frame, observation, record.id))
            for entity in frame.get("entities", []):
                catalog.setdefault(entity["id"], entity)
                appearances[entity["id"]].add(n)
                witnessed_names[entity["id"]].add(entity["name"])
                witnessed_names[entity["id"]].update(label["text"] for label in entity.get("labels", [])
                    if label.get("confidence", 0) >= .8)
        for role in record.role_bindings:
            if role.role in {"object", "source", "destination"} and role.confidence >= .8:
                roles[role.entity_id].add(role.role)
        for entity in record.entities:
            entity_records[entity.id].append(record)
            witnessed_names[entity.id].update(label.text for label in entity.labels if label.confidence >= .8)
            if record.id in {r.id for r in chosen}:
                participants[entity.id].update(role.role for role in record.role_bindings
                    if role.entity_id == entity.id and role.role in {"object", "source", "destination"} and role.confidence >= .8)
    grounded = {target["entity_id"] for target in (resolution or {}).get("targets", []) if target.get("entity_id")}
    named_entities = {identity for identity, entity in catalog.items() if entity["kind"] != "agent" and
        max((hint_score(hint, identity, " ".join(sorted(witnessed_names[identity]))) for hint in hints), default=0) >= .75}
    connected_locations = set()
    if plan.goal in {"source", "destination"} and named_entities:
        # Query object -> witnessed event role -> location identity. This is an
        # existing incidence link, not propagation through newly discovered views.
        for record in local:
            objects = {role.entity_id for role in record.role_bindings
                       if role.role == "object" and role.confidence >= .8}
            if objects & named_entities:
                connected_locations.update(role.entity_id for role in record.role_bindings
                    if role.role == plan.goal and role.confidence >= .8)
            for frame in record_frames(record).values():
                action = frame.get("interaction") or {}
                if action.get("object_id") in named_entities and action.get("confidence", 0) >= .8:
                    field = "source_id" if plan.goal == "source" else "target_id"
                    if action.get(field):
                        connected_locations.add(action[field])
    focus = []
    for identity, entity in catalog.items():
        if entity["kind"] == "agent" or entity.get("attributes", {}).get("role") == "robot_component":
            continue
        named = max((hint_score(hint, identity, " ".join(sorted(witnessed_names[identity]))) for hint in hints), default=0)
        if plan.goal == "uninteracted" and roles[identity]:
            continue
        wanted_roles = ({plan.goal} if plan.goal in {"source", "destination"} else
                        {"source", "destination"} if re.search(r"\breceptacles?\b", question, re.I) else {"object"})
        is_participant = bool(wanted_roles & participants[identity])
        if named < .5 and identity not in grounded and not is_participant and identity not in connected_locations:
            continue
        # A generic actor/background hub is not an expansion seed. Only named
        # entities or actual object/source/destination participants qualify.
        score = (identity in connected_locations, named >= .75, named, identity in grounded, is_participant,
                 identity in seed_visible, -len(appearances[identity]))
        focus.append((score, identity))
    focus.sort(key=lambda x: (x[0], x[1]), reverse=True)
    protected = {t["record_id"] for t in (resolution or {}).get("targets", [])} | set(
        (resolution or {}).get("anchor_record_ids", []))
    promoted = []
    current_frames = set(baseline_frames)
    extra_cap = min(MAX_ADDED_FRAMES, max(0, MAX_CANDIDATE_FRAMES - len(baseline_frames)))
    for _, identity in focus:
        if len(promoted) == MAX_ENTITIES:
            break
        allowance = extra_cap - len(current_frames - baseline_frames)
        trace = _entity_trace(identity, namespace, by_frame, entity_records[identity], appearances[identity],
                              current_frames, min(2, allowance), plan)
        if trace is None:
            continue
        if len(chosen) < RETRIEVAL_K:
            chosen.insert(0, trace)
        else:
            trace_frames = set(record_frames(trace))
            for victim in reversed(chosen):
                if victim.id in protected or "_trace_" in victim.id:
                    continue
                others = set().union(*(set(record_frames(r)) for r in chosen if r.id != victim.id), trace_frames)
                if set(record_frames(victim)) <= others:
                    chosen.remove(victim)
                    chosen.insert(0, trace)
                    break
            else:
                continue
        promoted.append(identity)
        current_frames = set().union(*(set(record_frames(r)) for r in chosen))
    context["trace_expansion"] = {"namespace": namespace, "input_unique_frames": len(baseline_frames),
        "output_unique_frames": len(current_frames), "added_unique_frames": len(current_frames - baseline_frames),
        "promoted_entities": len(promoted), "entity_ids": promoted, "max_entities": MAX_ENTITIES,
        "max_frames_per_trace": MAX_TRACE_FRAMES, "max_added_frames": MAX_ADDED_FRAMES,
        "candidate_frame_cap": MAX_CANDIDATE_FRAMES, "hops": 1}
    for record in chosen:
        context["candidate_sources"].setdefault(record.id, "same_identity_witness_trace")
    return chosen, context


def project_retrieval(records, seed_ids, question, resolution):
    selected, context = query_subgraph_v3(records, seed_ids, question, resolution)
    traces = {r.id: json.loads(r.answer) for r in selected if "_trace_" in r.id}
    reserved = len(json.dumps(traces, ensure_ascii=False).encode()) + 200 * len(traces)
    plan = parse_plan(question)
    evidence, revised, audit = pack_subgraph(selected, resolution, max_bytes=40000 - reserved,
        require_all_targets=plan.all_targets or bool(plan.visit_order))
    for item in evidence:
        if item["id"] in traces:
            item["content"]["retrieval_projection"] = traces[item["id"]]
    size = len(json.dumps(evidence, ensure_ascii=False).encode())
    if size > 40000:
        raise ValueError("Subgraph projection exceeded its evidence budget")
    if audit["dropped_record_ids"]:
        # Added provenance must not push previously retainable witnesses out of
        # the fixed context. A bounded fallback is label-free and conservative.
        indexed = {record.id: record for record in records}
        base = [indexed[identity] for identity in context.get("seed_projection_record_ids", [])]
        base_evidence, base_resolution, base_audit = pack_subgraph(base, resolution,
            require_all_targets=plan.all_targets or bool(plan.visit_order))
        old_frames = {n for item in base_evidence for n in item["frame_indices"]}
        new_frames = {n for item in evidence for n in item["frame_indices"]}
        if not old_frames <= new_frames:
            evidence, revised, audit = base_evidence, base_resolution, base_audit
            size = len(json.dumps(evidence, ensure_ascii=False).encode())
            audit["trace_budget_fallback"] = "preserve_previous_projection_witnesses"
    kept = {item["id"] for item in evidence}
    if revised is not None:
        revised["observed_event_fragments"] = [event for event in context.get("observed_event_fragments", []) if event["record_id"] in kept]
        revised["event_links"] = [edge for edge in context.get("event_links", [])
                                  if edge["from_record_id"] in kept and edge["to_record_id"] in kept]
        revised["event_inventory_caveat"] = context.get("caveat")
    actual = dict(context["trace_expansion"])
    indexed = {record.id: record for record in records}
    baseline_frames = {n for identity in context.get("seed_projection_record_ids", [])
                       for n in record_frames(indexed[identity])}
    final_frames = {n for item in evidence for n in item["frame_indices"]}
    actual["output_unique_frames"] = len(final_frames)
    actual["entity_ids"] = [traces[identity]["entity_id"] for identity in traces if identity in kept]
    actual["promoted_entities"] = len(actual["entity_ids"])
    actual["added_unique_frames"] = len(final_frames - baseline_frames)
    audit.update(bytes=size, max_bytes=40000, trace_expansion=actual,
                 candidate_sources=context["candidate_sources"])
    return evidence, revised, audit


class HypergraphRecallV3(HypergraphRecallV2):
    def retrieve(self, episode, question):
        settings = self.client.settings
        query = history_query(question, episode, self.records[episode])
        query.recall_plan = parse_plan(question)
        selected, retrieval = self.graphs[episode].search(query, RETRIEVAL_K, strategy="hypergraph",
            embedder=self.client.embed,
            embedding_namespace=(settings.embedding_url or settings.llm_url).rstrip("/") + "|" + settings.embedding_model,
            minimum_similarity=settings.semantic_min_similarity)
        if retrieval["semantic_backend"] != "embedding":
            raise ValueError("Hypergraph embedding fallback; comparison stopped")
        evidence, resolution, audit = project_retrieval(self.records[episode], [r.id for r in selected[:RETRIEVAL_K]],
            question, retrieval.get("plan_evidence"))
        retrieval["read_projection_v3"] = audit
        return evidence, resolution, retrieval
