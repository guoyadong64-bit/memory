"""Retrieve uncertain action boundaries and their bounded temporal context.

Context edges carry original observation provenance, not an assertion that a
nearby frame is another completed action, the same physical entity or a room.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re

from benchmarks.hypergraph_evidence_subgraph import pack_subgraph, record_frames
from benchmarks.hypergraph_recall_v3 import (HypergraphRecallV3, _hints, _namespace,
    query_subgraph_v3, project_retrieval as project_v3)
from benchmarks.findingdory_mem0 import RETRIEVAL_K
from benchmarks.findingdory_memory import history_query
from memory_agent.models import RoleBinding
from memory_agent.recall import hint_score, observation_frame, parse_plan


def _context_view(root, indices, database, reasons):
    by_id = {o.id: o for o in root.observations}
    entities = {e.id: e for e in root.entities}
    roles = list(root.role_bindings)
    witnesses = []
    for n in sorted(indices):
        frame, observation, source = database[n]
        by_id[observation.id] = observation
        for entity in source.entities:
            labels = [label.model_copy(update={"observation_ids": [observation.id]})
                      for label in entity.labels if observation.id in label.observation_ids]
            if entity.id not in entities:
                entities[entity.id] = entity.model_copy(update={"labels": labels})
            elif labels:
                existing = entities[entity.id]
                merged = {(label.text, label.confidence, tuple(label.observation_ids)): label
                          for label in [*existing.labels, *labels]}
                entities[entity.id] = existing.model_copy(update={"labels": list(merged.values())})
        visible = {e["id"] for e in frame.get("entities", [])}
        for identity in visible:
            if identity in entities:
                roles.append(RoleBinding(role="visible", entity_id=identity,
                    observation_ids=[observation.id], confidence=.8))
        witnesses.append({"frame_index": n, "source_record_id": source.id,
                          "observation_id": observation.id, "edge": reasons[n]})
    digest = hashlib.sha256((root.id + "|" + str(sorted(indices))).encode()).hexdigest()[:20]
    frames = sorted({n for observation in by_id.values()
                     if (frame := observation_frame(observation)) is not None for n in [frame["frame_index"]]})
    provenance = {"kind": "event_boundary_context_view", "root_record_id": root.id,
        "root_frame_indices": sorted(record_frames(root)),
        "context_only_frames": sorted(indices), "source_observations": witnesses,
        "certified_event_membership": False,
        "interpretation": "Context frames are independent nearby observations or incomplete event candidates. "
                          "Do not treat adjacency/approach as a completed action, identity merge or metric geometry."}
    if "_trace_" in root.id:
        provenance["root_projection"] = json.loads(root.answer)
    view = root.model_copy(update={"id": f"context_view_{digest}", "observations": list(by_id.values()),
        "entities": list(entities.values()), "role_bindings": roles,
        "answer": json.dumps({"frame_indices": frames, "retrieval_projection": provenance}, ensure_ascii=False),
        "source": "Query-time witnessed context view; temporal proximity is not event/identity proof"})
    return view, provenance


def _remap(resolution, mapping):
    if resolution is None:
        return None
    revised = dict(resolution)
    for key in ("record_ids", "target_record_ids", "anchor_record_ids"):
        if key in revised:
            revised[key] = [mapping.get(identity, identity) for identity in revised[key]]
    revised["targets"] = [target | {"record_id": mapping.get(target["record_id"], target["record_id"])}
                          for target in revised.get("targets", [])]
    return revised


def query_subgraph_v4(records, seed_ids, question, resolution):
    chosen, context = query_subgraph_v3(records, seed_ids, question, resolution)
    if not chosen:
        return chosen, resolution, context, {}
    namespace = context["trace_expansion"]["namespace"]
    local = [record for record in records if _namespace(record) == namespace]
    originals = {record.id: record for record in local}
    seed_frames = {n for identity in context["seed_projection_record_ids"] for n in record_frames(originals[identity])}
    current = {n for record in chosen for n in record_frames(record)}
    allowance = min(8 - len(current - seed_frames), max(0, 32 - len(current)))
    if allowance <= 0:
        traces = {root.id: json.loads(root.answer) for root in chosen if "_trace_" in root.id}
        return chosen, resolution, context, traces
    plan = parse_plan(question)
    hints = _hints(question, plan)
    database = {}
    for record in sorted(local, key=lambda r: (bool(r.evolution), r.id)):
        for observation in record.observations:
            frame = observation_frame(observation)
            if frame is not None:
                database.setdefault(frame["frame_index"], (frame, observation, record))

    def matches(frame, allowed=None):
        return max((hint_score(hint, entity["id"], entity["name"] + " " + " ".join(
            label["text"] for label in entity.get("labels", []) if label.get("confidence", 0) >= .8))
            for entity in frame.get("entities", []) if entity["kind"] != "agent"
            and (allowed is None or entity["id"] in allowed) for hint in hints), default=0)

    temporal = bool(plan.anchor_hint or plan.ordinal or plan.visit_order or plan.all_targets or
        re.search(r"\b(?:longest|shortest|duration|interacted|rearrange)\b", question, re.I))
    candidate = {}
    # Incomplete approach/settled observations are indexed as hypotheses. They
    # must not silently disappear just because the strict event resolver lacks
    # a certified source, grasp or placement in a sparse clip.
    phase_groups = defaultdict(list)
    for n, (frame, _, _) in database.items():
        action = frame.get("interaction") or {}
        if action.get("confidence", 0) >= .8 and action.get("object_id"):
            phase_groups[action["object_id"]].append(n)
    selected_objects = {role.entity_id for record in chosen for role in record.role_bindings if role.role == "object"}
    for identity, indices in phase_groups.items():
        if plan.goal == "uninteracted":
            continue
        # Background visibility does not make an unrelated action relevant.
        # Only its object and witnessed location participants can ground it.
        def participants(frame):
            allowed = {identity}
            if plan.goal in {"source", "destination"}:
                action = frame.get("interaction") or {}
                field = "source_id" if plan.goal == "source" else "target_id"
                if action.get(field):
                    allowed.add(action[field])
                allowed.update(relation["target"] for relation in frame.get("relations", [])
                    if relation["source"] == identity and relation["predicate"] in {"on", "inside"}
                    and relation.get("confidence", 0) >= .8)
            return allowed

        match = max(matches(database[n][0], participants(database[n][0])) for n in indices)
        if not temporal and match < .5:
            continue
        # Take both sides of an observed fragment; neither is called a complete
        # interaction and frame-count gaps are never called duration.
        for n in (min(indices), max(indices)):
            if n not in current:
                priority = 3.0 + match + .4 * (identity not in selected_objects)
                candidate[n] = (priority, "incomplete_action_boundary", identity)

    root_frames = {record.id: sorted(record_frames(record)) for record in chosen}
    for rank, root in enumerate(chosen):
        indices = root_frames[root.id]
        if not indices:
            continue
        named = max(matches(database[n][0]) for n in indices if n in database)
        important = root.evolution is not None or root.id in (resolution or {}).get("record_ids", []) or named >= .5
        if not important:
            continue
        boundaries = [indices[0], indices[-1]]
        if plan.goal == "source":
            offsets = (-1, -2, 1, 2)
        elif plan.goal == "destination":
            offsets = (1, 2, -1, -2)
        else:
            offsets = (1, -1, 2, -2)
        for boundary in boundaries:
            for offset_rank, offset in enumerate(offsets):
                n = boundary + offset
                if n not in database or n in current:
                    continue
                frame = database[n][0]
                shared = {e["id"] for e in frame.get("entities", []) if e["kind"] != "agent"} & {
                    e.id for e in root.entities if e.kind != "agent"}
                score = 2 + named + .15 * min(len(shared), 3) - .15 * offset_rank - .015 * rank
                reason = "observed_before_context" if offset < 0 else "observed_after_context"
                value = (score, reason, root.id)
                if n not in candidate or value[0] > candidate[n][0]:
                    candidate[n] = value
    selected = []
    counts = defaultdict(int)
    for n in sorted(candidate, key=lambda n: (candidate[n][0], -n), reverse=True):
        group = candidate[n][2]
        if counts[group] >= 2:
            continue
        selected.append(n)
        counts[group] += 1
        if len(selected) == allowance:
            break
    assignments = defaultdict(list)
    for n in selected:
        # Attach to the nearest retrieved witness, with an explicit context edge.
        root = min(chosen, key=lambda r: (min(abs(n - i) for i in root_frames[r.id]), r.id))
        assignments[root.id].append(n)
    result, mapping, metadata = [], {}, {}
    reasons = {n: candidate[n][1] for n in selected}
    for root in chosen:
        if root.id in assignments:
            view, provenance = _context_view(root, assignments[root.id], database, reasons)
            result.append(view)
            mapping[root.id] = view.id
            metadata[view.id] = provenance
        else:
            result.append(root)
            if "_trace_" in root.id:
                metadata[root.id] = json.loads(root.answer)
    context["context_expansion"] = {"added_context_frames": selected, "max_added_total": 8,
        "max_candidates_when_expanding": 32, "seed_unique_frames": len(seed_frames),
        "added_unique_frames": len(({n for r in result for n in record_frames(r)}) - seed_frames),
        "edges_are_hypotheses": True}
    return result, _remap(resolution, mapping), context, metadata


def project_retrieval(records, seed_ids, question, resolution):
    selected, revised, context, metadata = query_subgraph_v4(records, seed_ids, question, resolution)
    reserve = len(json.dumps(metadata, ensure_ascii=False).encode()) + len(metadata) * 200
    plan = parse_plan(question)
    evidence, revised, audit = pack_subgraph(selected, revised, max_bytes=max(2, 40000 - reserve),
        require_all_targets=plan.all_targets or bool(plan.visit_order))
    for item in evidence:
        if item["id"] in metadata:
            projection = metadata[item["id"]]
            item["content"]["retrieval_projection"] = projection
            if projection.get("kind") == "event_boundary_context_view" and item["content"].get("trajectory"):
                # The common selector interprets trajectory as certifying the
                # whole record as one event. Independent context cannot carry
                # that grouping; retain root stage proof under an explicit key.
                item["content"]["root_event_trajectory"] = item["content"].pop("trajectory")
                item["content"]["root_event_frame_indices"] = projection["root_frame_indices"]
    size = len(json.dumps(evidence, ensure_ascii=False).encode())
    if size > 40000 or audit["dropped_record_ids"]:
        previous, previous_resolution, previous_audit = project_v3(records, seed_ids, question, resolution)
        old_frames = {n for item in previous for n in item["frame_indices"]}
        new_frames = {n for item in evidence for n in item["frame_indices"]}
        if size > 40000 or not old_frames <= new_frames:
            previous_audit["context_budget_fallback"] = True
            return previous, previous_resolution, previous_audit
    if revised is not None:
        kept = {item["id"] for item in evidence}
        mapping = {projection["root_record_id"]: identity for identity, projection in metadata.items()
                   if projection.get("kind") == "event_boundary_context_view"}
        revised["observed_event_fragments"] = [event | {"record_id": mapping.get(event["record_id"], event["record_id"])}
            for event in context.get("observed_event_fragments", [])
            if mapping.get(event["record_id"], event["record_id"]) in kept]
        revised["event_links"] = [edge | {key: mapping.get(edge[key], edge[key])
            for key in ("from_record_id", "to_record_id")} for edge in context.get("event_links", [])
            if all(mapping.get(edge[key], edge[key]) in kept for key in ("from_record_id", "to_record_id"))]
        revised["event_inventory_caveat"] = context.get("caveat")
        revised["context_edge_caveat"] = "Additional boundary/context observations are hypotheses, not newly certified interactions."
    indexed = {record.id: record for record in records}
    seed_frames = {n for identity in context.get("seed_projection_record_ids", [])
                   for n in record_frames(indexed[identity])}
    final_frames = {n for item in evidence for n in item["frame_indices"]}
    actual = dict(context["trace_expansion"])
    actual.update(output_unique_frames=len(final_frames), added_unique_frames=len(final_frames - seed_frames))
    expansion = dict(context.get("context_expansion", {}))
    if expansion:
        expansion["added_context_frames"] = [n for n in expansion["added_context_frames"] if n in final_frames]
        expansion["added_unique_frames"] = len(final_frames - seed_frames)
    audit.update(bytes=size, max_bytes=40000, context_expansion=expansion, trace_expansion=actual)
    return evidence, revised, audit


class HypergraphRecallV4(HypergraphRecallV3):
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
        retrieval["read_projection_v4"] = audit
        return evidence, resolution, retrieval
