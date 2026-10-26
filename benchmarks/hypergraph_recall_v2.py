"""Opt-in hypergraph read projection; historical baseline adapters stay frozen."""
from __future__ import annotations

import re

from benchmarks.comparison_methods import HypergraphMethod
from benchmarks.findingdory_mem0 import RETRIEVAL_K
from benchmarks.findingdory_memory import history_query
from benchmarks.hypergraph_evidence_subgraph import observed_events, pack_subgraph, record_frames
from memory_agent.recall import entity_name, hint_score, parse_plan


def query_subgraph(records, seed_ids, question, resolution):
    """Expand only through witnessed event roles and chronological neighbors.

    Fixed top-20 record budget. These neighbors are candidates, not certified
    temporal answers. No category synonym table or benchmark label is used.
    """
    index = {record.id: record for record in records}
    plan = parse_plan(question)
    events = observed_events(records)
    support = []
    reasons = {}

    def include(identity, reason):
        if identity in index and identity not in support:
            support.append(identity)
            reasons[identity] = reason

    for target in (resolution or {}).get("targets", []):
        include(target["record_id"], "grounded_role_target")
    for identity in (resolution or {}).get("anchor_record_ids", []):
        include(identity, "grounded_anchor")
    if plan.anchor_hint:
        anchors = [i for i, event in enumerate(events) if hint_score(
            plan.anchor_hint, event.get("object_id") or "",
            entity_name(index[event["record_id"]], event.get("object_id"))) >= .75]
        for i in anchors:
            include(events[i]["record_id"], "observed_anchor_identity")
            direction = -1 if plan.relation == "before" else 1
            for step in range(1, min(plan.offset + 2, 5)):
                n = i + direction * step
                if 0 <= n < len(events):
                    include(events[n]["record_id"], "observed_event_neighbor_uncertain")
    if plan.all_targets and plan.goal in {"source", "destination"}:
        for event in events:
            if event["source_ids" if plan.goal == "source" else "destination_ids"]:
                include(event["record_id"], "witnessed_event_role_inventory")
    if re.search(r"\b(?:longest|shortest|duration|longer|shorter)\b", question, re.I):
        for event in events:
            include(event["record_id"], "observed_event_span_not_true_duration")
    # Reserve some semantic/graph seeds even when the role inventory is large.
    support = support[:12]
    chosen = list(dict.fromkeys(identity for identity in seed_ids if identity in index))[:RETRIEVAL_K]
    frame_sets = {identity: set(record_frames(index[identity])) for identity in {*chosen, *support}}
    protected = {t["record_id"] for t in (resolution or {}).get("targets", [])} | set(
        (resolution or {}).get("anchor_record_ids", []))
    for identity in support:
        if identity in chosen:
            continue
        if len(chosen) < RETRIEVAL_K:
            chosen.append(identity)
            continue
        # Replace only redundant witnesses. Temporal expansion cannot discard
        # the sole frame evidence of an existing seed merely to add a neighbor.
        for victim in reversed(chosen):
            if victim in protected or victim in support:
                continue
            other_frames = set().union(*(frame_sets[k] for k in chosen if k != victim), frame_sets[identity])
            if frame_sets[victim] <= other_frames:
                chosen[chosen.index(victim)] = identity
                break
    priority = {identity: i for i, identity in enumerate(support)}
    chosen.sort(key=lambda identity: priority.get(identity, len(priority)))
    chosen_events = [event for event in events if event["record_id"] in chosen]
    return [index[identity] for identity in chosen if identity in index], {
        "candidate_sources": {identity: reasons.get(identity, "original_hypergraph_seed") for identity in chosen},
        "observed_event_fragments": chosen_events,
        "event_links": [{"from_record_id": a["record_id"], "to_record_id": b["record_id"],
                         "relation": "next_observed_fragment", "complete_history": False}
                        for a, b in zip(events, events[1:])
                        if a["record_id"] in chosen and b["record_id"] in chosen],
        "caveat": "Fragments can describe the same physical interaction; unobserved actions remain unknown. "
                  "Frame gaps and sampled spans are not certified interaction durations."}


def project_retrieval(records, seed_ids, question, resolution):
    selected, context = query_subgraph(records, seed_ids, question, resolution)
    plan = parse_plan(question)
    evidence, revised, packing = pack_subgraph(selected, resolution,
        require_all_targets=plan.all_targets or bool(plan.visit_order))
    if revised is not None:
        kept = {item["id"] for item in evidence}
        revised["observed_event_fragments"] = [e for e in context["observed_event_fragments"] if e["record_id"] in kept]
        revised["event_links"] = [e for e in context["event_links"]
                                  if e["from_record_id"] in kept and e["to_record_id"] in kept]
        revised["event_inventory_caveat"] = context["caveat"]
    return evidence, revised, {**packing, "candidate_sources": context["candidate_sources"]}


class HypergraphRecallV2(HypergraphMethod):
    def retrieve(self, episode, question):
        settings = self.client.settings
        query = history_query(question, episode, self.records[episode])
        query.recall_plan = parse_plan(question)
        selected, retrieval = self.graphs[episode].search(query, RETRIEVAL_K,
            strategy="hypergraph", embedder=self.client.embed,
            embedding_namespace=(settings.embedding_url or settings.llm_url).rstrip("/") + "|" + settings.embedding_model,
            minimum_similarity=settings.semantic_min_similarity)
        if retrieval["semantic_backend"] != "embedding":
            raise ValueError("Hypergraph embedding fallback; comparison stopped")
        evidence, resolution, audit = project_retrieval(self.records[episode],
            [record.id for record in selected[:RETRIEVAL_K]], question, retrieval.get("plan_evidence"))
        retrieval["read_projection_v2"] = audit
        return evidence, resolution, retrieval
