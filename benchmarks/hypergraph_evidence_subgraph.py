"""Export a witnessed memory subgraph with shared entities and frame provenance.

This is a read projection: it neither edits stored observations nor resolves new
physical identities. Query labels and evaluation answers are not accepted here.
"""
from __future__ import annotations

import json

from memory_agent.recall import observation_frame


def record_frames(record):
    return {frame["frame_index"]: frame for observation in record.observations
            if (frame := observation_frame(observation)) is not None}


def observed_events(records):
    """Chronological event fragments with witnessed roles, not a complete history."""
    result = []
    for record in records:
        if not record.evolution:
            continue
        stages = record.evolution.stages
        if not any(s.phase in {"grasp", "transport", "release"} for s in stages) and not any(
                role.role == "destination" for role in record.role_bindings):
            continue
        frames = record_frames(record)
        result.append({"record_id": record.id,
                       "object_id": stages[0].state.get("object_id"),
                       "frame_indices": sorted(frames),
                       "start_frame": min(frames) if frames else None,
                       "end_frame": max(frames) if frames else None,
                       "start_clock": frames[min(frames)].get("clock") if frames else None,
                       "end_clock": frames[max(frames)].get("clock") if frames else None,
                       "phases": list(dict.fromkeys(s.phase for s in stages)),
                       "source_ids": sorted({r.entity_id for r in record.role_bindings if r.role == "source"}),
                       "destination_ids": sorted({r.entity_id for r in record.role_bindings if r.role == "destination"}),
                       "start_at": stages[0].at.isoformat()})
    result.sort(key=lambda x: (x["start_at"], x["record_id"]))
    return result


def _bytes(value):
    # The same JSON byte accounting as the original 40 KB evidence packer.
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _project(records, description_limit):
    entities = {}
    for record in records:
        for entity in record.entities:
            old = entities.setdefault(entity.id, {"id": entity.id, "name": entity.name,
                "kind": entity.kind, "labels": set(), "attributes": entity.attributes})
            old["labels"].update(label.text for label in entity.labels if label.confidence >= .8)
    catalog = [entities[key] | {"labels": sorted(entities[key]["labels"])} for key in sorted(entities)]
    refs = {entity["id"]: index for index, entity in enumerate(catalog)}
    schema = {
        "entity_refs": "Zero-based indices into this shared entity_catalog; distinct IDs remain distinct.",
        "label_confidence_floor": .8,
        "frame_columns": ["frame_index", "clock", "description_excerpt", "visible_entity_refs",
                          "relations", "interaction", "uncertainty"],
        "relation_columns": ["source_ref", "predicate", "target_ref", "confidence"],
        "interaction_columns": ["object_ref", "actor_ref", "phase", "source_ref", "target_ref",
                                "action", "outcome", "confidence"],
        "role_columns": ["role", "entity_ref", "supporting_frames", "confidence", "basis"],
        "provenance": "Shared frame rows appear once. Each memory's frame_refs refer to those rows. "
                      "Only that memory's own observed frames occur in its frame_indices. "
                      "Descriptions may be excerpts; absence of a phrase does not negate an observation. "
                      "Linked roles are memory inferences; observed roles are model interpretations."
    }
    emitted = set()
    result = []
    for record in records:
        by_frame = record_frames(record)
        observation_frames = {o.id: frame["frame_index"] for o in record.observations
                              if (frame := observation_frame(o)) is not None}
        rows = []
        for n, frame in sorted(by_frame.items()):
            if n in emitted:
                continue
            emitted.add(n)
            interaction = frame.get("interaction")
            action = None
            if interaction:
                action = [refs.get(interaction.get("object_id")), refs.get(interaction.get("actor_id")),
                          interaction.get("phase"), refs.get(interaction.get("source_id")),
                          refs.get(interaction.get("target_id")), interaction.get("action"),
                          interaction.get("outcome"), interaction.get("confidence")]
            description = frame.get("description", "")
            if len(description) > description_limit:
                description = description[:description_limit] + (" …" if description_limit else "")
            rows.append([n, frame.get("clock"), description,
                         [refs[e["id"]] for e in frame.get("entities", []) if e["id"] in refs],
                         [[refs[r["source"]], r["predicate"], refs[r["target"]], r.get("confidence")]
                          for r in frame.get("relations", []) if r["source"] in refs and r["target"] in refs],
                         action, frame.get("uncertainty", [])])
        roles = {}
        for role in record.role_bindings:
            if role.role == "visible":
                continue  # Visibility already has exact frame-bound rows.
            key = (role.role, role.entity_id, role.basis)
            entry = roles.setdefault(key, [role.role, refs[role.entity_id], set(), role.confidence, role.basis])
            entry[2].update(observation_frames[oid] for oid in role.observation_ids if oid in observation_frames)
            entry[3] = min(entry[3], role.confidence)
        content = {"frame_refs": sorted(by_frame), "frame_rows": rows,
                   "event_roles": [row[:2] + [sorted(row[2])] + row[3:] for row in roles.values()]}
        if record.evolution:
            # Preserve event grouping for the unchanged common image selector.
            content["trajectory"] = [{"phase": stage.phase, "witness_frames": sorted({
                observation_frames[oid] for oid in stage.observation_ids if oid in observation_frames})}
                for stage in record.evolution.stages]
        if not result:
            content.update(subgraph_schema=schema, entity_catalog=catalog)
        result.append({"id": record.id, "frame_indices": sorted(by_frame), "content": content})
    return result


def pack_subgraph(records, resolution=None, *, max_bytes=40000, require_all_targets=True):
    """Deduplicate before dropping evidence; never invent frame coverage.

    First reduce descriptive redundancy, keeping identity, relation, action and
    provenance facts. Drop a whole record only if this bounded projection still
    does not fit. Nothing is selected using an evaluation label.
    """
    if max_bytes < 2:
        raise ValueError("Context budget must fit a JSON list")
    unique = list({r.id: r for r in records}.values())
    priority = list(dict.fromkeys([t["record_id"] for t in (resolution or {}).get("targets", [])] +
                                 (resolution or {}).get("anchor_record_ids", [])))
    order = {identity: index for index, identity in enumerate(priority)}
    unique = sorted(enumerate(unique), key=lambda p: (order.get(p[1].id, len(order)), p[0]))
    retained = [record for _, record in unique]
    dropped = []
    evidence = []
    limit = 240
    while retained:
        for limit in (240, 120, 40, 0):
            evidence = _project(retained, limit)
            if _bytes(evidence) <= max_bytes:
                break
        if _bytes(evidence) <= max_bytes:
            break
        dropped.append(retained.pop().id)
        evidence = []
    revised = None
    if resolution is not None:
        revised = dict(resolution)
        kept = {item["id"]: set(item["frame_indices"]) for item in evidence}
        targets = resolution.get("targets", [])
        revised["targets"] = [t for t in targets if t.get("frame_index") in kept.get(t["record_id"], set())]
        missing = ((len(revised["targets"]) != len(targets) if require_all_targets else
                    bool(targets) and not revised["targets"]) or
                   any(identity not in kept for identity in resolution.get("anchor_record_ids", [])))
        if missing:
            revised["complete"] = False
            revised["warnings"] = [*resolution.get("warnings", []), "Context budget omitted required target or anchor evidence"]
        revised["context_dropped_record_ids"] = dropped
    return evidence, revised, {"packing": "shared-entity-frame-subgraph-v1", "max_bytes": max_bytes,
        "bytes": _bytes(evidence), "input_records": len(unique), "retained_records": len(evidence),
        "dropped_record_ids": dropped, "description_excerpt_limit": limit,
        "input_unique_frames": len({n for _, r in unique for n in record_frames(r)}),
        "output_unique_frames": len({n for item in evidence for n in item["frame_indices"]})}
