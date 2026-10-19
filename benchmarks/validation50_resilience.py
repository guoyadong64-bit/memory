"""Label-blind, deterministic repair of visual JSON consistency errors.

Never add a visual fact. Keep descriptions and usable evidence; discard an
unsupported structured claim or allocate a new ID rather than guessing.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from benchmarks.findingdory_memory import VisualFrame, validate_batch


def _new_id(prefix: str, parts: tuple[str, ...], used: set[str]) -> str:
    stem = re.sub(r"[^a-z0-9_]+", "_", prefix.lower()).strip("_") or "id"
    if not stem[0].isalpha():
        stem = "id_" + stem
    digest = hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:12]
    candidate = f"{stem[:49]}_{digest}"
    assert len(candidate) <= 64
    return candidate


def normalize_observations(output, indices, previous):
    """Return (valid JSON, audit changes); fail if the model omitted a frame.

    This is a data-quality fallback, not a correction against answer labels.
    """
    raw = output.get("observations") if isinstance(output, dict) else None
    if not isinstance(raw, list) or [r.get("frame_index") if isinstance(r, dict) else None
                                   for r in raw] != list(indices):
        raise ValueError("Visual output omitted, duplicated or reordered indices")
    changes = []
    result = []
    for source in raw:
        committed = [*previous, *result]
        identities = {e.id: (e.name, e.kind) for frame in committed for e in frame.entities}
        cases = {frame.interaction.id: frame.interaction.object_id for frame in committed
                 if frame.interaction}
        used_entities = set(identities)
        used_cases = set(cases)
        item = copy.deepcopy(source)
        index = item["frame_index"]
        notes = list(item.get("uncertainty") or [])

        def changed(field, reason, before=None, after=None):
            changes.append({"frame_index": index, "field": field, "reason": reason,
                            "from": before, "to": after})

        entities = []
        aliases = {}
        for entity in item.get("entities") or []:
            if not isinstance(entity, dict) or not isinstance(entity.get("id"), str):
                changed("entities", "malformed entity removed")
                continue
            eid, name, kind = entity["id"], entity.get("name"), entity.get("kind")
            if eid in identities and identities[eid] != (name, kind):
                # A changed visual name is not proof of the same physical object.
                # Keep the observation but give it a distinct deterministic ID.
                replacement = _new_id("entity", (eid, str(name), str(kind)), used_entities)
                aliases[eid] = replacement
                entity["id"] = replacement
                changed("entities.id", "old ID changed name or kind; preserve as distinct hypothesis",
                        eid, replacement)
                eid = replacement
            if eid in {e["id"] for e in entities}:
                changed("entities", "duplicate entity removed", eid)
                continue
            entities.append(entity)
            if isinstance(name, str) and kind in {"object", "place", "agent"}:
                identities[eid] = (name, kind)
                used_entities.add(eid)
        item["entities"] = entities
        kinds = {e["id"]: e.get("kind") for e in entities}
        relations = []
        for relation in item.get("relations") or []:
            if not isinstance(relation, dict):
                changed("relations", "malformed relation removed")
                continue
            for endpoint in ("source", "target"):
                relation[endpoint] = aliases.get(relation.get(endpoint), relation.get(endpoint))
            a, b, predicate = relation.get("source"), relation.get("target"), relation.get("predicate")
            if (a not in kinds or b not in kinds or a == b or
                    (predicate == "held_by" and (kinds[a] != "object" or kinds[b] != "agent"))):
                changed("relations", "undeclared or invalid relation removed", relation)
                continue
            relations.append(relation)
        item["relations"] = relations

        interaction = item.get("interaction")
        if isinstance(interaction, dict):
            for endpoint in ("object_id", "actor_id", "source_id", "target_id"):
                interaction[endpoint] = aliases.get(interaction.get(endpoint), interaction.get(endpoint))
            obj, actor = interaction.get("object_id"), interaction.get("actor_id")
            if (kinds.get(obj) != "object" or
                    (actor is not None and kinds.get(actor) != "agent")):
                item["interaction"] = None
                changed("interaction", "undeclared object or actor; claim removed", interaction)
                notes.append("Action phase unresolved: referenced object or actor is not declared.")
            else:
                iid = interaction.get("id")
                if iid in cases and cases[iid] != obj:
                    replacement = _new_id("interaction", (str(iid), str(obj)), used_cases)
                    interaction["id"] = replacement
                    changed("interaction.id", "old interaction ID belongs to a different object",
                            iid, replacement)
                    iid = replacement
                if isinstance(iid, str):
                    cases[iid] = obj
                    used_cases.add(iid)
        elif interaction is not None:
            item["interaction"] = None
            changed("interaction", "malformed interaction removed", interaction)
        if item.get("interaction") is None and any(
                r.get("predicate") == "held_by" for r in relations):
            notes.append("Manipulation is visible, but its temporal phase is unresolved.")
        item["uncertainty"] = list(dict.fromkeys(notes))[:10]

        try:
            frame = VisualFrame.model_validate(item)
        except ValueError as exc:
            # Retain observations, but never invent an action to satisfy a rule.
            if item.get("interaction") is not None:
                changed("interaction", "invalid interaction dropped", item["interaction"])
                item["interaction"] = None
                item["uncertainty"] = (item["uncertainty"] +
                    ["Action phase unresolved after schema validation."])[:10]
            try:
                frame = VisualFrame.model_validate(item)
            except ValueError:
                description = item.get("description")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError(f"Frame {index} has no usable visual description") from exc
                changed("structured_fields", "invalid structure dropped; description retained")
                frame = VisualFrame(frame_index=index, description=description[:1200],
                                    clock=item.get("clock") if isinstance(item.get("clock"), str)
                                    and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", item["clock"])
                                    else None,
                                    uncertainty=["Structured visual evidence unavailable; see raw audit."])
        result.append(frame)
    validate_batch([frame.model_dump(mode="json") for frame in result], indices, previous)
    return {"observations": [frame.model_dump(mode="json") for frame in result]}, changes
