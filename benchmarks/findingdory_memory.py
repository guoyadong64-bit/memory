"""Validated, label-blind visual evidence to generic spatiotemporal memory.

Object identity and interactions are VLM hypotheses, not oracle annotations.
Dates are episode-relative anchors; the visible clock supplies time-of-day.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from memory_agent.identity import contextual_labels, direct_lift

from memory_agent.models import (Evolution, Observation, Query, Record, SpatialContext,
                                 SpatialRelation, Stage, TemporalContext, MemoryEntity,
                                 EntityLabel, RoleBinding)

SCHEMA_VERSION = "findingdory-role-memory-v11"
MIN_CONFIDENCE = .8
ORIGIN = datetime(2000, 1, 1, tzinfo=timezone.utc)


class VisualLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0, le=1)


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    name: str = Field(min_length=1, max_length=160)
    kind: Literal["object", "place", "agent"]
    labels: list[VisualLabel] = Field(default_factory=list, max_length=10)
    attributes: dict[str, str] = Field(default_factory=dict)


class Relation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    predicate: Literal["on", "inside", "near", "held_by"]
    target: str
    confidence: float = Field(ge=0, le=1)


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    object_id: str
    actor_id: str | None = None
    phase: Literal["approach", "grasp", "transport", "release", "settled"]
    action: Literal["approach", "pick_up", "carry", "release", "place"] | None = None
    source_id: str | None = None
    target_id: str | None = None
    outcome: str | None = Field(default=None, min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)


class VisualFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_index: int = Field(ge=0, lt=96, strict=True)
    description: str = Field(min_length=1, max_length=1200)
    clock: str | None = Field(default=None, pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    entities: list[Entity] = Field(default_factory=list, max_length=30)
    relations: list[Relation] = Field(default_factory=list, max_length=50)
    interaction: Interaction | None = None
    uncertainty: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def references(self):
        kinds = {e.id: e.kind for e in self.entities}
        if len(kinds) != len(self.entities):
            raise ValueError("Duplicate entity IDs")
        for r in self.relations:
            missing = {r.source, r.target} - kinds.keys()
            if missing:
                raise ValueError(f"Relation endpoints {sorted(missing)} are not declared endpoints; "
                                 f"add them to entities or correct IDs to {sorted(kinds)}")
            if r.source == r.target:
                raise ValueError("Spatial relation requires distinct declared endpoints")
            if r.predicate == "held_by" and (kinds[r.source] != "object" or kinds[r.target] != "agent"):
                raise ValueError(
                    f"Invalid relation {r.source} held_by {r.target}: source kind={kinds[r.source]}, "
                    f"target kind={kinds[r.target]}. held_by must be OBJECT -> AGENT actively gripping "
                    "the object. Furniture/surfaces are not gripping agents. Re-read this image: "
                    "use on only for visible surface support, inside only for visible containment, "
                    "near only for visible proximity, or omit the unsupported relation and explain "
                    "uncertainty. Do not swap endpoints or relabel furniture as an agent.")
        if self.interaction:
            i = self.interaction
            if kinds.get(i.object_id) != "object":
                raise ValueError("Interaction requires a declared object")
            if i.actor_id is not None and kinds.get(i.actor_id) != "agent":
                raise ValueError("Interaction actor must be an agent")
            # Catch an explicit self-contradiction such as naming a visible purple
            # object with an ID whose fixed registry name denotes another color.
            # This only fires when the description ties a color/object phrase to
            # the manipulated ID; background colors alone are not identity proof.
            item = next(e for e in self.entities if e.id == i.object_id)
            colors = {"black", "white", "purple", "brown", "red", "green", "blue",
                      "yellow", "orange", "pink", "beige", "grey", "gray"}
            named_colors = colors & set(re.findall(r"[a-z]+", item.name.lower()))
            mention = re.search(r"\b" + re.escape(i.object_id) + r"\b", self.description, re.I)
            if i.confidence >= MIN_CONFIDENCE and named_colors and mention:
                prefix = self.description[max(0, mention.start() - 65):mention.start()].lower()
                if re.search(r"\b(?:object|toy|figure|ring|cylinder|cup|ball|bottle|box|"
                             r"plush|coil|item)\s*\(?\s*$", prefix):
                    visible_colors = re.findall(r"\b(?:" + "|".join(sorted(colors)) + r")\b", prefix)
                    if visible_colors and visible_colors[-1] not in named_colors:
                        raise ValueError("Appearance conflict: the manipulated object's stated "
                                         "color differs from its registered ID/name; use a new "
                                         "entity and interaction ID if the visible object differs")
            # The model may describe the manipulated object by a noun phrase rather
            # than its ID. A direct action-object phrase is still usable evidence;
            # other colored objects in the scene must not trigger this check.
            action_object = re.search(
                r"\b(?:grasping|holding|carrying|lifting|transporting|placing|releasing|around|"
                r"picking up)\s+(?:a|an|the)\s+((?:[a-z-]+\s+){0,5}"
                r"(?:object|toy|figure|ring|cylinder|cup|ball|bottle|box|plush|coil|item))\b",
                self.description.lower())
            if i.confidence >= MIN_CONFIDENCE and named_colors and action_object:
                described_colors = colors & set(re.findall(r"[a-z]+", action_object.group(1)))
                if described_colors and not described_colors <= named_colors:
                    raise ValueError("Appearance conflict: directly manipulated object's color "
                                     "differs from its declared entity name; keep visible color "
                                     "and assign a distinct descriptive entity ID/name if needed")
            if i.phase != "settled" and i.actor_id is None:
                raise ValueError("Active interaction requires the visible actor as a declared agent; "
                                 "if no actor is evidenced, leave interaction null")
            for place in (i.source_id, i.target_id):
                if place is not None and kinds.get(place) not in {"place", "object"}:
                    raise ValueError(f"Interaction location {place!r} must be a declared place or supporting object")
            allowed = {"approach": {None, "approach"}, "grasp": {None, "pick_up"},
                       "transport": {None, "carry"}, "release": {None, "release", "place"},
                       "settled": {None}}
            if i.action not in allowed[i.phase]:
                raise ValueError("Action contradicts phase")
            if i.outcome and i.phase not in {"release", "settled"}:
                raise ValueError("Outcome requires visible release or settled state")
            held = any(r.source == i.object_id and r.predicate == "held_by" and
                       r.target == i.actor_id and r.confidence >= MIN_CONFIDENCE
                       for r in self.relations)
            if (i.confidence >= MIN_CONFIDENCE and i.phase in {"grasp", "transport"}
                    and not held):
                raise ValueError("Grasp/transport requires high-confidence object held_by actor "
                                 "in this image; use approach or null interaction if grip is not visible")
            if i.phase in {"release", "settled"} and any(
                r.source == i.object_id and r.predicate == "held_by" and r.confidence >= MIN_CONFIDENCE
                for r in self.relations):
                raise ValueError("Completed release/settled object cannot still be held_by the agent; "
                                 "correct phase or relation using the current image")
            supported = any(r.source == i.object_id and r.predicate in {"on", "inside"} and
                            r.confidence >= MIN_CONFIDENCE for r in self.relations)
            if i.confidence >= MIN_CONFIDENCE and i.outcome and not supported:
                raise ValueError("Placement outcome requires high-confidence visible on/inside support "
                                 "for the manipulated object; otherwise leave outcome null")
        # A visible held_by relation or action description remains useful evidence
        # even when the temporal interaction phase is uncertain. Do not force the
        # model to invent grasp/transport just to satisfy a schema constraint.
        return self


def validate_batch(raw, expected, previous=()):
    frames = [VisualFrame.model_validate(x) for x in raw]
    if [f.frame_index for f in frames] != list(expected):
        raise ValueError("Visual output omitted, duplicated or reordered indices")
    identities = {}
    cases = {}
    for frame in [*previous, *frames]:
        for e in frame.entities:
            identity = (e.name, e.kind)
            if e.id in identities and identities[e.id] != identity:
                raise ValueError("Stable entity ID changed name or kind")
            identities[e.id] = identity
        if frame.interaction:
            i = frame.interaction
            if i.id in cases and cases[i.id] != i.object_id:
                raise ValueError("Interaction ID changed its object")
            cases[i.id] = i.object_id
    return frames


class VisualBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[VisualFrame]


def frame_times(frames):
    """Preserve read clocks; tolerate missing/regressing OCR without inventing clocks."""
    day = 0
    previous = None
    result = {}
    for f in sorted(frames, key=lambda x: x.frame_index):
        minute = None
        if f.clock:
            hour, mins = map(int, f.clock.split(":"))
            candidate = hour * 60 + mins
            if previous is not None and candidate < previous:
                if previous >= 18 * 60 and candidate <= 6 * 60:
                    day += 1
                else:
                    result[f.frame_index] = (ORIGIN + timedelta(milliseconds=f.frame_index), "frame_sequence")
                    continue
            minute = candidate
            previous = candidate
        if minute is None:
            result[f.frame_index] = (ORIGIN + timedelta(milliseconds=f.frame_index), "frame_sequence")
        else:
            result[f.frame_index] = (ORIGIN + timedelta(days=day, minutes=minute,
                microseconds=f.frame_index), "visible_clock_episode_date")
    return result


def adjacent_settlement(previous: VisualFrame | None, current: VisualFrame) -> Interaction | None:
    """Link an observed hold to an adjacent supported state without editing the image record.

    The transition is a memory inference, not a VLM-labeled release. Requiring the
    same object and actor, a visible open gripper, and one supported destination
    prevents a static object-on-furniture relation from becoming a placement.
    """
    if previous is None or current.interaction is not None or current.frame_index != previous.frame_index + 1:
        return None
    prior = previous.interaction
    if prior is None or prior.confidence < MIN_CONFIDENCE or prior.phase not in {"grasp", "transport"}:
        return None
    held = [r for r in previous.relations if r.source == prior.object_id and
            r.predicate == "held_by" and r.target == prior.actor_id and r.confidence >= MIN_CONFIDENCE]
    if not held or not prior.actor_id:
        return None
    entities = {e.id: e for e in current.entities}
    if prior.object_id not in entities or entities.get(prior.actor_id, None) is None or \
            entities[prior.actor_id].kind != "agent":
        return None
    if any(r.source == prior.object_id and r.predicate == "held_by" and
           r.confidence >= MIN_CONFIDENCE for r in current.relations):
        return None
    supports = [r for r in current.relations if r.source == prior.object_id and
                r.predicate in {"on", "inside"} and r.confidence >= MIN_CONFIDENCE]
    if len({r.target for r in supports}) != 1:
        return None
    if not re.search(r"\b(?:gripper|robot(?:ic)? arm)\b.{0,65}\b(?:open|empty|released)\b",
                     current.description, re.I):
        return None
    return Interaction(id=prior.id, object_id=prior.object_id, actor_id=prior.actor_id,
                       phase="settled", target_id=supports[0].target,
                       confidence=min(prior.confidence, held[0].confidence,
                                      min(r.confidence for r in supports)))


def structured_records(ep, raw_frames):
    frames = validate_batch(raw_frames, [x["frame_index"] for x in raw_frames])
    if [f.frame_index for f in frames] != sorted({f.frame_index for f in frames}):
        raise ValueError("History must contain unique chronological frames")
    times = frame_times(frames)
    records = []
    groups = defaultdict(list)
    previous_segment = None
    inferred_settlements = {}
    phase_order = {phase:n for n,phase in enumerate(["approach", "grasp", "transport", "release", "settled"])}
    evidence = {}
    for index, original in enumerate(frames):
        inferred = adjacent_settlement(frames[index - 1] if index else None, original)
        f = (original.model_copy(update={"interaction": inferred}) if inferred else original)
        if inferred:
            inferred_settlements[f.frame_index] = frames[index - 1].frame_index
        at, basis = times[f.frame_index]
        i = f.interaction if f.interaction and f.interaction.confidence >= MIN_CONFIDENCE else None
        observed_i = (original.interaction if original.interaction and
                      original.interaction.confidence >= MIN_CONFIDENCE else None)
        payload = original.model_dump(mode="json") | {"time_basis": basis, "frame_indices": [f.frame_index]}
        relations = [SpatialRelation(source=r.source, predicate=r.predicate, target=r.target)
                     for r in f.relations if r.confidence >= MIN_CONFIDENCE]
        # Provenance remains frame-bound even when a process later groups frames.
        observation = Observation(id=f"{ep}_image_{f.frame_index}", source=f"{ep}/frames/{f.frame_index:03d}.jpg",
            modality="image", namespace=ep, entity=observed_i.object_id if observed_i else ep,
            metric="visual_state", value=json.dumps(payload, ensure_ascii=False), unit="description",
            observed_at=at, provenance="VLM interpretation of this image; not independently verified",
            quality=observed_i.confidence if observed_i else .5)
        evidence[f.frame_index] = observation
        action = (f"{observed_i.actor_id or 'unknown_agent'}:{observed_i.action}:{observed_i.object_id}"
                  if observed_i and observed_i.action else None)
        entities = memory_entities(ep, [f], evidence)
        roles = [RoleBinding(role="visible", entity_id=e.id,
                 observation_ids=[observation.id], confidence=.8) for e in entities]
        if observed_i and observed_i.phase in {"grasp", "transport", "release", "settled"}:
            roles.append(RoleBinding(role="object", entity_id=observed_i.object_id,
                                     observation_ids=[observation.id], confidence=observed_i.confidence))
            if observed_i.actor_id:
                roles.append(RoleBinding(role="actor", entity_id=observed_i.actor_id,
                                         observation_ids=[observation.id], confidence=observed_i.confidence))
        records.append(Record(id=f"{ep}_frame_{f.frame_index}", key="findingdory.history",
            question=f"Frame {f.frame_index}; clock {f.clock}; {f.description}",
            answer=json.dumps(payload, ensure_ascii=False), source=observation.provenance + "; " + observation.source,
            verified=True, kind="experience", time=TemporalContext(event_at=at),
            space=SpatialContext(namespace=ep, entities=[ep, *[e.id for e in f.entities]], relations=relations),
            event="visual_history", action=action, outcome=observed_i.outcome if observed_i else None,
            observations=[observation], entities=entities, role_bindings=roles))
        if i:
            # Model-written IDs are hypotheses, not reliable event boundaries. Link only
            # the same instance/actor, close frames, monotone phases, and one clock basis.
            open_carry = (previous_segment is not None and
                previous_segment[1].interaction.phase in {"grasp","transport"} and
                i.phase in {"transport","release","settled"} and
                any(r.source == i.object_id and r.predicate == "held_by" and
                    r.target == i.actor_id and r.confidence >= MIN_CONFIDENCE
                    for r in previous_segment[1].relations))
            join = (previous_segment is not None and
                previous_segment[1].interaction.object_id == i.object_id and
                i.actor_id is not None and previous_segment[1].interaction.actor_id == i.actor_id and
                f.frame_index > previous_segment[1].frame_index and
                (f.frame_index - previous_segment[1].frame_index <= 3 or
                 previous_segment[1].interaction.id == i.id or open_carry) and
                times[f.frame_index][1] == times[previous_segment[1].frame_index][1] and
                phase_order[i.phase] >= phase_order[previous_segment[1].interaction.phase] and
                not (previous_segment[1].interaction.phase in {"release", "settled"} and
                     i.phase not in {"release", "settled"}))
            case = previous_segment[0] if join else f"linked_{i.object_id}_{f.frame_index}"
            groups[case].append(f)
            previous_segment = (case, f)
    for case, members in groups.items():
        # Never make one trajectory from mixed clock and frame-sequence axes.
        if len({times[f.frame_index][1] for f in members}) != 1:
            continue
        phases = [f.interaction.phase for f in members]
        order = {phase: n for n, phase in enumerate(["approach", "grasp", "transport", "release", "settled"])}
        # A repeated pick/place episode needs a new interaction ID, not a backwards transition.
        if any(order[b] < order[a] for a, b in zip(phases, phases[1:])):
            continue
        stages = []
        extra_evidence = {}
        inferred_bindings = []
        first_hold_seen = False
        event_source = None
        for f in members:
            i = f.interaction
            state = {"object_id": i.object_id}
            if i.phase == "grasp" and not first_hold_seen and i.source_id and i.source_id != i.object_id:
                state["source_id"] = i.source_id
            if i.target_id and i.target_id != i.object_id:
                state["target_id"] = i.target_id
            for relation in f.relations:
                if relation.source == i.object_id and relation.confidence >= MIN_CONFIDENCE:
                    state[relation.predicate] = relation.target
                    if i.phase in {"release", "settled"} and relation.predicate in {"on", "inside"}:
                        state.setdefault("target_id", relation.target)
            ids = [evidence[f.frame_index].id]
            if f.frame_index in inferred_settlements:
                prior_index = inferred_settlements[f.frame_index]
                ids.append(evidence[prior_index].id)
                inferred_bindings.append({"role": "target_id", "entity": state["target_id"],
                    "observation_ids": ids.copy(), "method": "adjacent_held_to_supported_open_gripper",
                    "direct_action_observed": False})
            if (i.phase in {"grasp", "transport"} and state.get("held_by") and
                    not state.get("source_id") and not any(s.state.get("held_by") for s in stages)):
                prior = [p for p in frames if 0 < f.frame_index-p.frame_index <= 3 and
                         times[p.frame_index][1] == times[f.frame_index][1] and
                         0 <= (times[f.frame_index][0]-times[p.frame_index][0]).total_seconds() <= 300]
                for p in reversed(prior):
                    supports = {r.target for r in p.relations if r.source == i.object_id and
                                r.predicate in {"on", "inside"} and r.confidence >= MIN_CONFIDENCE}
                    if len(supports) == 1:
                        state["source_id"] = next(iter(supports))
                        ids.append(evidence[p.frame_index].id)
                        extra_evidence[p.frame_index] = evidence[p.frame_index]
                        inferred_bindings.append({"role": "source_id", "entity": state["source_id"],
                            "observation_id": evidence[p.frame_index].id, "method": "visible_pre_grasp_support"})
                        break
            if state.get("held_by"):
                if not first_hold_seen:
                    event_source = state.get("source_id")
                first_hold_seen = True
                if event_source:
                    state["source_id"] = event_source
            if state.get("source_id") and state.get("held_by") and direct_lift(f.model_dump(mode="json")):
                # Certify only the local pickup; destination/event completion stays unknown.
                state["pickup_evidence"] = "direct_lift"
            stages.append(Stage(id=f"frame_{f.frame_index}", phase=i.phase, at=times[f.frame_index][0],
                state=state, action=i.action, outcome=i.outcome,
                observation_ids=ids))
        identities = {e.id: e.name for f in members for e in f.entities}
        identities.update({e.id:e.name for f in frames if f.frame_index in extra_evidence for e in f.entities})
        indices = sorted({f.frame_index for f in members} | extra_evidence.keys())
        retained = [f for f in frames if f.frame_index in indices]
        entities = memory_entities(ep, retained, evidence)
        roles = []
        for f, stage in zip(members, stages):
            i = f.interaction
            if i.phase in {"grasp", "transport", "release", "settled"}:
                roles.append(RoleBinding(role="object", entity_id=i.object_id, stage_id=stage.id,
                    observation_ids=[evidence[f.frame_index].id], confidence=i.confidence))
                if i.actor_id:
                    roles.append(RoleBinding(role="actor", entity_id=i.actor_id, stage_id=stage.id,
                        observation_ids=[evidence[f.frame_index].id], confidence=i.confidence))
            if stage.state.get("source_id") and stage.state.get("held_by"):
                # A carried source belongs to the event's first hold and its evidence,
                # never to a later transport location proposed by the model.
                source_stage = next(s for s in stages if s.state.get("source_id") and s.state.get("held_by"))
                roles.append(RoleBinding(role="source", entity_id=stage.state["source_id"],
                    stage_id=stage.id, observation_ids=source_stage.observation_ids,
                    confidence=i.confidence, basis="linked"))
            if i.phase in {"release", "settled"} and stage.state.get("target_id") and any(
                r.source == i.object_id and r.target == stage.state["target_id"] and
                r.predicate in {"on", "inside"} and r.confidence >= MIN_CONFIDENCE for r in f.relations):
                roles.append(RoleBinding(role="destination", entity_id=stage.state["target_id"],
                    stage_id=stage.id, observation_ids=stage.observation_ids,
                    confidence=i.confidence,
                    basis="linked" if f.frame_index in inferred_settlements else "observed"))
        payload = {"interaction_id": case, "frame_indices": indices, "entities": identities,
                   "observations": [json.loads(evidence[n].value) for n in indices],
                   "grouping_policy": "same_instance_actor; close_frames_or_same_declared_event_or_open_carry; monotone_phase; same_time_basis; source_from_first_confirmed_hold; mark_unobserved_intervals",
                   "raw_interaction_ids": sorted({f.interaction.id for f in members}),
                   "inferred_bindings": inferred_bindings,
                   "coverage": {"observed_phases": sorted(set(phases), key=phase_order.get),
                       "source_grounded": any(r.role == "source" for r in roles),
                       "destination_grounded": any(r.role == "destination" for r in roles),
                       "local_pickup_confirmed": any(s.state.get("pickup_evidence") for s in stages),
                       "event_complete": any(s.phase in {"release", "settled"} and s.outcome for s in stages),
                       "unobserved_intervals": [[a.frame_index,b.frame_index] for a,b in zip(members,members[1:])
                                                if b.frame_index-a.frame_index>3],
                       "history_inventory_complete": False}}
        records.append(Record(id=f"{ep}_process_{case}", key="findingdory.history",
            question="Interaction " + case + ": " + "; ".join(f.description for f in members),
            answer=json.dumps(payload, ensure_ascii=False), kind="experience", verified=True,
            source="Grouped VLM frame evidence; temporal order is not causal proof",
            time=TemporalContext(event_at=stages[-1].at),
            space=SpatialContext(namespace=ep, entities=[ep, *identities]), event="visual_history",
            action="; ".join(s.action for s in stages if s.action) or None,
            outcome=next((s.outcome for s in reversed(stages) if s.outcome), None),
            observations=[evidence[n] for n in indices],
            entities=entities, role_bindings=roles,
            evolution=Evolution(namespace=ep, case_id=case, stages=stages)))
    return records


def memory_entities(ep, frames, evidence):
    collected = {}
    for f in frames:
        oid = evidence[f.frame_index].id
        for e in f.entities:
            item = collected.setdefault(e.id, {"name": e.name, "kind": e.kind,
                                               "labels": {}, "attributes": e.attributes})
            for label in e.labels:
                value = item["labels"].setdefault(label.text, {"confidence": label.confidence, "ids": []})
                value["confidence"] = min(value["confidence"], label.confidence)
                value["ids"].append(oid)
        for eid, text, confidence in contextual_labels(f.entities, f.description):
            value = collected[eid]["labels"].setdefault(text, {"confidence": confidence, "ids": []})
            value["confidence"] = min(value["confidence"], confidence)
            if oid not in value["ids"]:
                value["ids"].append(oid)
    return [MemoryEntity(id=eid, namespace=ep, name=v["name"], kind=v["kind"], attributes=v["attributes"],
            labels=[EntityLabel(text=text, confidence=l["confidence"], observation_ids=l["ids"])
                    for text,l in v["labels"].items()]) for eid,v in collected.items()]


def history_query(question, ep, records):
    # Dataset tasks refer to this clip as the preceding historical day. This is an
    # episode-relative anchor, not an assertion of an actual calendar date/timezone.
    available_at = max(r.time.event_at for r in records) + timedelta(days=1, seconds=1)
    return Query(question=question, purpose="recall", output_kind="goal_frames", event="visual_history", at=available_at,
                 space=SpatialContext(namespace=ep, entities=[ep]))


def supported_frames(records):
    return sorted({i for r in records for i in json.loads(r.answer)["frame_indices"]})
