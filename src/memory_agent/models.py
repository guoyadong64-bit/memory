from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def now() -> datetime:
    return datetime.now(timezone.utc)


class TemporalContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_at: datetime | None = None
    applicable_from: datetime | None = None
    applicable_until: datetime | None = None
    weekdays: list[int] = Field(default_factory=list)
    local_start_minute: int | None = Field(default=None,ge=0,lt=1440)
    local_end_minute: int | None = Field(default=None,ge=0,lt=1440)
    utc_offset_minutes: int = Field(default=0,ge=-720,le=840)

    @field_validator("event_at","applicable_from","applicable_until")
    @classmethod
    def aware(cls,value):
        if value is not None and value.tzinfo is None:
            raise ValueError("Temporal timestamps must include a timezone.")
        return value

    @model_validator(mode="after")
    def constraints(self):
        if self.applicable_from and self.applicable_until and self.applicable_until <= self.applicable_from:
            raise ValueError("Applicability interval must have positive length.")
        if len(set(self.weekdays)) != len(self.weekdays) or any(d < 0 or d > 6 for d in self.weekdays):
            raise ValueError("Weekdays must be unique integers from Monday=0 to Sunday=6.")
        if (self.local_start_minute is None) != (self.local_end_minute is None):
            raise ValueError("Periodic time window requires both endpoints.")
        if self.local_start_minute is not None and self.local_start_minute == self.local_end_minute:
            raise ValueError("Equal periodic endpoints are ambiguous; omit them for all day.")
        return self


class SpatialRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    target: str = Field(min_length=1)


class SpatialContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    namespace: str = Field(min_length=1)
    entities: list[str] = Field(min_length=1,max_length=100)
    relations: list[SpatialRelation] = Field(default_factory=list,max_length=200)

    @model_validator(mode="after")
    def unique(self):
        if any(not e.strip() for e in self.entities) or len(set(self.entities)) != len(self.entities):
            raise ValueError("Spatial entities must be nonempty and unique.")
        return self


class Observation(BaseModel):
    """Normalized observation; raw payloads remain in external data systems."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: str | float
    unit: str = Field(min_length=1)
    observed_at: datetime
    provenance: str = Field(min_length=1)
    quality: float = Field(default=1,ge=0,le=1)

    @field_validator("observed_at")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None:
            raise ValueError("Observation timestamps require a timezone.")
        return value

    @field_validator("value")
    @classmethod
    def finite(cls, value):
        import math
        if isinstance(value,float) and not math.isfinite(value):
            raise ValueError("Observation values must be finite.")
        return value


class EvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    modality: str | None = None
    entity: str | None = None
    minimum: float | None = Field(default=None,allow_inf_nan=False)
    maximum: float | None = Field(default=None,allow_inf_nan=False)
    equals: str | None = None
    min_sources: int = Field(default=1,ge=1,le=20)
    max_age_seconds: int = Field(default=300,ge=0,le=604800)
    alignment_seconds: int = Field(default=60,ge=0,le=86400)
    min_quality: float = Field(default=.8,ge=0,le=1)
    conflict_tolerance: float = Field(default=0,ge=0,allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent(self):
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Evidence range is reversed.")
        if self.equals is not None and (self.minimum is not None or self.maximum is not None):
            raise ValueError("Choose a categorical value or numeric range.")
        return self


class Stage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    at: datetime
    state: dict[str,str] = Field(default_factory=dict)
    action: str | None = None
    outcome: str | None = None
    observation_ids: list[str] = Field(default_factory=list)

    @field_validator("at")
    @classmethod
    def aware(cls,value):
        if value.tzinfo is None:
            raise ValueError("Stage timestamps require a timezone.")
        return value


class Evolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    namespace: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    stages: list[Stage] = Field(min_length=2,max_length=100)
    observation_alignment_seconds: int = Field(default=300,ge=0,le=86400)

    @model_validator(mode="after")
    def ordered(self):
        if len({s.id for s in self.stages}) != len(self.stages):
            raise ValueError("Stage IDs must be unique within a trajectory.")
        if any(a.at >= b.at for a,b in zip(self.stages,self.stages[1:])):
            raise ValueError("Trajectory stages must be strictly chronological.")
        return self


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    key: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    conditions: dict[str, str] = Field(default_factory=dict)
    source: str = Field(min_length=1)
    verified: bool = False
    kind: Literal["fact", "experience"] = "fact"
    scope: Literal["long", "short"] = "long"
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=now)
    valid_until: datetime | None = None
    superseded_by: str | None = None
    failures: list[str] = Field(default_factory=list)
    time: TemporalContext | None = None
    space: SpatialContext | None = None
    event: str | None = Field(default=None,min_length=1)
    action: str | None = Field(default=None,min_length=1)
    outcome: str | None = Field(default=None,min_length=1)
    observations: list[Observation] = Field(default_factory=list,max_length=1000)
    evolution: Evolution | None = None
    required_state: dict[str,str] = Field(default_factory=dict)
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list,max_length=50)

    @model_validator(mode="after")
    def validate_observations(self):
        indexed = {o.id:o for o in self.observations}
        if len(indexed) != len(self.observations):
            raise ValueError("Observation IDs must be unique within a record.")
        if self.evolution:
            if any(k in self.evolution.stages[0].state and self.evolution.stages[0].state[k] != v
                   for k,v in self.required_state.items()):
                raise ValueError("Required state contradicts the initial trajectory state.")
            for stage in self.evolution.stages:
                for oid in stage.observation_ids:
                    if oid not in indexed:
                        raise ValueError("Stage references an unknown observation.")
                    if indexed[oid].namespace != self.evolution.namespace:
                        raise ValueError("Stage evidence belongs to a different namespace.")
                    delta = (stage.at-indexed[oid].observed_at).total_seconds()
                    if not 0 <= delta <= self.evolution.observation_alignment_seconds:
                        raise ValueError("Stage evidence must precede and align with its stage.")
        return self

    @model_validator(mode="after")
    def validate_scope(self) -> "Record":
        if self.scope == "short" and (not self.session_id or self.valid_until is None):
            raise ValueError("Short memory requires session_id and valid_until.")
        if self.scope == "long" and self.session_id is not None:
            raise ValueError("Long memory must not be bound to a session.")
        return self

    @field_validator("id")
    @classmethod
    def forbid_reserved_id(cls, value: str) -> str:
        if value in {"synthesize", "abstain"}:
            raise ValueError("Record id is reserved for a decision option.")
        return value

    @field_validator("created_at", "valid_until")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("Timestamps must include a timezone.")
        return value


class Node(BaseModel):
    id: str
    kind: Literal["intent", "condition", "answer", "evidence", "feedback",
                  "time", "space", "spatial_relation", "event", "action", "outcome",
                  "observation", "source", "process", "stage", "state", "requirement", "transition"]
    value: str


class Edge(BaseModel):
    id: str
    kind: Literal["episode", "family", "invalidation", "spatial_relation",
                  "observation", "stage_evidence", "state_transition", "process"]
    members: list[str]
    record_ids: list[str]


class Graph(BaseModel):
    version: Literal[1,2,3] = 3
    records: dict[str, Record] = Field(default_factory=dict)
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: dict[str, Edge] = Field(default_factory=dict)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1)
    key: str | None = None
    conditions: dict[str, str] = Field(default_factory=dict)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    at: datetime | None = None
    space: SpatialContext | None = None
    event: str | None = Field(default=None,min_length=1)
    phase: str | None = Field(default=None,min_length=1)
    state: dict[str,str] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list,max_length=1000)

    @model_validator(mode="after")
    def unique_observations(self):
        if len({o.id for o in self.observations}) != len(self.observations):
            raise ValueError("Query observation IDs must be unique.")
        seen = {}
        for o in self.observations:
            key = (o.namespace,o.source,o.entity,o.metric,o.unit,o.observed_at)
            if key in seen and seen[key] != o.value:
                raise ValueError("One source reports inconsistent values at the same timestamp.")
            seen[key] = o.value
        return self

    @field_validator("at")
    @classmethod
    def aware(cls,value):
        if value is not None and value.tzinfo is None:
            raise ValueError("Query time must include a timezone.")
        return value


class Judgment(BaseModel):
    action: Literal["reuse", "synthesize", "abstain"]
    record_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    model: str
    usage: dict = Field(default_factory=dict)
    support: float = Field(default=0,ge=0,le=2)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class Result(BaseModel):
    query: Query
    route: Literal["reuse", "synthesize", "abstain"]
    answer: Answer
    candidate_ids: list[str]
    judgment: Judgment | None = None
    events: list[dict] = Field(default_factory=list)
    latency_ms: float
    mode: str
    retrieval: dict = Field(default_factory=dict)


class ValidationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex)
    category: str = Field(min_length=1)
    input_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    verifier_version: str = Field(min_length=1)
    passed: bool
    details: dict
    created_at: datetime = Field(default_factory=now)
