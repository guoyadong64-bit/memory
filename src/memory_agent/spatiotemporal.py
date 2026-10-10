"""Explicit context gates: ranking never overrides an incompatible condition."""
from datetime import timedelta, timezone
import json

from .models import Query, Record


def context_key(record: Record) -> str:
    return json.dumps({"conditions":record.conditions,"time":record.time.model_dump(mode="json") if record.time else None,
                       "space":record.space.model_dump(mode="json") if record.space else None,
                       "event":record.event,"required_state":record.required_state,
                       "evidence_requirements":[r.model_dump(mode="json") for r in record.evidence_requirements],
                       "evolution":record.evolution.model_dump(mode="json") if record.evolution else None},
                      sort_keys=True,ensure_ascii=False)


def evidence_check(record: Record, query: Query) -> dict:
    """Use only current observations; historical outcome data must not prove current state."""
    if not record.evidence_requirements:
        return {"reason":None,"matched_observations":[]}
    if query.at is None or query.space is None:
        return {"reason":"current_evidence_context_missing","matched_observations":[]}
    matched = []
    for requirement in record.evidence_requirements:
        entities = {requirement.entity} if requirement.entity else set(query.space.entities)
        if not entities.issubset(query.space.entities):
            return {"reason":"required_entity_missing","matched_observations":[]}
        relevant = [o for o in query.observations
            if o.namespace == query.space.namespace and o.entity in entities
            and o.metric == requirement.metric and o.unit == requirement.unit
            and (requirement.modality is None or o.modality == requirement.modality)
            and o.quality >= requirement.min_quality
            and 0 <= (query.at-o.observed_at).total_seconds() <= requirement.max_age_seconds]
        if not relevant:
            return {"reason":"current_evidence_missing_or_stale","matched_observations":[]}
        # Align to the latest relevant observation rather than cherry-picking an older passing cohort.
        anchor = max(o.observed_at for o in relevant)
        cohort = [o for o in relevant if (anchor-o.observed_at).total_seconds() <= requirement.alignment_seconds]
        latest = {}
        for o in sorted(cohort,key=lambda o:o.observed_at):
            latest[(o.source,o.entity)] = o
        cohort = list(latest.values())
        # The same entity/metric observed by different sources can disagree. Never silently average.
        for i,a in enumerate(cohort):
            for b in cohort[i+1:]:
                if a.source != b.source and a.entity == b.entity:
                    disagree = (abs(a.value-b.value) > requirement.conflict_tolerance
                                if isinstance(a.value,float) and isinstance(b.value,float)
                                else a.value != b.value)
                    if disagree:
                        return {"reason":"current_evidence_conflict","matched_observations":[]}
        if any(len({o.source for o in cohort if o.entity == entity}) < requirement.min_sources for entity in entities):
            return {"reason":"insufficient_aligned_sources","matched_observations":[]}
        for o in cohort:
            if requirement.equals is not None and o.value != requirement.equals:
                return {"reason":"observation_value_mismatch","matched_observations":[]}
            if requirement.minimum is not None or requirement.maximum is not None:
                if (not isinstance(o.value,float)
                    or (requirement.minimum is not None and o.value < requirement.minimum)
                    or (requirement.maximum is not None and o.value > requirement.maximum)):
                    return {"reason":"observation_value_mismatch","matched_observations":[]}
        matched.extend(cohort)
    if matched:
        span = (max(o.observed_at for o in matched)-min(o.observed_at for o in matched)).total_seconds()
        if span > min(r.alignment_seconds for r in record.evidence_requirements):
            return {"reason":"cross_metric_time_misalignment","matched_observations":[]}
    return {"reason":None,"matched_observations":sorted({o.id for o in matched})}


def spatial_score(record: Record, query: Query) -> float:
    if record.space is None:
        return 0.0
    current = query.space
    if current is None or current.namespace != record.space.namespace:
        return -1.0
    facts = {(r.source,r.predicate,r.target) for r in current.relations}
    required = {(r.source,r.predicate,r.target) for r in record.space.relations}
    if not required.issubset(facts):
        return -1.0
    scope = set(record.space.entities)
    if scope.intersection(current.entities):
        return 1.0
    # Only explicit directional containment propagates scope. Nearby/adjacent does not imply applicability.
    reached = set(current.entities)
    for _ in range(len(facts)+1):
        added = {target for source,predicate,target in facts if predicate == "within" and source in reached}
        if added.issubset(reached):
            break
        reached.update(added)
    return 0.5 if scope.intersection(reached) else -1.0


def context_rejection(record: Record, query: Query) -> str | None:
    expected = (record.evolution.stages[0].state if record.evolution else {}) | record.required_state
    if any(query.state.get(k) != v for k,v in expected.items()):
        return "state_mismatch_or_missing"
    if record.evolution:
        if query.phase != record.evolution.stages[0].phase:
            return "phase_mismatch_or_missing"
        if query.at is None or record.evolution.stages[-1].at > query.at:
            return "historical_outcome_not_yet_available"
    if record.event is not None and record.event != query.event:
        return "event_mismatch_or_missing"
    t = record.time
    if t is not None:
        if query.at is None:
            return "query_time_missing"
        if t.event_at and t.event_at > query.at:
            return "future_observation"
        if t.applicable_from and query.at < t.applicable_from:
            return "before_applicability"
        if t.applicable_until and query.at >= t.applicable_until:
            return "after_applicability"
        local = query.at.astimezone(timezone(timedelta(minutes=t.utc_offset_minutes)))
        if t.weekdays and local.weekday() not in t.weekdays:
            return "weekday_mismatch"
        if t.local_start_minute is not None:
            minute = local.hour*60+local.minute
            start,end = t.local_start_minute,t.local_end_minute
            inside = start <= minute < end if start < end else minute >= start or minute < end
            if not inside:
                return "periodic_window_mismatch"
    if spatial_score(record,query) < 0:
        return "space_mismatch_or_missing"
    return evidence_check(record,query)["reason"]


def ranking_context(record: Record, query: Query) -> dict:
    temporal = 0.0
    if record.time is not None:
        temporal = 1.0
        if record.time.event_at and query.at:
            days = max(0,(query.at-record.time.event_at).total_seconds()/86400)
            temporal = 1/(1+days)
    return {"temporal":temporal,"spatial":max(0,spatial_score(record,query)),
            "phase":float(record.evolution is not None),
            "state":float(len(record.required_state)),
            "multi_source":float(len(record.evidence_requirements)),
            "conditions":len(record.conditions),"event":float(record.event is not None)}
