"""Bounded, record-local hyperedge matching. Shared nodes never certify current facts."""
import hashlib
import json

from .models import Query, Record
from .spatiotemporal import evidence_check
from .recall import record_match, hint_score


def nid(kind, value):
    return hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()[:24]


def encoded(kind, value):
    return nid(kind,json.dumps(value,ensure_ascii=False,sort_keys=True))


def context_coverage(record: Record, query: Query) -> float:
    if query.purpose == "recall" and query.recall_plan:
        return record_match(record, query.recall_plan)
    requested = {"task":query.key is not None,"time":query.at is not None,
        "space":query.space is not None,"event":query.event is not None,
        "phase":query.phase is not None,"state":bool(query.state),
        "evidence":bool(query.observations)}
    supported = {"task":True,"time":record.time is not None or record.evolution is not None,
        "space":record.space is not None,"event":record.event is not None,
        "phase":record.evolution is not None,
        "state":bool(record.required_state or (record.evolution and record.evolution.stages[0].state)),
        "evidence":bool(record.evidence_requirements)}
    active = [k for k,v in requested.items() if v]
    return sum(supported[k] for k in active)/len(active) if active else 0.0


def binding(db, record: Record, query: Query) -> dict:
    groups = {}
    for row in db.execute("SELECT e.id,e.kind,n.id,n.kind,n.value FROM hyperedges e "
        "JOIN hyperedge_records er ON er.edge_id=e.id JOIN hyperedge_members m ON m.edge_id=e.id "
        "JOIN nodes n ON n.id=m.node_id WHERE er.record_id=?",(record.id,)):
        group = groups.setdefault(row[0],{"kind":row[1],"nodes":{}})
        group["nodes"][row[2]] = (row[3],row[4])
    episode = groups.get(record.id,{"kind":None,"nodes":{}})
    required = {nid("intent",record.key)}
    anchors = {}
    if query.key:
        anchors["task"] = {nid("intent",record.key)}
    if record.time:
        ids = {nid("time",record.time.model_dump_json())}
        required |= ids
        if query.at:
            anchors["time"] = ids
    if record.space:
        ids = {encoded("space",[record.space.namespace,x]) for x in record.space.entities}
        required |= ids
        if query.space:
            reached = set(query.space.entities)
            for _ in range(len(query.space.relations)+1):
                added = {r.target for r in query.space.relations if r.predicate == "within" and r.source in reached}
                if added <= reached:
                    break
                reached |= added
            anchors["space"] = {encoded("space",[record.space.namespace,x])
                                 for x in record.space.entities if x in reached and
                                 not (query.purpose == "recall" and x == query.space.namespace)}
    for role in ("event","action","outcome"):
        if value := getattr(record,role):
            ids = {nid(role,value)}
            required |= ids
            if role == "event" and query.event:
                anchors["event"] = ids
    if record.required_state:
        ids = {encoded("state",record.required_state)}
        required |= ids
        if query.state:
            anchors["state"] = ids
    evidence = evidence_check(record,query)
    if record.evidence_requirements and query.observations and evidence["reason"] is None:
        ids = {encoded("requirement",r.model_dump(mode="json")) for r in record.evidence_requirements}
        required |= ids
        anchors["evidence"] = ids
    entry_edge, transitions = None, []
    trajectory_ok = True
    observations_bound = True
    for observation in record.observations:
        oid = encoded("observation",observation.model_dump(mode="json"))
        expected = {oid,encoded("source",[observation.namespace,observation.source,observation.modality]),
                    encoded("space",[observation.namespace,observation.entity]),
                    encoded("time",{"observed_at":observation.observed_at.isoformat()})}
        required |= expected
        observations_bound &= expected <= groups.get("observation:"+oid,{"nodes":{}})["nodes"].keys()
    if record.evolution:
        evolution = record.evolution
        process = encoded("process",[record.scope,record.session_id,evolution.namespace,evolution.case_id])
        stage_ids = [encoded("stage",{"process":process,**s.model_dump(mode="json")}) for s in evolution.stages]
        required |= {process,*stage_ids}
        process_group = groups.get("process:"+process,{"nodes":{}})
        trajectory_ok = {process,*stage_ids} <= process_group["nodes"].keys()
        for stage,sid in zip(evolution.stages,stage_ids):
            expected = {process,sid,encoded("state",stage.state),encoded("time",{"stage_at":stage.at.isoformat()})}
            expected |= {nid(k,v) for k in ("action","outcome") if (v := getattr(stage,k))}
            observations = {o.id:o for o in record.observations}
            expected |= {encoded("observation",observations[oid].model_dump(mode="json")) for oid in stage.observation_ids}
            stage_group = groups.get("stage:"+sid,{"nodes":{}})
            trajectory_ok &= expected <= stage_group["nodes"].keys()
        entry_edge = "stage:"+stage_ids[0]
        first = evolution.stages[0]
        if query.phase:
            anchors["phase"] = {stage_ids[0]}
        if query.state and first.state:
            anchors.setdefault("state",set()).add(encoded("state",first.state))
        if query.at:
            anchors.setdefault("time",set()).add(encoded("time",{"stage_at":first.at.isoformat()}))
        for before,after,stage in zip(stage_ids,stage_ids[1:],evolution.stages):
            transition = encoded("transition",{"from":before,"to":after,"relation":"precedes","action":stage.action})
            eid = "transition:"+transition
            transitions.append(eid)
            trajectory_ok &= {process,before,after,transition} <= groups.get(eid,{"nodes":{}})["nodes"].keys()
    episode_ok = episode["kind"] == "episode" and required <= episode["nodes"].keys()
    identity_ok = True
    entity_hint = (query.recall_plan.entity_hint or query.recall_plan.anchor_hint or
                   query.recall_plan.location_hint) if query.recall_plan else None
    for entity in record.entities:
        identity = encoded("entity", [entity.namespace, entity.id])
        required.add(identity)
        identity_ok &= identity in episode["nodes"]
        names = " ".join([entity.name, *[x.text for x in entity.labels if x.confidence >= .8]])
        if entity_hint and hint_score(entity_hint, entity.id, names) >= .5:
            anchors.setdefault("identity", set()).add(identity)
        for label in entity.labels:
            lid = encoded("entity_label", label.model_dump(mode="json"))
            expected = {identity, lid, *[encoded("observation", next(o for o in record.observations
                        if o.id == oid).model_dump(mode="json")) for oid in label.observation_ids]}
            identity_ok &= expected <= groups.get("identity:"+identity+":"+lid, {"nodes":{}})["nodes"].keys()
    for role in record.role_bindings:
        rid = encoded("role", [record.scope,record.session_id,record.id,role.model_dump(mode="json")])
        entity = next(e for e in record.entities if e.id == role.entity_id)
        expected = {encoded("entity", [entity.namespace, entity.id]), rid,
            *[encoded("observation", next(o for o in record.observations if o.id == oid).model_dump(mode="json"))
              for oid in role.observation_ids]}
        identity_ok &= expected <= groups.get("role:"+rid, {"nodes":{}})["nodes"].keys()
    episode_ok &= identity_ok
    matched = {role:sorted(ids & episode["nodes"].keys()) for role,ids in anchors.items()}
    # Only query-compatible context nodes can support bounded neighbourhood expansion.
    expand_types = {"space","event","state","time","requirement","stage","entity"}
    shared = {node_id:episode["nodes"][node_id][0] for ids in matched.values() for node_id in ids
              if episode["nodes"][node_id][0] in expand_types}
    return {"episode_bound":episode_ok and observations_bound,"identity_bound":bool(identity_ok),
            "observations_bound":bool(observations_bound),
            "trajectory_bound":bool(trajectory_ok),
            "matched_roles":matched,"matched_nodes":shared,
            "entry_edge":entry_edge,"transition_edges":transitions,
            "coverage":context_coverage(record,query),
            "current_observation_ids":evidence["matched_observations"]}
