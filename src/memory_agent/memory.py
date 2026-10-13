from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Callable

from .models import Edge, Graph, Node, Query, Record, Result, ValidationEvidence, now
from .spatiotemporal import context_key, context_rejection, ranking_context, evidence_check
from .semantic import lexical_scores, query_text, record_text, validate_vectors, vector_scores
from .retrieval import binding, context_coverage
from .recall import graph_plan


def terms(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9_]+", text.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(run[i:i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            words.add(run)
    return words


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records (
 id TEXT PRIMARY KEY, intent TEXT NOT NULL, scope TEXT NOT NULL CHECK(scope IN ('long','short')),
 session_id TEXT, verified INTEGER NOT NULL, valid_until REAL, created_at REAL NOT NULL,
 superseded_by TEXT, failure_count INTEGER NOT NULL, payload TEXT NOT NULL,
 CHECK((scope='long' AND session_id IS NULL) OR
       (scope='short' AND session_id IS NOT NULL AND valid_until IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS record_lookup ON records(intent,scope,session_id,verified);
CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, kind TEXT NOT NULL, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hyperedges (id TEXT PRIMARY KEY, kind TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hyperedge_members (
 edge_id TEXT NOT NULL REFERENCES hyperedges(id) ON DELETE CASCADE,
 node_id TEXT NOT NULL REFERENCES nodes(id), PRIMARY KEY(edge_id,node_id)
);
CREATE TABLE IF NOT EXISTS hyperedge_records (
 edge_id TEXT NOT NULL REFERENCES hyperedges(id) ON DELETE CASCADE,
 record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
 PRIMARY KEY(edge_id,record_id)
);
CREATE INDEX IF NOT EXISTS edge_record_lookup ON hyperedge_records(record_id);
CREATE TABLE IF NOT EXISTS session_turns (
 id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
 created_at REAL NOT NULL, valid_until REAL NOT NULL, query_json TEXT NOT NULL, result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turn_lookup ON session_turns(session_id,id);
CREATE TABLE IF NOT EXISTS validation_evidence (
 id TEXT PRIMARY KEY, category TEXT NOT NULL, input_hash TEXT NOT NULL,
 verifier_version TEXT NOT NULL, passed INTEGER NOT NULL, created_at REAL NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS record_evidence (
 record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
 evidence_id TEXT NOT NULL REFERENCES validation_evidence(id), PRIMARY KEY(record_id,evidence_id)
);
CREATE TABLE IF NOT EXISTS embedding_cache (
 record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
 provider TEXT NOT NULL, content_hash TEXT NOT NULL, vector_json TEXT NOT NULL,
 PRIMARY KEY(record_id,provider)
);
"""


class HypergraphMemory:
    """SQLite hypergraph: normalized relations, transactional incremental writes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.suffix.lower() == ".json":
            raise ValueError("Use a SQLite store path; import old JSON with migrate-json.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3, 4, 5, 6):
                raise ValueError("Unsupported memory database schema version.")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            if version < 6:
                for row in db.execute("SELECT payload FROM records").fetchall():
                    self._index_context(db, Record.model_validate_json(row[0]))
            db.execute("PRAGMA user_version=6")

    @contextmanager
    def connection(self, write: bool = False):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _family_id(record: Record) -> str:
        # Keep legacy long-memory IDs so imported hyperedges remain usable.
        label = record.key if record.scope == "long" else json.dumps(
            [record.scope, record.session_id, record.key], ensure_ascii=False)
        prefix = "family:" if record.scope == "long" else "family:short:"
        return prefix + hashlib.sha256(label.encode()).hexdigest()[:24]

    @staticmethod
    def _node(db, kind: str, value: str) -> str:
        nid = hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()[:24]
        db.execute("INSERT OR IGNORE INTO nodes VALUES (?,?,?)", (nid, kind, value))
        return nid

    @staticmethod
    def _edge(db, eid: str, kind: str, members: list[str], record_ids: list[str]):
        db.execute("INSERT OR IGNORE INTO hyperedges VALUES (?,?)", (eid, kind))
        db.executemany("INSERT OR IGNORE INTO hyperedge_members VALUES (?,?)", [(eid,n) for n in members])
        db.executemany("INSERT OR IGNORE INTO hyperedge_records VALUES (?,?)", [(eid,r) for r in record_ids])

    def _context_members(self, db, record: Record) -> list[str]:
        members = []
        if record.time:
            members.append(self._node(db, "time", record.time.model_dump_json()))
        if record.space:
            space = record.space
            def entity(value):
                return self._node(db, "space", json.dumps([space.namespace, value], ensure_ascii=False))
            members.extend(entity(value) for value in space.entities)
            for relation in space.relations:
                # Encode roles explicitly: unordered hyperedge membership cannot express direction.
                value = json.dumps([space.namespace, relation.source, relation.predicate,
                                    relation.target], ensure_ascii=False)
                relation_node = self._node(db, "spatial_relation", value)
                endpoints = [entity(relation.source), entity(relation.target), relation_node]
                eid = "spatial:" + hashlib.sha256(value.encode()).hexdigest()[:24]
                self._edge(db, eid, "spatial_relation", endpoints, [record.id])
                members.extend(endpoints)
        for kind in ("event", "action", "outcome"):
            value = getattr(record, kind)
            if value:
                members.append(self._node(db, kind, value))
        members += self._evolution_members(db, record)
        members += self._identity_members(db, record)
        return list(dict.fromkeys(members))

    def _identity_members(self, db, record: Record) -> list[str]:
        def node(kind, value):
            return self._node(db, kind, json.dumps(value, ensure_ascii=False, sort_keys=True))
        observed = {o.id: node("observation", o.model_dump(mode="json")) for o in record.observations}
        identities, members = {}, []
        for entity in record.entities:
            identity = node("entity", [entity.namespace, entity.id])
            identities[entity.id] = identity
            members.append(identity)
            for label in entity.labels:
                lid = node("entity_label", label.model_dump(mode="json"))
                group = [identity, lid, *[observed[oid] for oid in label.observation_ids]]
                self._edge(db, "identity:" + identity + ":" + lid, "entity_identity", group, [record.id])
                members.extend(group)
        for role in record.role_bindings:
            rid = node("role", [record.scope, record.session_id, record.id, role.model_dump(mode="json")])
            group = [identities[role.entity_id], rid, *[observed[oid] for oid in role.observation_ids]]
            if record.evolution and role.stage_id:
                process = node("process", [record.scope, record.session_id,
                    record.evolution.namespace, record.evolution.case_id])
                stage = next(s for s in record.evolution.stages if s.id == role.stage_id)
                group.extend([process, node("stage", {"process": process, **stage.model_dump(mode="json")})])
            self._edge(db, "role:" + rid, "role_binding", group, [record.id])
            members.extend(group)
        return members

    def _evolution_members(self, db, record: Record) -> list[str]:
        members, observations = [], {}
        def node(kind, value):
            return self._node(db, kind, json.dumps(value,ensure_ascii=False,sort_keys=True))
        for observation in record.observations:
            oid = node("observation", observation.model_dump(mode="json"))
            observations[observation.id] = oid
            group = [oid, node("source",[observation.namespace,observation.source,observation.modality]),
                     node("space",[observation.namespace,observation.entity]),
                     node("time",{"observed_at":observation.observed_at.isoformat()})]
            self._edge(db,"observation:"+oid,"observation",group,[record.id])
            members.extend(group)
        members.extend(node("requirement",r.model_dump(mode="json")) for r in record.evidence_requirements)
        if record.required_state:
            members.append(node("state",record.required_state))
        if not record.evolution:
            return members
        evolution = record.evolution
        # Process identity includes scope/session; session-only episodes must not merge across sessions.
        process = node("process",[record.scope,record.session_id,evolution.namespace,evolution.case_id])
        stage_nodes = []
        for stage in evolution.stages:
            sid = node("stage",{"process":process,**stage.model_dump(mode="json")})
            stage_nodes.append(sid)
            group = [process,sid,node("state",stage.state),node("time",{"stage_at":stage.at.isoformat()})]
            if stage.action:
                group.append(self._node(db,"action",stage.action))
            if stage.outcome:
                group.append(self._node(db,"outcome",stage.outcome))
            group.extend(observations[oid] for oid in stage.observation_ids)
            self._edge(db,"stage:"+sid,"stage_evidence",group,[record.id])
            members.extend(group)
        for index,(before,after) in enumerate(zip(stage_nodes,stage_nodes[1:])):
            transition = node("transition",{"from":before,"to":after,"relation":"precedes",
                "action":evolution.stages[index].action})
            group = [process,before,after,transition]
            self._edge(db,"transition:"+transition,"state_transition",group,[record.id])
            members.extend(group)
        self._edge(db,"process:"+process,"process",[process,*stage_nodes],[record.id])
        return members

    def trace(self, namespace: str, case_id: str, session_id: str | None = None) -> list[dict]:
        """Diagnostic process trace, with session isolation and explicit source provenance."""
        with self.connection() as db:
            rows = db.execute("SELECT DISTINCT r.payload FROM records r JOIN hyperedge_records er "
                "ON er.record_id=r.id JOIN hyperedges e ON e.id=er.edge_id "
                "WHERE e.kind='process' AND (r.scope='long' OR r.session_id=?) "
                "AND (r.valid_until IS NULL OR r.valid_until>?)",(session_id,now().timestamp())).fetchall()
        result = []
        for row in rows:
            record = Record.model_validate_json(row[0])
            evolution = record.evolution
            if evolution and evolution.namespace == namespace and evolution.case_id == case_id:
                result.append({"record_id":record.id,"verified":record.verified,
                    "superseded_by":record.superseded_by,"failures":record.failures,
                    "evolution":evolution.model_dump(mode="json"),
                    "observations":[o.model_dump(mode="json") for o in record.observations]})
        return sorted(result,key=lambda r:datetime.fromisoformat(r["evolution"]["stages"][0]["at"]))

    def _index_context(self, db, record: Record):
        members = self._context_members(db, record)
        if members:
            self._edge(db, record.id, "episode", members, [record.id])
            self._edge(db, self._family_id(record), "family", members, [record.id])

    @staticmethod
    def _write_record(db, record: Record, insert: bool = False):
        values = (record.key, record.scope, record.session_id, int(record.verified),
                  record.valid_until.timestamp() if record.valid_until else None,
                  record.created_at.timestamp(), record.superseded_by, len(record.failures),
                  record.model_dump_json(), record.id)
        if insert:
            db.execute("INSERT INTO records(intent,scope,session_id,verified,valid_until,created_at,"
                       "superseded_by,failure_count,payload,id) VALUES (?,?,?,?,?,?,?,?,?,?)", values)
        else:
            db.execute("UPDATE records SET intent=?,scope=?,session_id=?,verified=?,valid_until=?,"
                       "created_at=?,superseded_by=?,failure_count=?,payload=? WHERE id=?", values)

    @staticmethod
    def _get(db, record_id: str) -> Record:
        row = db.execute("SELECT payload FROM records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(record_id)
        return Record.model_validate_json(row[0])

    def get(self, record_id: str) -> Record:
        with self.connection() as db:
            return self._get(db, record_id)

    def _invalidate(self, db, old: Record, new: Record):
        old.superseded_by = new.id
        self._write_record(db, old)
        self._edge(db, f"supersedes:{old.id}:{new.id}", "invalidation",
                   [self._node(db,"answer",old.answer), self._node(db,"answer",new.answer),
                    self._node(db,"feedback","explicit verified update")], [old.id,new.id])

    def _add(self, db, record: Record) -> str:
        record = Record.model_validate(record.model_dump())
        if db.execute("SELECT 1 FROM records WHERE id=?", (record.id,)).fetchone():
            raise ValueError("Record id already exists.")
        superseded = []
        if record.verified and record.kind == "fact":
            rows = db.execute("SELECT payload FROM records WHERE intent=? AND scope=? "
                              "AND session_id IS ? AND verified=1 AND superseded_by IS NULL",
                              (record.key,record.scope,record.session_id)).fetchall()
            for row in rows:
                old = Record.model_validate_json(row[0])
                if old.kind == "fact" and context_key(old) == context_key(record):
                    if record.created_at < old.created_at:
                        raise ValueError("An older record cannot supersede a newer assertion.")
                    superseded.append(old)
        self._write_record(db,record,insert=True)
        members = [self._node(db,"intent",record.key), self._node(db,"answer",record.answer),
                   self._node(db,"evidence",record.source)]
        members += [self._node(db,"condition",json.dumps([k,v],ensure_ascii=False))
                    for k,v in sorted(record.conditions.items())]
        members.append(self._node(db,"feedback","verified" if record.verified else "unverified"))
        members += self._context_members(db, record)
        self._edge(db,record.id,"episode",members,[record.id])
        self._edge(db,self._family_id(record),"family",members,[record.id])
        for old in superseded:
            self._invalidate(db,old,record)
        return record.id

    def add(self, record: Record) -> str:
        return self.add_many([record])[0]

    def add_many(self, records: list[Record]) -> list[str]:
        with self.connection(write=True) as db:
            return [self._add(db,r) for r in records]

    def save_validation(self, evidence: ValidationEvidence, record: Record | None = None) -> None:
        if record and record.verified and not evidence.passed:
            raise ValueError("Failed validation cannot support verified memory.")
        with self.connection(write=True) as db:
            db.execute("INSERT INTO validation_evidence VALUES (?,?,?,?,?,?,?)",
                       (evidence.id,evidence.category,evidence.input_hash,evidence.verifier_version,
                        int(evidence.passed),evidence.created_at.timestamp(),evidence.model_dump_json()))
            if record:
                self._add(db,record)
                db.execute("INSERT INTO record_evidence VALUES (?,?)",(record.id,evidence.id))

    def get_validation(self, evidence_id: str) -> ValidationEvidence:
        with self.connection() as db:
            row = db.execute("SELECT payload FROM validation_evidence WHERE id=?",(evidence_id,)).fetchone()
        if row is None:
            raise KeyError(evidence_id)
        return ValidationEvidence.model_validate_json(row[0])

    def has_validation(self, record_id: str, evidence_id: str) -> bool:
        with self.connection() as db:
            return db.execute("SELECT 1 FROM record_evidence WHERE record_id=? AND evidence_id=?",
                              (record_id,evidence_id)).fetchone() is not None

    def fail(self, record_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("Failure feedback must include a reason.")
        with self.connection(write=True) as db:
            record = self._get(db,record_id)
            record.failures.append(reason)
            self._write_record(db,record)
            self._edge(db,f"failure:{record_id}:{len(record.failures)}","invalidation",
                       [self._node(db,"answer",record.answer),self._node(db,"feedback",reason)], [record_id])

    @staticmethod
    def eligible(record: Record, query: Query) -> bool:
        return (record.verified and not record.superseded_by and not record.failures
                and (record.valid_until is None or record.valid_until > now())
                and (record.scope == "long" or record.session_id == query.session_id)
                and (query.key is None or record.key == query.key)
                and all(query.conditions.get(k) == v for k,v in record.conditions.items())
                and context_rejection(record, query) is None)

    def reusable(self, record: Record, query: Query, strategy: str = "hypergraph") -> bool:
        if not self.eligible(record,query):
            return False
        if strategy == "flat":
            return True
        with self.connection() as db:
            checked = binding(db,record,query)
        return checked["episode_bound"] and checked["trajectory_bound"]

    @staticmethod
    def retrieval_scores(record: Record, query: Query) -> dict:
        scores = ranking_context(record, query)
        scores["semantic"] = float(len(terms(query.question) & terms(record.question)))
        # Transparent heuristic, not a learned or calibrated probability.
        scores["total"] = sum(scores[k] for k in
            ("semantic","temporal","spatial","event","phase","state","multi_source"))
        return scores

    def _semantic(self, query, records, embedder, namespace):
        if embedder is None:
            return lexical_scores(query,records),"tfidf",None
        from .clients import ModelError
        try:
            if not namespace:
                raise ValueError("Embedding provider identity is required.")
            texts = {r.id:record_text(r) for r in records}
            hashes = {rid:hashlib.sha256(t.encode()).hexdigest() for rid,t in texts.items()}
            cached = {}
            with self.connection() as db:
                for r in records:
                    row = db.execute("SELECT vector_json FROM embedding_cache WHERE record_id=? "
                                     "AND provider=? AND content_hash=?",(r.id,namespace,hashes[r.id])).fetchone()
                    if row:
                        cached[r.id] = validate_vectors([json.loads(row[0])],1)[0]
            missing = [r for r in records if r.id not in cached]
            pending = [query_text(query),*[texts[r.id] for r in missing]]
            fresh = []
            for start in range(0,len(pending),64):
                batch = pending[start:start+64]
                fresh.extend(validate_vectors(embedder(batch),len(batch)))
            fresh = validate_vectors(fresh,len(pending))
            new = {r.id:v for r,v in zip(missing,fresh[1:])}
            vectors = cached | new
            scores = vector_scores(fresh[0],[vectors[r.id] for r in records],records)
            # Never keep a transaction open while calling a remote model.
            with self.connection(write=True) as db:
                for rid,vector in new.items():
                    db.execute("INSERT OR REPLACE INTO embedding_cache "
                        "SELECT ?,?,?,? WHERE EXISTS (SELECT 1 FROM records WHERE id=?)",
                        (rid,namespace,hashes[rid],json.dumps(vector),rid))
            return scores,"embedding",None
        except (ModelError,ValueError,TypeError):
            # Failure details can contain provider data; expose only a stable reason.
            return lexical_scores(query,records),"tfidf","embedding_unavailable_or_invalid"

    def search(self, query: Query, top_k: int = 5, *, strategy: str = "hypergraph",
               embedder: Callable | None = None, embedding_namespace: str = "",
               minimum_similarity: float = .05) -> tuple[list[Record],dict]:
        if strategy not in {"hypergraph","flat"} or not 1 <= top_k <= 100 or not 0 <= minimum_similarity <= 1:
            raise ValueError("Invalid retrieval options.")
        with self.connection() as db:
            rows = db.execute("SELECT DISTINCT r.payload FROM records r "
                "JOIN hyperedge_records er ON er.record_id=r.id JOIN hyperedges e ON e.id=er.edge_id "
                "WHERE e.kind='family' AND r.verified=1 AND r.superseded_by IS NULL AND r.failure_count=0 "
                "AND (r.valid_until IS NULL OR r.valid_until>?) AND (r.scope='long' OR r.session_id=?) "
                "AND (? IS NULL OR r.intent=?)",
                (now().timestamp(),query.session_id,query.key,query.key)).fetchall()
        candidates = [Record.model_validate_json(row[0]) for row in rows]
        filtered = [{"record_id":r.id,"reason":context_rejection(r,query) or "lifecycle_or_conditions"}
                    for r in candidates if not self.eligible(r,query)]
        candidates = [r for r in candidates if self.eligible(r,query)]
        # Temporary assertions override equivalent long assertions only in this session.
        local = {(r.key,context_key(r)) for r in candidates if r.scope == "short"}
        candidates = [r for r in candidates if r.scope == "short" or
                      (r.key,context_key(r)) not in local]
        report = {"strategy":strategy,"semantic_backend":"tfidf","fallback":None,
                  "eligible_count":len(candidates),"filtered":filtered,"scores":{},
                  "query_plan": query.recall_plan.model_dump(mode="json") if query.recall_plan else None}
        if not candidates:
            return [],report
        semantic,backend,fallback = self._semantic(query,candidates,embedder,embedding_namespace)
        report.update(semantic_backend=backend,fallback=fallback)
        planned_ids = set()
        report["plan_evidence"] = None
        bound = {}
        if strategy == "hypergraph":
            with self.connection() as db:
                bound = {r.id:binding(db,r,query) for r in candidates}
            usable = []
            for r in candidates:
                b = bound[r.id]
                if not b["episode_bound"] or not b["trajectory_bound"]:
                    report["filtered"].append({"record_id":r.id,"reason":"hyperedge_binding_incomplete"})
                else:
                    usable.append(r)
            candidates = usable
        if query.purpose == "recall" and query.recall_plan is not None:
            # Execute the same explicit task constraints for both retrieval baselines.
            # Hypergraph binding remains an additional structural check, not a different
            # interpretation of the user's requested source/destination/ordinal.
            report["plan_evidence"] = graph_plan(candidates, query)
            planned_ids = set(report["plan_evidence"].get("record_ids", []))
            if planned_ids and not report["plan_evidence"].get("warnings"):
                candidates = [r for r in candidates if r.id in planned_ids]
        direct = {r.id for r in candidates if query.key is not None or semantic[r.id] > minimum_similarity}
        seeds = sorted((r for r in candidates if r.id in direct),
                       key=lambda r:semantic[r.id],reverse=True)[:max(1,top_k)]
        expanded = []
        for r in candidates:
            paths = []
            if strategy == "hypergraph":
                for seed in seeds:
                    if seed.id == r.id or seed.key != r.key:
                        continue
                    shared = bound[r.id]["matched_nodes"].keys() & bound[seed.id]["matched_nodes"].keys()
                    types = {bound[r.id]["matched_nodes"][nid] for nid in shared}
                    # A shared topic/source alone must never expand a candidate.
                    if len(types) >= 2:
                        paths.append({"seed_record_id":seed.id,"shared_node_ids":sorted(shared),
                                      "shared_types":sorted(types)})
            if r.id not in direct and not paths and r.id not in planned_ids:
                report["filtered"].append({"record_id":r.id,"reason":"no_semantic_or_joint_graph_match"})
                continue
            coverage = context_coverage(r,query)
            # Bounded evidence for multi-role co-binding, not a probability or graph degree bonus.
            support = min(1.0,max((len(p["shared_types"])/4 for p in paths),default=0.0))
            graph_support = .5*coverage + .5*support if strategy == "hypergraph" else coverage
            total = .55*semantic[r.id] + .30*coverage + .15*graph_support
            if r.id in planned_ids:
                total += .45  # executed ordinal/relative evidence outranks mere semantic proximity
            score = self.retrieval_scores(r,query) | {
                "semantic":round(semantic[r.id],6),"context_coverage":round(coverage,6),
                "joint_graph_support":round(graph_support,6),"total":round(total,6),
                "recall_route":"structured_plan" if r.id in planned_ids else
                    "semantic_or_task" if r.id in direct else "joint_hyperedge_expansion",
                "shared_paths":paths,"binding":bound.get(r.id)}
            report["scores"][r.id] = score
            expanded.append(r)
        expanded.sort(key=lambda r:(report["scores"][r.id]["total"],len(r.conditions),
                                   r.scope == "short",r.created_at),reverse=True)
        planned_targets = len((report.get("plan_evidence") or {}).get("target_record_ids", []))
        result_limit = min(100, max(top_k, planned_targets)) if (
            query.recall_plan and query.recall_plan.all_targets and planned_ids and
            not report["plan_evidence"].get("warnings")) else top_k
        chosen = expanded[:result_limit]
        report["scores"] = {r.id:report["scores"][r.id] for r in chosen}
        return chosen,report

    def retrieve(self, query: Query, top_k: int = 5) -> list[Record]:
        return self.search(query,top_k)[0]

    def explain(self, query: Query) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT payload FROM records WHERE (? IS NULL OR intent=?) "
                "AND (scope='long' OR session_id=?)",(query.key,query.key,query.session_id)).fetchall()
        result = []
        for row in rows:
            record = Record.model_validate_json(row[0])
            reason = context_rejection(record,query)
            if not self.eligible(record,query) and reason is None:
                reason = "lifecycle_or_condition_mismatch"
            result.append({"record_id":record.id,"eligible":self.eligible(record,query),
                           "reason":reason,"evidence":evidence_check(record,query)})
        return result

    def promote(self, record_id: str, source: str) -> str:
        if not source.strip():
            raise ValueError("Promotion requires an explicit confirmation source.")
        with self.connection(write=True) as db:
            old = self._get(db,record_id)
            # Promotion confirms storage scope, not applicability to a fabricated current query.
            if (old.scope != "short" or not old.verified or old.superseded_by or old.failures
                    or old.valid_until is None or old.valid_until <= now()):
                raise ValueError("Only active verified short memory can be promoted.")
            new = Record(key=old.key,question=old.question,answer=old.answer,
                         conditions=old.conditions,source=source,verified=True,kind=old.kind,
                         time=old.time, space=old.space, event=old.event,
                         action=old.action, outcome=old.outcome,observations=old.observations,
                         entities=old.entities, role_bindings=old.role_bindings,
                         evolution=old.evolution,required_state=old.required_state,
                         evidence_requirements=old.evidence_requirements)
            self._add(db,new)
            self._invalidate(db,old,new)
            db.execute("INSERT INTO record_evidence SELECT ?,evidence_id FROM record_evidence WHERE record_id=?",
                       (new.id,old.id))
            return new.id

    def remember_turn(self, result: Result, ttl_minutes: int = 60, limit: int = 20) -> None:
        if not 1 <= ttl_minutes <= 10080 or not 1 <= limit <= 100:
            raise ValueError("Invalid session TTL or history limit.")
        session = result.query.session_id
        if session is None:
            return
        with self.connection(write=True) as db:
            stamp = now()
            db.execute("INSERT INTO session_turns(session_id,created_at,valid_until,query_json,result_json) "
                       "VALUES (?,?,?,?,?)", (session,stamp.timestamp(),
                       (stamp+timedelta(minutes=ttl_minutes)).timestamp(),
                       result.query.model_dump_json(),result.model_dump_json()))
            db.execute("DELETE FROM session_turns WHERE session_id=? AND (valid_until<=? OR id NOT IN "
                       "(SELECT id FROM session_turns WHERE session_id=? ORDER BY id DESC LIMIT ?))",
                       (session,stamp.timestamp(),session,limit))

    def history(self, session_id: str, limit: int = 20) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT query_json,result_json FROM session_turns WHERE session_id=? "
                              "AND valid_until>? ORDER BY id DESC LIMIT ?",
                              (session_id,now().timestamp(),limit)).fetchall()
        return [{"question":json.loads(r[0])["question"], "answer":json.loads(r[1])["answer"]["text"],
                 "route":json.loads(r[1])["route"]} for r in reversed(rows)]

    def prune(self) -> dict:
        with self.connection(write=True) as db:
            stamp = now().timestamp()
            dead = [r[0] for r in db.execute("SELECT id FROM records WHERE scope='short' AND valid_until<=?",(stamp,))]
            for rid in dead:
                db.execute("DELETE FROM hyperedges WHERE kind IN ('episode','invalidation') "
                           "AND id IN (SELECT edge_id FROM hyperedge_records WHERE record_id=?)",(rid,))
                db.execute("DELETE FROM records WHERE id=?",(rid,))
            db.execute("DELETE FROM hyperedges WHERE id NOT IN (SELECT edge_id FROM hyperedge_records)")
            db.execute("DELETE FROM hyperedge_members WHERE edge_id IN (SELECT id FROM hyperedges WHERE kind='process')")
            for row in db.execute("SELECT payload FROM records").fetchall():
                record = Record.model_validate_json(row[0])
                if record.evolution:
                    self._evolution_members(db, record)
            db.execute("DELETE FROM hyperedge_members WHERE edge_id IN (SELECT id FROM hyperedges WHERE kind='family')")
            db.execute("INSERT OR IGNORE INTO hyperedge_members SELECT family.edge_id,member.node_id "
                       "FROM hyperedge_records family JOIN hyperedges e ON e.id=family.edge_id "
                       "JOIN hyperedge_members member ON member.edge_id=family.record_id WHERE e.kind='family'")
            db.execute("DELETE FROM nodes WHERE id NOT IN (SELECT node_id FROM hyperedge_members)")
            turns = db.execute("DELETE FROM session_turns WHERE valid_until<=?",(stamp,)).rowcount
        return {"expired_short_records_removed":len(dead),"expired_turns_removed":turns}

    @property
    def graph(self) -> Graph:
        """Read-only diagnostic snapshot, not the mutable storage cache."""
        with self.connection() as db:
            db.execute("BEGIN")
            graph = Graph()
            graph.records = {r[0]:Record.model_validate_json(r[1]) for r in db.execute("SELECT id,payload FROM records")}
            graph.nodes = {r[0]:Node(id=r[0],kind=r[1],value=r[2]) for r in db.execute("SELECT * FROM nodes")}
            for row in db.execute("SELECT * FROM hyperedges"):
                graph.edges[row[0]] = Edge(id=row[0],kind=row[1],
                    members=[r[0] for r in db.execute("SELECT node_id FROM hyperedge_members WHERE edge_id=?",(row[0],))],
                    record_ids=[r[0] for r in db.execute("SELECT record_id FROM hyperedge_records WHERE edge_id=?",(row[0],))])
            return graph

    def counts(self) -> dict:
        with self.connection() as db:
            db.execute("BEGIN")
            return {"storage":"sqlite", "path":str(self.path),
                "schema_version":db.execute("PRAGMA user_version").fetchone()[0],
                "cached_embeddings":db.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0],
                "node_types":dict(db.execute("SELECT kind,COUNT(*) FROM nodes GROUP BY kind").fetchall()),
                "edge_types":dict(db.execute("SELECT kind,COUNT(*) FROM hyperedges GROUP BY kind").fetchall()),
                "records":db.execute("SELECT COUNT(*) FROM records").fetchone()[0],
                "long_records":db.execute("SELECT COUNT(*) FROM records WHERE scope='long'").fetchone()[0],
                "short_records":db.execute("SELECT COUNT(*) FROM records WHERE scope='short'").fetchone()[0],
                "nodes":db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
                "hyperedges":db.execute("SELECT COUNT(*) FROM hyperedges").fetchone()[0],
                "validation_evidence":db.execute("SELECT COUNT(*) FROM validation_evidence").fetchone()[0],
                "active_session_turns":db.execute("SELECT COUNT(*) FROM session_turns WHERE valid_until>?",(now().timestamp(),)).fetchone()[0]}

    def migrate_json(self, legacy: Path) -> dict:
        content = legacy.read_bytes()
        graph = Graph.model_validate_json(content)
        if any(k != r.id for k,r in graph.records.items()) or any(k != n.id for k,n in graph.nodes.items()):
            raise ValueError("Legacy graph identifiers are inconsistent.")
        for k,edge in graph.edges.items():
            if (k != edge.id or any(n not in graph.nodes for n in edge.members)
                    or any(r not in graph.records for r in edge.record_ids)):
                raise ValueError("Legacy graph contains dangling or inconsistent references.")
        for record in graph.records.values():
            edge = graph.edges.get(record.id)
            family = graph.edges.get(self._family_id(record))
            if not edge or edge.kind != "episode" or not family or record.id not in family.record_ids:
                raise ValueError("Legacy record lacks its episode or family hyperedge.")
        digest = hashlib.sha256(content).hexdigest()
        with self.connection(write=True) as db:
            previous = db.execute("SELECT value FROM metadata WHERE key='legacy_sha256'").fetchone()
            if previous and previous[0] == digest:
                return {"imported":0,"already_migrated":True}
            if db.execute("SELECT 1 FROM records LIMIT 1").fetchone():
                raise ValueError("JSON migration requires an empty SQLite record store.")
            for record in graph.records.values():
                self._write_record(db,record,insert=True)
            db.executemany("INSERT OR IGNORE INTO nodes VALUES (?,?,?)",[(n.id,n.kind,n.value) for n in graph.nodes.values()])
            for edge in graph.edges.values():
                self._edge(db,edge.id,edge.kind,edge.members,edge.record_ids)
            for record in graph.records.values():
                self._index_context(db, record)
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('legacy_sha256',?)",(digest,))
        return {"imported":len(graph.records),"already_migrated":False,"source_preserved":True}
