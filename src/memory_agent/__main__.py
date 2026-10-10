from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sqlite3
import sys

from pydantic import ValidationError

from .agents import MemoryCoordinator
from .memory import HypergraphMemory
from .models import Query, Record, now
from .settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone evidence-memory agent prototype")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--mode", choices=["mock", "live"])
    parser.add_argument("--store", type=Path)
    parser.add_argument("--retrieval-strategy",choices=["hypergraph","flat"])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Check config without any network requests")
    sub.add_parser("probe", help="Test configured APIs with synthetic data (may incur API cost)")
    sub.add_parser("demo", help="Run an isolated, offline project-preference example")
    ingest = sub.add_parser("ingest", help="Add explicit evidence (one record or a JSON list)")
    ingest.add_argument("file", type=Path)
    ingest.add_argument("--scope", choices=["long", "short"])
    ingest.add_argument("--session")
    ingest.add_argument("--ttl-minutes", type=int)
    ask = sub.add_parser("ask")
    ask.add_argument("question", nargs="?")
    ask.add_argument("--query-file", type=Path, help="JSON Query; avoids shell JSON escaping")
    ask.add_argument("--key")
    ask.add_argument("--session")
    ask.add_argument("--conditions", default="{}", help="JSON string-to-string map")
    failure = sub.add_parser("feedback")
    failure.add_argument("record_id")
    failure.add_argument("--reason", required=True)
    migration = sub.add_parser("migrate-json", help="Import old graph into empty SQLite; keep source")
    migration.add_argument("file", type=Path)
    promotion = sub.add_parser("promote", help="Explicitly confirm short memory as long memory")
    promotion.add_argument("record_id")
    promotion.add_argument("--source", required=True)
    history = sub.add_parser("history", help="Show unexpired dialogue for one session")
    history.add_argument("--session", required=True)
    sub.add_parser("prune", help="Remove expired short records and dialogue")
    sub.add_parser("inspect", help="Show graph counts, never credentials")
    sub.add_parser("graph", help="Show normalized graph snapshot (includes memory contents)")
    retrieval = sub.add_parser("retrieve", help="Offline context filtering and ranking")
    retrieval.add_argument("--query-file", type=Path, required=True)
    comparison = sub.add_parser("compare-retrieval",help="Compare flat and hypergraph recall offline with identical gates")
    comparison.add_argument("--query-file",type=Path,required=True)
    explanation = sub.add_parser("explain", help="Explain lifecycle/context/evidence filtering")
    explanation.add_argument("--query-file", type=Path, required=True)
    trace = sub.add_parser("trace", help="Inspect an event trajectory without model calls")
    trace.add_argument("--namespace", required=True)
    trace.add_argument("--case-id", required=True)
    trace.add_argument("--session")
    normalize = sub.add_parser("normalize", help="Map heterogeneous JSON rows to normalized observations")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--mapping", type=Path, required=True)
    normalize.add_argument("--output", type=Path)
    evidence = sub.add_parser("evidence", help="Read independent validation evidence")
    evidence.add_argument("evidence_id")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env)
        if args.mode:
            settings.mode = args.mode
        if args.store:
            settings.store_path = args.store
        if args.retrieval_strategy:
            settings.retrieval_strategy = args.retrieval_strategy
        if args.command == "probe":
            from .probe import probe
            report = probe(settings)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not report["ok"]:
                raise SystemExit(1)
            return
        if args.command == "check":
            if settings.mode == "live":
                settings.validate_live()
            print(json.dumps({"mode": settings.mode, "llm_model": settings.llm_model,
                              "jev_model": settings.jev_model,
                              "llm_key_present": bool(settings.llm_key.strip()),
                              "jev_key_present": bool(settings.jev_key.strip()),
                              "store_path": str(settings.store_path),
                              "short_ttl_minutes": settings.short_ttl_minutes,
                              "session_turn_limit": settings.session_turn_limit,
                              "retrieval_strategy":settings.retrieval_strategy,
                              "embedding_model":settings.embedding_model or None,
                              "network_called": False}, ensure_ascii=False, indent=2))
            return
        if args.command == "demo":
            # An isolated temporary directory prevents demo data from polluting real memory.
            from uuid import uuid4
            demo_root = Path("runtime/memory_agent/demos").resolve()
            demo_root.mkdir(parents=True, exist_ok=True)
            directory = demo_root / uuid4().hex
            directory.mkdir()
            try:
                settings.mode = "mock"
                settings.store_path = directory / "memory.sqlite3"
                agent = MemoryCoordinator(settings)
                try:
                    record = Record(key="report.format", question="项目报告用什么格式？",
                                    answer="使用 Markdown。", conditions={"project": "demo"},
                                    source="演示用户明确说明", verified=True)
                    agent.memory.add(record)
                    result = agent.ask(Query(question=record.question, key=record.key,
                                             conditions=record.conditions))
                    print(result.model_dump_json(indent=2))
                finally:
                    agent.close()
            finally:
                # Only delete exact database files belonging to this isolated demo.
                for name in ("memory.sqlite3", "memory.sqlite3-wal", "memory.sqlite3-shm"):
                    (directory / name).unlink(missing_ok=True)
                directory.rmdir()
            return
        if args.command == "ask":
            if args.query_file:
                if args.question or args.key or args.conditions != "{}":
                    raise ValueError("Use --query-file alone, without question/key/conditions.")
                query = Query.model_validate_json(args.query_file.read_text(encoding="utf-8"))
                if args.session:
                    if query.session_id is not None and query.session_id != args.session:
                        raise ValueError("--session conflicts with query file session_id.")
                    query.session_id = args.session
            else:
                if not args.question:
                    raise ValueError("Provide a question or --query-file.")
                query = Query(question=args.question, key=args.key,
                              conditions=json.loads(args.conditions), session_id=args.session)
            agent = MemoryCoordinator(settings)
            try:
                print(agent.ask(query).model_dump_json(indent=2))
            finally:
                agent.close()
            return
        if args.command == "normalize":
            from .observations import ObservationMapping, normalize_rows
            mapping = ObservationMapping.model_validate_json(args.mapping.read_text(encoding="utf-8"))
            rows = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(rows,list):
                raise ValueError("Raw observations must be a JSON list.")
            output = json.dumps([o.model_dump(mode="json") for o in normalize_rows(rows,mapping)],
                                ensure_ascii=False,indent=2)
            if args.output:
                args.output.write_text(output,encoding="utf-8")
            else:
                print(output)
            return
        memory = HypergraphMemory(settings.store_path)
        if args.command == "ingest":
            raw = json.loads(args.file.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else [raw]
            records = []
            ttl = args.ttl_minutes if args.ttl_minutes is not None else settings.short_ttl_minutes
            if not 1 <= ttl <= 10080:
                raise ValueError("TTL must be between 1 and 10080 minutes.")
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("Each record must be a JSON object.")
                item = dict(item)
                if args.scope:
                    item["scope"] = args.scope
                if args.session:
                    item["session_id"] = args.session
                if item.get("scope") == "short":
                    if "valid_until" not in item or args.ttl_minutes is not None:
                        item["valid_until"] = (now() + timedelta(minutes=ttl)).isoformat()
                elif args.ttl_minutes is not None or args.session:
                    raise ValueError("Session and TTL flags are only for short memory.")
                records.append(Record.model_validate(item))
            print(json.dumps({"record_ids": memory.add_many(records)}, indent=2))
        elif args.command == "feedback":
            memory.fail(args.record_id, args.reason)
            print("Failure recorded; this record will be excluded from retrieval.")
        elif args.command == "migrate-json":
            print(json.dumps(memory.migrate_json(args.file), indent=2))
        elif args.command == "promote":
            print(json.dumps({"long_record_id": memory.promote(args.record_id, args.source)}, indent=2))
        elif args.command == "history":
            print(json.dumps(memory.history(args.session, settings.session_turn_limit), ensure_ascii=False, indent=2))
        elif args.command == "prune":
            print(json.dumps(memory.prune(), indent=2))
        elif args.command == "evidence":
            print(memory.get_validation(args.evidence_id).model_dump_json(indent=2))
        elif args.command == "graph":
            print(memory.graph.model_dump_json(indent=2))
        elif args.command == "trace":
            print(json.dumps(memory.trace(args.namespace,args.case_id,args.session),ensure_ascii=False,indent=2))
        elif args.command == "explain":
            query = Query.model_validate_json(args.query_file.read_text(encoding="utf-8"))
            print(json.dumps(memory.explain(query),ensure_ascii=False,indent=2))
        elif args.command == "retrieve":
            query = Query.model_validate_json(args.query_file.read_text(encoding="utf-8"))
            records,report = memory.search(query,settings.top_k,strategy=settings.retrieval_strategy,
                                          minimum_similarity=settings.semantic_min_similarity)
            print(json.dumps({"records":[r.model_dump(mode="json") for r in records],
                              "retrieval":report},ensure_ascii=False,indent=2))
        elif args.command == "compare-retrieval":
            query = Query.model_validate_json(args.query_file.read_text(encoding="utf-8"))
            output = {}
            for strategy in ("flat","hypergraph"):
                records,report = memory.search(query,settings.top_k,strategy=strategy,
                    minimum_similarity=settings.semantic_min_similarity)
                output[strategy] = {"record_ids":[r.id for r in records],"retrieval":report}
            print(json.dumps(output,ensure_ascii=False,indent=2))
        else:
            print(json.dumps(memory.counts(), indent=2))
    except (ValueError, KeyError, OSError, ValidationError, sqlite3.Error) as error:
        # Pydantic error dumps can contain submitted data; print field names, not values.
        if isinstance(error, ValidationError):
            fields = [".".join(map(str, e["loc"])) for e in error.errors()]
            message = "Invalid fields: " + ", ".join(fields)
        elif isinstance(error, (OSError, KeyError, sqlite3.Error)):
            message = "File, database or record could not be accessed; check path and database locks."
        else:
            message = str(error)
        print(message, file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
