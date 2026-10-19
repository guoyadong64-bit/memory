"""Resume the paid validation-50 perception run with versioned, audited ID handling.

Original checkpoints and model responses stay untouched. This command creates a
new run directory, reuses their visual evidence and starts new API calls only at
the first unsaved batch. No questions or answers enter the vision pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from benchmarks import prepare_validation50_observations as stage
from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_memory import validate_batch
from benchmarks.findingdory_pilot import load
from benchmarks.findingdory_structured import extract_batch, freeze
from benchmarks.validation50_resilience import normalize_observations
from memory_agent.settings import Settings


SOURCE = stage.DEFAULT_SHARED
DESTINATION = SOURCE.with_name(SOURCE.name + "_v2")


def _same_or_copy(source: Path, target: Path, *, link=False):
    if target.exists():
        if stage.checksum(target) != stage.checksum(source):
            raise RuntimeError(f"Already copied file differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _live_extractor(client, frames, ep, indices, previous, diagnostic, **kwargs):
    audit_dir = diagnostic.parents[2] / "normalizations" / ep

    def normalize(raw, expected, history):
        output, changes = normalize_observations(raw, expected, history)
        if changes:
            raw_digest = hashlib.sha256(json.dumps(
                raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            name = ("_".join(f"{index:03d}" for index in expected) +
                    "_" + raw_digest[:16] + ".json")
            record = {"episode": ep, "indices": expected,
                      "raw_output": raw, "raw_sha256": raw_digest, "changes": changes,
                      "normalized_sha256": hashlib.sha256(json.dumps(
                          output, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                      "used_answer_labels": False}
            path = audit_dir / name
            if path.exists() and load(path) != record:
                raise RuntimeError(f"Normalization audit changed: {path}")
            if not path.exists():
                atomic_json(path, record)
        return output

    return extract_batch(client, frames, ep, indices, previous, diagnostic,
                         normalizer=normalize, **kwargs)


def _reuse_failed_response(source: Path, destination: Path, ep: str, start: int,
                           previous: list):
    diagnostic = source / "perception" / "validation_checks" / ep / f"{start:03d}.json"
    if not diagnostic.exists():
        return False
    candidates = []
    for attempt in load(diagnostic):
        raw = attempt.get("output")
        try:
            normalized, changes = normalize_observations(raw, [start, start + 1], previous)
            validate_batch(normalized["observations"], [start, start + 1], previous)
            candidates.append((len(changes), -sum(len(f["entities"]) + len(f["relations"])
                                                  + int(f["interaction"] is not None)
                                                  for f in normalized["observations"]),
                               attempt["attempt"], raw, normalized, changes))
        except (ValueError, KeyError, TypeError):
            continue
    if not candidates:
        return False
    _, _, attempt_number, raw, normalized, changes = min(candidates,
        key=lambda item: item[:3])
    checkpoint = destination / "perception" / "batches" / ep / f"{start:03d}.json"
    atomic_json(checkpoint, normalized["observations"])
    atomic_json(destination / "perception" / "normalizations" / ep /
                f"{start:03d}_source_recovery.json", {
                    "episode": ep, "indices": [start, start + 1],
                    "source_diagnostic": str(diagnostic.resolve()),
                    "source_diagnostic_sha256": stage.checksum(diagnostic),
                    "source_attempt": attempt_number, "raw_output": raw,
                    "changes": changes, "checkpoint_sha256": stage.checksum(checkpoint),
                    "used_answer_labels": False, "new_model_requests": 0})
    return True


def bootstrap(data: Path, source: Path, destination: Path, settings):
    if source.resolve() == destination.resolve():
        raise ValueError("V2 run directory must differ from the original")
    manifest = stage.selected_data(data)
    old = load(source / "config.json")
    if (old["dataset"] != manifest or old["vision_url"] != settings.llm_url or
            old["vision_model"] != (settings.vision_model or settings.llm_model) or
            old["credential_fingerprint"] != stage.credential_fingerprint(source, settings.llm_key)):
        raise RuntimeError("Source run dataset or model credentials changed")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Missing ffmpeg")
    stage.freeze(destination / "manifest.json", manifest)
    current = stage.freeze(destination / "config.json", stage.frozen_config(
        settings, destination, manifest, stage.executable_version(ffmpeg)))
    resilience = freeze(destination / "resilience_config.json", {
        "protocol": "findingdory-validation50-resilient-v2",
        "source_directory": str(source.resolve()),
        "source_config_fingerprint": old["fingerprint"],
        "destination_config_fingerprint": current["fingerprint"],
        "normalizer_sha256": stage.checksum(Path(__file__).with_name("validation50_resilience.py")),
        "runner_sha256": stage.checksum(Path(__file__)),
        "policy": "ID repair is deterministic; uncertain unsupported structure is dropped with audit; no answer labels"})
    snapshot = destination / "source_reuse.json"
    if snapshot.exists():
        prior = load(snapshot)
        if prior["resilience_fingerprint"] != resilience["fingerprint"]:
            raise RuntimeError("V2 source snapshot configuration changed")
        for name, digest in prior["reused_sha256"].items():
            if stage.checksum(source / name) != digest:
                raise RuntimeError(f"Reused source file changed: {name}")
        return

    reused = {}
    for ep in manifest["episodes"]:
        original_frames = source / "frames" / ep
        if (original_frames / "frames_manifest.json").exists():
            video = f"videos/val/{ep}.mp4"
            stage.ensure_frames(data, source, ep, video,
                                manifest["videos"][video]["sha256"], ffmpeg,
                                stage.executable_version(ffmpeg))
            for file in original_frames.iterdir():
                if file.is_file():
                    _same_or_copy(file, destination / "frames" / ep / file.name,
                                  link=file.suffix == ".jpg")
        complete = source / "observations" / f"{ep}.json"
        if complete.exists():
            visual = load(complete)
            validate_batch(visual["observations"], list(range(96)))
            reused[str(complete.relative_to(source))] = stage.checksum(complete)
            visual["configuration_fingerprint"] = current["fingerprint"]
            visual["source_observation_sha256"] = stage.checksum(complete)
            visual["provenance"] = "V2 reuse of paid V1 visual observations; no answer labels"
            atomic_json(destination / "observations" / f"{ep}.json", visual)
            continue
        previous = []
        for start in range(0, 96, 2):
            original = source / "perception" / "batches" / ep / f"{start:03d}.json"
            if original.exists():
                rows = load(original)
                parsed = validate_batch(rows, [start, start + 1], previous)
                reused[str(original.relative_to(source))] = stage.checksum(original)
                atomic_json(destination / "perception" / "batches" / ep / original.name, rows)
                previous.extend(parsed)
            elif _reuse_failed_response(source, destination, ep, start, previous):
                break
            else:
                break
    atomic_json(snapshot, {"resilience_fingerprint": resilience["fingerprint"],
                           "reused_sha256": reused,
                           "source_requests_started": load(source / "perception" /
                                                            "api_budget.json")["requests_started"],
                           "source_reported_tokens": load(source / "perception" /
                                                            "api_budget.json")["reported_tokens"]})
    print(f"[复用] 旧观测和断点已复制到 {destination}；原目录保持不变", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=stage.DEFAULT_DATA)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--run-dir", type=Path, default=DESTINATION)
    parser.add_argument("--check-only", action="store_true", help="只迁移并验证断点，不调用模型")
    parser.add_argument("--max-requests", type=int, default=10000)
    parser.add_argument("--max-reported-tokens", type=int, default=0)
    args = parser.parse_args(argv)
    data, source, destination = (p.resolve() for p in
                                 (args.data_dir, args.source_dir, args.run_dir))
    settings = Settings.load(stage.REPO / ".env")
    bootstrap(data, source, destination, settings)
    if args.check_only:
        print("[预检完成] 未调用模型；去掉 --check-only 即可从 V2 断点继续", flush=True)
        return
    print("[提示] 从 V2 断点继续；以下请求/token 仅统计 V2 新调用，旧用量见 source_reuse.json",
          flush=True)
    stage.run(data, destination, settings, max_requests=args.max_requests,
              max_reported_tokens=args.max_reported_tokens,
              extractor=_live_extractor)


if __name__ == "__main__":
    main()
