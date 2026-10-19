"""Run one complete FindingDory memory experiment with shared, frozen perception."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
from pathlib import Path
import shutil

from benchmarks.comparison_worker import atomic_json
from benchmarks.findingdory_memory import SCHEMA_VERSION, validate_batch
from benchmarks.findingdory_pilot import ROOT, load
from benchmarks.findingdory_structured import caption_config, fingerprint, freeze
from benchmarks.run_comparison import (REPO, checksum, cleanup_success, ensure_frames,
    environment_check, execute_method, frozen_config, observe, require_live_settings,
    summarize, verify_data)
from memory_agent.settings import Settings


METHODS = ("hypergraph", "mem0", "amem", "readagent")
DEFAULT_SHARED = Path("runtime/experiments/findingdory_shared_perception")
DEFAULT_SEED = Path("runtime/experiments/compare_four_methods")


def credential_fingerprint(directory: Path, settings, *, embedding=False):
    salt_path = directory / "credential_salt.bin"
    if not salt_path.exists():
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        salt_path.write_bytes(os.urandom(32))
    salt = salt_path.read_bytes()
    if len(salt) != 32:
        raise RuntimeError(f"凭据指纹盐文件无效：{salt_path}")
    values = {"llm": settings.llm_key}
    if embedding:
        values["embedding"] = settings.embedding_key or settings.llm_key
    return {name: hmac.new(salt, value.encode(), hashlib.sha256).hexdigest()
            for name, value in values.items()}


def shared_config(settings, data, manifest, tasks, shared):
    sources = [REPO / "benchmarks" / name for name in
               ("findingdory_structured.py", "comparison_worker.py", "run_budget.py")]
    sources.append(REPO / "src" / "memory_agent" / "clients.py")
    return {"protocol": "findingdory-shared-perception-v1", "dataset": manifest,
            "selected_tasks_sha256": fingerprint(tasks), "caption": caption_config(settings, data),
            "vision_url": settings.llm_url, "vision_model": settings.vision_model or settings.llm_model,
            "llm_thinking": settings.llm_thinking,
            "code_sha256": {str(path.relative_to(REPO)): checksum(path) for path in sources},
            "credential_fingerprint": credential_fingerprint(shared, settings)}


def complete_observations(shared, manifest):
    for ep in manifest["episodes"]:
        path = shared / "observations" / f"{ep}.json"
        if not path.exists():
            return False
        try:
            saved = load(path)
            if saved.get("episode") != ep or saved.get("schema_version") != SCHEMA_VERSION:
                return False
            validate_batch(saved["observations"], range(96))
        except (ValueError, KeyError, TypeError):
            return False
    return (shared / "perception" / "api_budget.json").exists()


def copy_verified(source: Path, destination: Path):
    if not source.exists():
        raise FileNotFoundError(f"共享视觉观测缺失：{source}")
    expected = checksum(source)
    if destination.exists():
        if checksum(destination) != expected:
            raise RuntimeError(f"现有实验文件与共享视觉观测不一致：{destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def seal_observations(shared: Path, manifest):
    hashes = {ep: checksum(shared / "observations" / f"{ep}.json")
              for ep in manifest["episodes"]}
    path = shared / "observations_manifest.json"
    if path.exists() and load(path) != hashes:
        raise RuntimeError("共享视觉观测内容已变化；请使用新的 --shared-perception-dir")
    if not path.exists():
        atomic_json(path, hashes)
    return hashes


def seed_from_comparison(source: Path, shared: Path, data: Path, manifest, tasks, settings):
    """Reuse a complete prior perception only when its actual input contract matches."""
    config_path = source / "config.json"
    if not config_path.exists():
        return False
    prior = load(config_path)
    if prior.get("dataset") != manifest or prior.get("selected_tasks_sha256") != fingerprint(tasks):
        return False
    old_caption, current_caption = prior.get("caption", {}), caption_config(settings, data)
    # Older runs hashed the entire memory module as their validator. The schema,
    # prompt, extraction settings and every imported frame are checked directly.
    if any(old_caption.get(key) != value for key, value in current_caption.items()
           if key != "validator_sha256"):
        return False
    if (prior.get("llm_url") != settings.llm_url or
            prior.get("vision_model") != settings.vision_model or
            prior.get("llm_model") != settings.llm_model):
        return False
    budget = source / "perception" / "api_budget.json"
    if not budget.exists():
        return False
    for ep in manifest["episodes"]:
        path = source / "observations" / f"{ep}.json"
        if not path.exists():
            return False
        saved = load(path)
        if saved.get("episode") != ep or saved.get("schema_version") != SCHEMA_VERSION:
            return False
        validate_batch(saved["observations"], range(96))
    for ep in manifest["episodes"]:
        copy_verified(source / "observations" / f"{ep}.json",
                      shared / "observations" / f"{ep}.json")
    copy_verified(budget, shared / "perception" / "api_budget.json")
    atomic_json(shared / "import_origin.json", {
        "source_directory": str(source), "source_config_fingerprint": prior.get("fingerprint"),
        "source_observation_sha256": {ep: checksum(source / "observations" / f"{ep}.json")
                                      for ep in manifest["episodes"]}})
    return True


def main(method: str, argv=None):
    if method not in METHODS:
        raise ValueError(f"未知方法：{method}")
    parser = argparse.ArgumentParser(description=f"完整运行 {method} 的 FindingDory 记忆实验")
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runtime/experiments") / f"{method}_single")
    parser.add_argument("--shared-perception-dir", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--seed-perception-from", type=Path, default=DEFAULT_SEED,
                        help="已完成的兼容实验目录；不存在时重新提取视觉观测")
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    run, shared, data = (p.resolve() for p in
                         (args.run_dir, args.shared_perception_dir, args.data_dir))
    seed = args.seed_perception_from.resolve()
    if (run == shared or run in shared.parents or shared in run.parents or
            run == data or data in run.parents or shared == data or data in shared.parents):
        parser.error("实验目录、共享视觉目录和数据目录必须相互独立")

    settings = Settings.load()
    require_live_settings(settings)
    audit = environment_check((method,))
    manifest, tasks = verify_data(data)
    freeze(shared / "config.json", shared_config(settings, data, manifest, tasks, shared))
    print(f"[预检] {method}：2 段视频、12 题；共享视觉目录 {shared}", flush=True)
    ensure_frames(data, manifest)
    if complete_observations(shared, manifest):
        print("[视觉观测] 校验通过，复用共享的 192 帧；本次不重复请求视觉模型", flush=True)
    else:
        started = any((shared / "perception" / "batches").rglob("*.json"))
        imported = not started and seed_from_comparison(seed, shared, data, manifest, tasks, settings)
        if imported:
            print(f"[视觉观测] 从已完成实验 {seed} 导入并校验 192 帧；本次不请求视觉模型", flush=True)
        else:
            print("[视觉观测] 构建共享观测；后续单方法脚本可直接复用", flush=True)
            observe(shared, data, manifest, settings)
        if not complete_observations(shared, manifest):
            raise RuntimeError("共享视觉观测未完整保存")
        cleanup_success(shared, ())
    observation_hashes = seal_observations(shared, manifest)

    frozen = frozen_config(settings, data, manifest, tasks, (method,), audit, ())
    frozen["single_runner_sha256"] = checksum(Path(__file__))
    frozen["shared_perception"] = {
        "directory": str(shared), "config_fingerprint": load(shared / "config.json")["fingerprint"],
        "observation_sha256": observation_hashes}
    frozen["credential_fingerprint"] = credential_fingerprint(run, settings, embedding=True)
    freeze(run / "config.json", frozen)
    for ep in manifest["episodes"]:
        copy_verified(shared / "observations" / f"{ep}.json", run / "observations" / f"{ep}.json")
    copy_verified(shared / "perception" / "api_budget.json", run / "perception" / "api_budget.json")
    atomic_json(run / "perception_origin.json", {
        "shared_directory": str(shared), **frozen["shared_perception"]})
    execute_method(run, data, method, ())
    summarize(run, tasks, (method,))
    cleanup_success(run, (method,))


if __name__ == "__main__":
    raise SystemExit("请运行 benchmarks.run_hypergraph / run_mem0 / run_amem / run_readagent")
