"""Build one frozen visual-observation set for the first 50 FindingDory validation videos.

Run from the repository root:
  python -m benchmarks.prepare_validation50_observations

This stage never sends questions or answers to the vision model. It makes real API
requests only when run without --check-only. All memory methods should consume
the same observations_manifest.json and observations/ directory.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmarks.comparison_worker import CachedClients, atomic_json
from benchmarks.findingdory_memory import SCHEMA_VERSION, validate_batch
from benchmarks.findingdory_pilot import load
from benchmarks.findingdory_structured import caption_config, extract_batch, freeze, strict_schema
from benchmarks.run_budget import RunBudget
from memory_agent.settings import Settings


DEFAULT_DATA = REPO / "runtime" / "findingdory"
DEFAULT_SHARED = REPO / "runtime" / "experiments" / "findingdory_validation50_perception"
EXTRACTION = {"scale": "640:-2", "jpeg_quality": 3, "start_number": 0,
              "frames_per_video": 96}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_data(data: Path) -> dict:
    manifest_path = data / "validation_first50_manifest.json"
    questions_path = data / "validation_first50_questions.jsonl"
    manifest = load(manifest_path)
    videos = list(manifest["videos"])
    expected = [f"videos/val/ep_{index}.mp4" for index in range(1, 51)]
    if (manifest.get("dataset") != "yali30/findingdory-subsampled-96" or
            manifest.get("split") != "validation" or videos != expected or
            manifest.get("selected_questions") != 2901):
        raise RuntimeError("数据清单不是固定的验证集前 50 个视频 / 2,901 道题")
    counts = {video: 0 for video in videos}
    with questions_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            row = json.loads(line)
            if row.get("row_idx") != line_number or row.get("video") not in counts:
                raise RuntimeError("题目清单的行号或视频归属不符合官方顺序")
            counts[row["video"]] += 1
    if sum(counts.values()) != 2901 or any(
            counts[video] != manifest["videos"][video]["questions"] for video in videos):
        raise RuntimeError("题目数与视频清单不一致")
    for video in videos:
        path = data / video
        item = manifest["videos"][video]
        if (not path.is_file() or path.stat().st_size != item["bytes"] or
                checksum(path) != item["sha256"]):
            raise RuntimeError(f"视频不存在或 SHA-256 校验失败：{path}")
    return {**manifest, "episodes": [f"ep_{index}" for index in range(1, 51)]}


def executable_version(ffmpeg: str) -> str:
    result = subprocess.run([ffmpeg, "-version"], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.splitlines()[0]


def credential_fingerprint(shared: Path, key: str) -> str:
    salt_path = shared / "credential_salt.bin"
    if not salt_path.exists():
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        salt_path.write_bytes(os.urandom(32))
    salt = salt_path.read_bytes()
    if len(salt) != 32:
        raise RuntimeError("凭据指纹盐文件长度异常")
    return hmac.new(salt, key.encode(), hashlib.sha256).hexdigest()


def frozen_config(settings, shared: Path, manifest: dict, ffmpeg_version: str) -> dict:
    sources = [REPO / "benchmarks" / name for name in (
        "prepare_validation50_observations.py", "run_budget.py")]
    sources.append(REPO / "src" / "memory_agent" / "clients.py")
    # Memory construction and answering may evolve without invalidating paid-for
    # visual observations. Freeze only the captioning functions and schema contract.
    extractor_source = inspect.getsource(extract_batch) + inspect.getsource(strict_schema)
    return {"protocol": "findingdory-validation50-shared-perception-v1",
            "dataset": manifest, "caption": caption_config(settings, shared),
            "extractor_sha256": hashlib.sha256(extractor_source.encode()).hexdigest(),
            "response_cache_sha256": hashlib.sha256(
                inspect.getsource(CachedClients).encode()).hexdigest(),
            "frame_extraction": EXTRACTION | {"ffmpeg_version": ffmpeg_version},
            "vision_url": settings.llm_url,
            "vision_model": settings.vision_model or settings.llm_model,
            "llm_thinking": settings.llm_thinking,
            "code_sha256": {str(path.relative_to(REPO)): checksum(path) for path in sources},
            "credential_fingerprint": credential_fingerprint(shared, settings.llm_key),
            "observation_role": "图像的模型解释；不输入验证集问题与答案，也非官方真值"}


def ensure_frames(data: Path, shared: Path, ep: str, video: str, video_sha: str,
                  ffmpeg: str, ffmpeg_version: str) -> str:
    directory = shared / "frames" / ep
    record = directory / "frames_manifest.json"
    expected = {f"{index:03d}.jpg" for index in range(96)}
    contract = {"source_video": video, "video_sha256": video_sha,
                "extraction": EXTRACTION, "ffmpeg_version": ffmpeg_version}
    if record.exists():
        saved = load(record)
        if any(saved.get(key) != value for key, value in contract.items()):
            raise RuntimeError(f"抽帧条件已变化：{ep}；请使用新的 --shared-dir")
        actual = {path.name for path in directory.glob("*.jpg")}
        if actual != expected or any(
                checksum(directory / name) != sha for name, sha in saved["frame_sha256"].items()):
            raise RuntimeError(f"已保存的原始帧发生变化：{ep}")
        return checksum(record)
    if any((shared / "perception" / "batches" / ep).glob("*.json")):
        raise RuntimeError(f"{ep} 有视觉批次，但缺少抽帧清单；拒绝混用来源帧")
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(data / video), "-vf", "scale=640:-2", "-q:v", "3",
        "-frames:v", "96", "-start_number", "0", str(directory / "%03d.jpg")],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"{ep} 抽帧失败：{result.stderr[-500:]}")
    actual = {path.name for path in directory.glob("*.jpg")}
    if actual != expected or any(not (directory / name).stat().st_size for name in expected):
        raise RuntimeError(f"{ep} 应有 96 张非空原始帧，实际得到 {len(actual)} 张")
    atomic_json(record, contract | {"frame_sha256": {
        name: checksum(directory / name) for name in sorted(expected)}})
    return checksum(record)


def checked_observation(path: Path, ep: str, config_sha: str, frame_sha: str) -> list:
    saved = load(path)
    if (saved.get("episode") != ep or saved.get("schema_version") != SCHEMA_VERSION or
            saved.get("configuration_fingerprint") != config_sha or
            saved.get("frame_manifest_sha256") != frame_sha):
        raise RuntimeError(f"已保存观测与当前实验配置不一致：{path}")
    return validate_batch(saved["observations"], list(range(96)))


def seal(shared: Path, manifest: dict, config_sha: str) -> dict:
    hashes = {ep: checksum(shared / "observations" / f"{ep}.json")
              for ep in manifest["episodes"]}
    result = {"protocol": "findingdory-validation50-shared-perception-v1",
              "configuration_fingerprint": config_sha,
              "dataset_revision": manifest["revision"], "episodes": manifest["episodes"],
              "observation_sha256": hashes}
    path = shared / "observations_manifest.json"
    if path.exists() and load(path) != result:
        raise RuntimeError("共享观测内容被改动；请使用新的 --shared-dir")
    if not path.exists():
        atomic_json(path, result)
    return result


def run(data: Path, shared: Path, settings, *, max_requests=10000,
        max_reported_tokens=0, extractor=extract_batch, client_factory=CachedClients):
    if data == shared or shared in data.parents or data in shared.parents:
        raise ValueError("数据目录与共享观测目录必须相互独立")
    manifest = selected_data(data)
    key = settings.llm_key
    model = settings.vision_model or settings.llm_model
    if not key or key.startswith(("YOUR_", "填写")) or not model:
        raise RuntimeError("请先配置 MEMORY_LLM_API_KEY 和 MEMORY_VISION_MODEL 或 MEMORY_LLM_MODEL")
    if not settings.llm_url.startswith(("https://", "http://")):
        raise RuntimeError("MEMORY_LLM_BASE_URL 无效")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("缺少 ffmpeg，无法从视频抽帧")
    version = executable_version(ffmpeg)
    # The caption contract hashes this manifest, never the legacy 2-video pilot.
    freeze(shared / "manifest.json", manifest)
    config = freeze(shared / "config.json", frozen_config(settings, shared, manifest, version))
    if (shared / "observations_manifest.json").exists():
        for ep in manifest["episodes"]:
            video = f"videos/val/{ep}.mp4"
            frame_sha = ensure_frames(data, shared, ep, video,
                manifest["videos"][video]["sha256"], ffmpeg, version)
            checked_observation(shared / "observations" / f"{ep}.json",
                                ep, config["fingerprint"], frame_sha)
        seal(shared, manifest, config["fingerprint"])
        print("[完成] 50 段共享观测已封存；没有重复请求模型", flush=True)
        return
    budget = RunBudget(shared / "perception", max_requests=max_requests,
                       max_reported_tokens=max_reported_tokens)
    client = client_factory(settings.model_copy(update={"timeout": 60, "retries": 0}),
                            budget, shared / "perception" / "response_cache")
    try:
        for episode_number, ep in enumerate(manifest["episodes"], 1):
            video = f"videos/val/{ep}.mp4"
            frame_sha = ensure_frames(data, shared, ep, video,
                    manifest["videos"][video]["sha256"], ffmpeg, version)
            target = shared / "observations" / f"{ep}.json"
            if target.exists():
                checked_observation(target, ep, config["fingerprint"], frame_sha)
                print(f"[视觉观测] {episode_number}/50 {ep} 已完成，直接复用", flush=True)
                continue
            accumulated = []
            for start in range(0, 96, 2):
                indices = [start, start + 1]
                checkpoint = shared / "perception" / "batches" / ep / f"{start:03d}.json"
                if checkpoint.exists():
                    batch = validate_batch(load(checkpoint), indices, accumulated)
                else:
                    batch = extractor(client, shared, ep, indices, accumulated,
                        shared / "perception" / "validation_checks" / ep / f"{start:03d}.json",
                        max_attempts=2, max_output_tokens=3600)
                    atomic_json(checkpoint, [frame.model_dump(mode="json") for frame in batch])
                accumulated.extend(batch)
                print(f"[视觉观测] {episode_number}/50 {ep} {len(accumulated)}/96 帧"
                      f" | 总帧 {((episode_number - 1) * 96 + len(accumulated))}/4800"
                      f" | 请求 {budget.state['requests_started']}"
                      f" | token {budget.state['reported_tokens']}", flush=True)
            atomic_json(target, {"episode": ep, "schema_version": SCHEMA_VERSION,
                "configuration_fingerprint": config["fingerprint"],
                "frame_manifest_sha256": frame_sha,
                "observations": [frame.model_dump(mode="json") for frame in accumulated],
                "provenance": "模型对原始视频帧的解释；未输入问题和答案；非官方真值"})
    finally:
        client.close()
    seal(shared, manifest, config["fingerprint"])
    print(f"[完成] 50 段、4800 帧；共享目录 {shared}", flush=True)
    print(f"[用量] 请求 {budget.state['requests_started']}，报告 token "
          f"{budget.state['reported_tokens']}；费用以模型平台账单为准", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--max-requests", type=int, default=10000)
    parser.add_argument("--max-reported-tokens", type=int, default=0,
                        help="0 表示不设累计 token 上限")
    parser.add_argument("--check-only", action="store_true", help="仅校验本地数据，不调用模型")
    args = parser.parse_args(argv)
    if args.max_requests < 1 or args.max_reported_tokens < 0:
        parser.error("请求上限需大于 0，token 上限不能为负")
    data, shared = args.data_dir.resolve(), args.shared_dir.resolve()
    if args.check_only:
        manifest = selected_data(data)
        print(f"[预检] {len(manifest['episodes'])} 个视频、{manifest['selected_questions']} 道题；"
              "4800 帧，通常至少约 2400 次视觉模型请求；本次未调用模型", flush=True)
        return
    print("[提示] 本命令会调用真实视觉模型；50 个视频共 4800 帧，正常批次至少 2400 次请求", flush=True)
    run(data, shared, Settings.load(REPO / ".env"),
        max_requests=args.max_requests, max_reported_tokens=args.max_reported_tokens)


if __name__ == "__main__":
    main()
