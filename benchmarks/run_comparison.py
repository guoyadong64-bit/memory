"""One-command, resumable four-method FindingDory development comparison."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

# The documented one-line command does not require setting PYTHONPATH by hand.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmarks.comparison_methods import REGISTRY
from benchmarks.comparison_worker import CachedClients, atomic_json
from benchmarks.findingdory_memory import SCHEMA_VERSION, validate_batch
from benchmarks.findingdory_pilot import DATASET, ROOT, load, prepare
from benchmarks.findingdory_structured import caption_config, extract_batch, fingerprint, freeze
from benchmarks.run_budget import RunBudget
from memory_agent.settings import Settings

DEFAULT_METHODS = ("hypergraph", "mem0", "amem", "readagent")
PYTHONS = {"core": REPO / ".." / ".." / "conda" / "envs" / "smp-memory" / "python.exe",
           "mem0": REPO / "runtime" / "mem0-env" / "Scripts" / "python.exe",
           "amem": REPO / "runtime" / "amem-env" / "Scripts" / "python.exe"}


def checksum(path: Path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def require_live_settings(settings):
    required = {"MEMORY_LLM_API_KEY": settings.llm_key,
                "MEMORY_LLM_MODEL": settings.llm_model,
                "MEMORY_EMBEDDING_API_KEY": settings.embedding_key or settings.llm_key,
                "MEMORY_EMBEDDING_MODEL": settings.embedding_model}
    missing = [key for key, val in required.items() if not val or str(val).startswith(("YOUR_", "填写"))]
    if missing:
        raise RuntimeError("请先填写 .env 中的配置：" + ", ".join(missing))
    if not settings.llm_url.startswith(("https://", "http://")):
        raise RuntimeError("Invalid MEMORY_LLM_BASE_URL")


def verify_data(data: Path, *, permit_download=True):
    if not (data / "manifest.json").exists():
        if data.resolve() != (REPO / ROOT).resolve() or not permit_download:
            raise FileNotFoundError("数据缺失；只能在默认数据目录自动下载")
        print("[数据] 准备 FindingDory 的两段视频和 12 道开发题", flush=True)
        prepare()
    manifest = load(data / "manifest.json")
    tasks = load(data / "selected_tasks.json")
    if (manifest.get("dataset") != DATASET or len(tasks) != 12 or
            manifest.get("episodes") != ["ep_1", "ep_33"] or
            len({row["row_idx"] for row in tasks}) != 12):
        raise RuntimeError("数据集或题目清单不符合固定的两段视频 / 12 题协议")
    for video, expected in manifest["files"].items():
        path = data / video
        if not path.exists() or path.stat().st_size != expected["bytes"] or checksum(path) != expected["sha256"]:
            raise RuntimeError(f"视频校验失败：{video}")
    return manifest, tasks


def environment_check(methods, plugins=()):
    audit = {}
    if "mem0" in methods:
        from benchmarks.mem0_adapter import MEM0_VERSION
        python = PYTHONS["mem0"]
        if not python.exists():
            raise RuntimeError(f"Mem0 隔离环境不存在：{python}")
        result = subprocess.check_output([str(python), "-c", "from importlib.metadata import version; print(version('mem0ai'))"], text=True).strip()
        if result != MEM0_VERSION:
            raise RuntimeError(f"Mem0 版本应为 {MEM0_VERSION}，实际 {result}")
        subprocess.run([str(python), "-c",
                        "from importlib.metadata import version; print(version('qdrant-client'))"],
                       check=True, stdout=subprocess.DEVNULL)
        audit["mem0"] = {"sdk_version": result}
    if "amem" in methods:
        from benchmarks.amem_adapter import REVISION as amem_revision, ROOT as amem_root, source_audit
        python = PYTHONS["amem"]
        if not python.exists():
            raise RuntimeError(f"A-Mem 隔离环境不存在：{python}")
        if not amem_root.exists():
            raise RuntimeError("A-Mem 官方源码不存在")
        revision = subprocess.check_output(["git", "-C", str(amem_root), "rev-parse", "HEAD"], text=True).strip()
        if revision != amem_revision:
            raise RuntimeError("A-Mem 官方源码版本改变")
        subprocess.run([str(python), "-c",
                        "from importlib.metadata import version; print(version('chromadb'), version('jsonschema'))"],
                       check=True, stdout=subprocess.DEVNULL)
        audit["amem"] = source_audit()
    if "readagent" in methods:
        from benchmarks.readagent_official_adapter import official_source
        _, audit["readagent"] = official_source()
    for name in methods:
        cls = REGISTRY[name]
        python = PYTHONS.get(cls.environment, Path(cls.environment))
        if cls.environment == "core":
            python = Path(sys.executable)
        if not python.exists():
            raise RuntimeError(f"{name} 运行环境不存在：{python}")
    return audit


def ensure_frames(data, manifest):
    for ep in manifest["episodes"]:
        directory = data / "frames" / ep
        frames = [directory / f"{n:03d}.jpg" for n in range(96)]
        if all(path.exists() and path.stat().st_size > 0 for path in frames):
            print(f"[抽帧] {ep}: 96/96（复用已校验视频的本地帧）", flush=True)
            continue
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("缺少 ffmpeg，无法从视频抽帧")
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i",
                        str(data / "videos" / "val" / f"{ep}.mp4"), "-vf", "scale=640:-2",
                        "-q:v", "3", "-start_number", "0", str(directory / "%03d.jpg")], check=True)
        if not all(path.exists() and path.stat().st_size > 0 for path in frames):
            raise RuntimeError(f"{ep} 抽帧后不足 96 帧")
        print(f"[抽帧] {ep}: 96/96", flush=True)


def frozen_config(settings, data, manifest, tasks, methods, audit, plugins):
    paths = [REPO / "benchmarks" / filename for filename in (
        "run_comparison.py", "comparison_worker.py", "comparison_methods.py",
        "findingdory_visual_answer.py", "findingdory_memory.py", "findingdory_structured.py",
        "findingdory_pilot.py", "answer_normalization.py")]
    adapters = {"mem0": "mem0_adapter.py", "amem": "amem_adapter.py",
                "readagent": "readagent_official_adapter.py"}
    paths.extend(REPO / "benchmarks" / adapters[name] for name in methods if name in adapters)
    paths += [REPO / "src" / "memory_agent" / filename for filename in (
        "memory.py", "recall.py", "retrieval.py", "semantic.py")]
    plugin_paths = []
    for name in plugins:
        module = importlib.import_module(name)
        if not module.__file__:
            raise ValueError(f"插件无源文件：{name}")
        plugin_paths.append(Path(module.__file__))
    return {"protocol": "four-memory-common-visual-v1", "dataset": manifest,
            "selected_tasks_sha256": fingerprint(tasks), "methods": list(methods),
            "plugins": list(plugins), "caption": caption_config(settings, data),
            "llm_url": settings.llm_url, "llm_model": settings.llm_model,
            "vision_model": settings.vision_model, "answer_model": settings.answer_model,
            "embedding_url": settings.embedding_url or settings.llm_url,
            "embedding_model": settings.embedding_model, "llm_thinking": settings.llm_thinking,
            "semantic_min_similarity": settings.semantic_min_similarity,
            "code_sha256": {str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path): checksum(path)
                            for path in paths + plugin_paths}, "official_sources": audit,
            "answer_images_max": 12, "token_limit": None,
            "data_role": "开发题：已参与方法调试，不能视为论文未见测试结果"}


def observe(run, data, manifest, settings):
    root = run / "perception"
    budget = RunBudget(root, max_requests=5000, max_reported_tokens=0)
    client = CachedClients(settings.model_copy(update={"timeout": 60, "retries": 0}), budget,
                           root / "response_cache")
    try:
        for ep in manifest["episodes"]:
            accumulated = []
            for start in range(0, 96, 2):
                indices = [start, start + 1]
                path = root / "batches" / ep / f"{start:03d}.json"
                if path.exists():
                    batch = validate_batch(load(path), indices, accumulated)
                else:
                    batch = extract_batch(client, data, ep, indices, accumulated,
                                          root / "validation_checks" / ep / f"{start:03d}.json",
                                          max_attempts=2, max_output_tokens=3600)
                    atomic_json(path, [frame.model_dump(mode="json") for frame in batch])
                accumulated.extend(batch)
                print(f"[视觉观测] {ep} {len(accumulated)}/96 帧 | 请求 {budget.state['requests_started']} | "
                      f"token {budget.state['reported_tokens']}", flush=True)
            atomic_json(run / "observations" / f"{ep}.json", {
                "episode": ep, "schema_version": SCHEMA_VERSION,
                "observations": [frame.model_dump(mode="json") for frame in accumulated],
                "provenance": "模型解释的图像观测，非官方真值"})
    finally:
        client.close()


def python_for(name):
    env = REGISTRY[name].environment
    return str(Path(sys.executable) if env == "core" else PYTHONS.get(env, Path(env)))


def execute_method(run, data, name, plugins):
    cmd = [python_for(name), "-m", "benchmarks.comparison_worker", "--method", name,
           "--run-dir", str(run), "--data-dir", str(data)]
    for plugin in plugins:
        cmd.extend(["--plugin", plugin])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(REPO), str(REPO / "src"), env.get("PYTHONPATH", "")))
    env["PYTHONIOENCODING"] = "utf-8"
    # Only this worker writes its budget. Read completed stages once before it
    # starts, then use the worker's own progress event for its running totals.
    # Reopening api_budget.json while Windows replaces it can raise WinError 5.
    fixed_budgets = [load(path) for path in (run / "perception").glob("api_budget.json")]
    fixed_budgets += [load(path) for path in (run / "methods").glob("*/api_budget.json")
                      if path.parent.name != name]
    fixed_requests = sum(item["requests_started"] for item in fixed_budgets)
    fixed_tokens = sum(item["reported_tokens"] for item in fixed_budgets)
    print(f"[记忆] 开始 {name}，环境 {REGISTRY[name].environment}", flush=True)
    with subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as process:
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith("@@PROGRESS@@"):
                event = json.loads(line[len("@@PROGRESS@@"):])
                with (run / "progress.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(event, ensure_ascii=False) + "\n")
                all_requests = fixed_requests + event["requests"]
                all_tokens = fixed_tokens + event["tokens"]
                print(f"[{name}·{event['stage']}] {event.get('episode','')} "
                      f"{event['completed']}/{event['total']} | 正确 {event.get('correct','—')} | "
                      f"本方法请求 {event['requests']} | 总请求 {all_requests} | "
                      f"累计 token {all_tokens}", flush=True)
            else:
                print(line, flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"{name} 失败，断点保存在 {run / 'methods' / name}；同一命令可续跑")


def summarize(run, tasks, methods):
    expected = {row["row_idx"] for row in tasks}
    rows, summaries, usage = [], [], {}
    for name in methods:
        directory = run / "methods" / name
        found = {int(path.stem): load(path) for path in (directory / "answers").glob("*.json")}
        if set(found) != expected:
            raise RuntimeError(f"{name} 题目数量不完整或重复")
        method_rows = [found[task["row_idx"]] for task in tasks]
        rows.extend(method_rows)
        state = load(directory / "api_budget.json")
        usage[name] = state
        summaries.append({"method": name, "correct": sum(r["correct"] for r in method_rows),
                          "total": len(tasks),
                          "accuracy": round(sum(r["correct"] for r in method_rows) / len(tasks), 4),
                          "candidate_frame_coverage": round(sum(r["candidate_frame_covered"] for r in method_rows) / len(tasks), 4),
                          "requests": state["requests_started"], "reported_tokens": state["reported_tokens"]})
    usage["perception"] = load(run / "perception" / "api_budget.json")
    atomic_json(run / "summary.json", summaries)
    atomic_json(run / "api_usage.json", usage)
    (run / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n"
                                                   for row in rows), encoding="utf-8")
    table = ["| 方法 | 答对 | 正确率 | 候选帧覆盖率 | 请求数 | 报告 token |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in summaries:
        table.append(f"| {item['method']} | {item['correct']}/{item['total']} | "
                     f"{item['accuracy']:.1%} | {item['candidate_frame_coverage']:.1%} | "
                     f"{item['requests']} | {item['reported_tokens']} |")
    title = "四种记忆方法对比" if len(methods) > 1 else f"{methods[0]} 单方法实验"
    report = f"# {title}\n\n" + "\n".join(table) + "\n\n"
    shared = usage["perception"]
    total_requests = shared["requests_started"] + sum(item["requests"] for item in summaries)
    total_tokens = shared["reported_tokens"] + sum(item["reported_tokens"] for item in summaries)
    report += (f"共 {len(tasks)} 道 FindingDory 开发题；同一视觉观测和同一视觉答题器，"
               "每题最多 12 张原图。正确率是目标帧严格匹配，候选帧覆盖率是每个目标的候选证据是否包含至少一个真值帧，再对题目求平均，"
               "二者不能互换。"
               f"共享视觉观测请求 {shared['requests_started']} 次、报告 token {shared['reported_tokens']}；"
               f"全流程请求 {total_requests} 次、报告 token {total_tokens}。费用以平台账单为准。\n\n"
               "**这组题已参与开发和调试，不能作为论文最终未见测试集结果。**\n")
    origin = run / "perception_origin.json"
    if origin.exists():
        source = load(origin)
        report += (f"\n视觉观测来自共享目录 `{source['shared_directory']}`，其 API 用量在各单方法报告中"
                   "作为共同前处理列示；汇总多个方法的费用时只计算这份视觉观测一次。\n")
    (run / "REPORT.md").write_text(report, encoding="utf-8")
    print("\n".join(table), flush=True)
    print(f"[完成] 报告：{run / 'REPORT.md'}", flush=True)
    return summaries


def cleanup_success(run, methods):
    """Remove only transient diagnostics after every method has completed."""
    diagnostics = (run / "perception" / "validation_checks").resolve()
    if diagnostics.exists():
        if run.resolve() not in diagnostics.parents:
            raise RuntimeError("临时目录越过实验目录，拒绝清理")
        shutil.rmtree(diagnostics)
    for name in methods:
        (run / "methods" / name / "last_failure.json").unlink(missing_ok=True)
    for path in run.rglob("*.tmp"):
        if run.resolve() not in path.resolve().parents:
            raise RuntimeError("临时文件越过实验目录，拒绝清理")
        path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runtime/experiments/compare_four_methods"))
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--plugin", action="append", default=[], help="importable module registering a MemoryMethod")
    args = parser.parse_args(argv)
    for plugin in args.plugin:
        importlib.import_module(plugin)
    methods = tuple(args.methods)
    if len(set(methods)) != len(methods) or any(name not in REGISTRY for name in methods):
        parser.error("方法重复或未注册")
    run, data = args.run_dir.resolve(), args.data_dir.resolve()
    if run == data or data in run.parents:
        parser.error("实验目录不能位于数据目录内")
    settings = Settings.load()
    require_live_settings(settings)
    audit = environment_check(methods, args.plugin)
    manifest, tasks = verify_data(data)
    frozen = frozen_config(settings, data, manifest, tasks, methods, audit, args.plugin)
    salt_path = run / "credential_salt.bin"
    if not salt_path.exists():
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        salt_path.write_bytes(os.urandom(32))
    salt = salt_path.read_bytes()
    if len(salt) != 32:
        raise RuntimeError("凭据指纹盐文件无效；不能安全续跑")
    frozen["credential_fingerprint"] = {
        name: hmac.new(salt, value.encode(), hashlib.sha256).hexdigest()
        for name, value in {"llm": settings.llm_key,
                            "embedding": settings.embedding_key or settings.llm_key}.items()}
    freeze(run / "config.json", frozen)
    print(f"[预检] 通过：{len(manifest['episodes'])} 段视频，{len(tasks)} 题，方法 {', '.join(methods)}", flush=True)
    ensure_frames(data, manifest)
    observe(run, data, manifest, settings)
    for name in methods:
        execute_method(run, data, name, args.plugin)
    summarize(run, tasks, methods)
    cleanup_success(run, methods)


if __name__ == "__main__":
    main()
