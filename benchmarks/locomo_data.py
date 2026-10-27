"""Pinned official LoCoMo inputs and the official category-aware F1 evaluator.

Only scorer function definitions are loaded from the verified upstream file:
this avoids importing BERTScore/transformer models for a token-F1 experiment.
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import string
import sys
import time
from uuid import uuid4

import httpx

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "runtime/locomo"
REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
UPSTREAM = "https://github.com/snap-research/locomo"
FILES = {
    "locomo10.json": ("data/locomo10.json", "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"),
    "official/evaluation.py": ("task_eval/evaluation.py", "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"),
    "official/LICENSE.txt": ("LICENSE.txt", "41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0"),
    "official/README.MD": ("README.MD", "9f8e6fd00a3400aa687109f40ed53715f0a2c028ee3f8c465bdfa96475640e8a"),
}
CATEGORIES = {1: "多跳", 2: "时间推理", 3: "开放域", 4: "单跳", 5: "不可回答"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    """Unique temporary file; retry Windows rename only, never repeat an API call."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        for attempt in range(30):
            try:
                temp.replace(path)
                return
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(.1)
    finally:
        temp.unlink(missing_ok=True)


def prepare(data=DEFAULT_DATA, *, download=False):
    data = Path(data)
    for local, (remote, expected) in FILES.items():
        path = data / local
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"缺少 LoCoMo 文件：{path}；先运行 --prepare-only")
            url = f"https://raw.githubusercontent.com/snap-research/locomo/{REVISION}/{remote}"
            print(f"[数据下载] 官方固定版本 {remote}", flush=True)
            # A failed local proxy is allowed to fall back to a direct public download.
            # This fallback is ONLY for these fixed public URLs, never model traffic.
            payload = None
            for trust_env in (True, False):
                try:
                    response = httpx.get(url, timeout=60, follow_redirects=True, trust_env=trust_env)
                    response.raise_for_status()
                    payload = response.content
                    break
                except httpx.HTTPError:
                    continue
            if payload is None:
                raise RuntimeError("LoCoMo 官方下载失败；检查网络后用同一命令重试")
            if hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError(f"下载校验失败：{remote}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".download")
            try:
                temp.write_bytes(payload)
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        if checksum(path) != expected:
            raise RuntimeError(f"官方 LoCoMo 文件哈希不一致：{path}；请保留原文件并检查来源")
    samples = load(data / "locomo10.json")
    counts = Counter(q["category"] for sample in samples for q in sample["qa"])
    if len(samples) != 10 or counts != {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}:
        raise RuntimeError("LoCoMo 会话数量或类别题量与固定版本不一致")
    return samples


def session_time(text):
    """Dataset-local naive time; UTC is a storage anchor, not an asserted timezone."""
    return datetime.strptime(text, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)


def conversation_view(sample):
    """Whitelisted input: no QA, gold evidence, event_summary or generated summaries."""
    conversation = sample["conversation"]
    speakers = [conversation["speaker_a"], conversation["speaker_b"]]
    turns = []
    sessions = sorted((key for key in conversation if re.fullmatch(r"session_\d+", key)),
                      key=lambda key: int(key.split("_")[1]))
    seen = set()
    for session in sessions:
        date = conversation[session + "_date_time"]
        anchor = session_time(date)
        for n, turn in enumerate(conversation[session]):
            identity = turn["dia_id"]
            if identity in seen or turn["speaker"] not in speakers or not isinstance(turn["text"], str):
                raise ValueError("Invalid or duplicate LoCoMo dialog identity")
            seen.add(identity)
            # Turn offsets encode dialogue order only, not measured speaking times.
            turns.append({"dia_id": identity, "speaker": turn["speaker"], "text": turn["text"],
                          "session": session, "session_date": date,
                          "observed_at": (anchor + timedelta(microseconds=n)).isoformat(),
                          "image_caption": str(turn.get("blip_caption") or "")})
    if not turns:
        raise ValueError("Conversation has no turns")
    return {"sample_id": sample["sample_id"], "speakers": speakers, "turns": turns}


def select_tasks(samples, categories=(1, 2, 3, 4), max_conversations=10, max_questions=None):
    tasks, views = [], {}
    for sample in samples[:max_conversations]:
        view = conversation_view(sample)
        selected = []
        for index, qa in enumerate(sample["qa"]):
            if qa["category"] not in categories:
                continue
            selected.append({"task_id": f"{sample['sample_id']}__q{index:04d}",
                             "sample_id": sample["sample_id"], "question_index": index,
                             "question": qa["question"], "category": qa["category"],
                             "answer": qa.get("answer", ""), "evidence": qa.get("evidence", [])})
        if max_questions is not None:
            selected = selected[:max(0, max_questions - len(tasks))]
        if selected:
            views[view["sample_id"]] = view
            tasks.extend(selected)
        if max_questions is not None and len(tasks) >= max_questions:
            break
    if not tasks:
        raise ValueError("No selected questions")
    return views, tasks


class OfficialScorer:
    def __init__(self, data):
        path = Path(data) / "official/evaluation.py"
        if checksum(path) != FILES["official/evaluation.py"][1]:
            raise RuntimeError("Official evaluator hash mismatch")
        local_deps = REPO / "runtime/locomo-deps"
        if local_deps.is_dir():
            sys.path.insert(0, str(local_deps))
        try:
            from nltk.stem import PorterStemmer
            import regex as upstream_regex
        except ImportError:
            raise RuntimeError("缺少 LoCoMo 评分依赖；运行 python -m pip install -r requirements-locomo.txt") from None
        selected = {"normalize_answer", "f1_score", "f1"}
        tree = ast.parse(path.read_text(encoding="utf-8"))
        body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        if {node.name for node in body} != selected:
            raise RuntimeError("Official evaluator function inventory changed")
        class Mean:
            @staticmethod
            def mean(values):
                return sum(values) / len(values)
        namespace = {"regex": upstream_regex, "string": string, "Counter": Counter,
                     "np": Mean, "ps": PorterStemmer()}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
        self.f1_score, self.multi_f1 = namespace["f1_score"], namespace["f1"]

    def score(self, prediction, task):
        # Match upstream eval_question_answering category handling exactly.
        category, gold = task["category"], str(task.get("answer", ""))
        if category == 3:
            gold = gold.split(";")[0].strip()
        if category in (2, 3, 4):
            return float(self.f1_score(prediction, gold))
        if category == 1:
            return float(self.multi_f1(prediction, gold))
        if category == 5:
            return float("no information available" in prediction.lower() or "not mentioned" in prediction.lower())
        raise ValueError("Unknown question category")

    @staticmethod
    def evidence_recall(task, context_ids):
        gold = set(task.get("evidence") or [])
        return len(gold & set(context_ids)) / len(gold) if gold else None
