"""Execute the pinned ReadAgent notebook's pagination and gisting functions.

The upstream QuALITY lookup/answer loop assumes multiple-choice questions and
reads a reference label inside the loop. Its lookup prompt and page-ID parsing
are adapted here for frame-recall questions; final answers use the shared visual
answerer so the comparison measures the memory stage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1] / "runtime" / "baselines" / "read_agent_official"
NOTEBOOK = ROOT / "assets" / "read_agent_demo.ipynb"
REPOSITORY = "https://github.com/read-agent/read-agent.github.io"
REVISION = "569dff3e47bb708ad084d267ac554f2534d21756"


def official_source():
    if not NOTEBOOK.exists():
        raise FileNotFoundError(
            f"ReadAgent notebook missing: {NOTEBOOK}. Clone {REPOSITORY} into {ROOT}"
        )
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Official ReadAgent revision changed")
    notebook = NOTEBOOK.read_bytes()
    return json.loads(notebook), {"repository": REPOSITORY, "revision": REVISION,
                                  "notebook_sha256": hashlib.sha256(notebook).hexdigest()}


def _source_without_demo_call(cell, marker):
    text = "".join(cell["source"])
    return text[:text.index(marker)]


class OfficialReadAgent:
    def __init__(self, client, cache_dir):
        notebook, self.audit = official_source()
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.role = None
        self.namespace = {
            "count_words": lambda value: len(value.split()),
            # The official Gutenberg parser joins single newlines. Frame
            # observations are already paragraphs, so preserve boundaries.
            "quality_gutenberg_parser": lambda value: value,
        }
        self.namespace["query_model"] = self._query_model
        import copy
        self.namespace["copy"] = copy
        pagination = _source_without_demo_call(notebook["cells"][7], "pages = quality_pagination(example)")
        gisting = _source_without_demo_call(notebook["cells"][8], "example_with_gists = quality_gisting(example, pages)")
        lookup = "".join(notebook["cells"][9]["source"])
        lookup = lookup[:lookup.index("def quality_parallel_lookup(")]
        for source in (pagination, gisting, lookup):
            exec(compile(source, str(NOTEBOOK), "exec"), self.namespace)

    def _query_model(self, prompt):
        if self.role not in {"readagent_pagination", "readagent_gist", "readagent_lookup"}:
            raise RuntimeError("ReadAgent stage is not set")
        model = self.client.settings.llm_model
        key = hashlib.sha256(json.dumps({"role": self.role, "model": model, "prompt": prompt},
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cache = self.cache_dir / f"{key}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))["response"]
        response = self.client.post(
            self.client.settings.llm_url.rstrip("/") + "/chat/completions",
            self.client.settings.llm_key,
            {"model": model, "temperature": 0,
             "max_tokens": 768 if self.role == "readagent_gist" else 320,
             **self.client.chat_options(model),
             "messages": [{"role": "user", "content": prompt}]}, self.role)
        content = response["choices"][0]["message"]["content"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps({"response": content}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache)
        return content

    def build(self, episode, observations):
        paragraphs = []
        for frame in observations:
            item = {key: frame[key] for key in (
                "frame_index", "clock", "description", "entities", "relations", "interaction", "uncertainty")}
            paragraphs.append(f"Frame {frame['frame_index']}: " + json.dumps(item, ensure_ascii=False))
        example = {"title": episode, "article": "\n".join(paragraphs)}
        self.role = "readagent_pagination"
        # This is the official notebook function, with its published 600/280
        # word thresholds and model-selected scene breaks.
        pages = self.namespace["quality_pagination"](example, verbose=False)
        self.role = "readagent_gist"
        state = self.namespace["quality_gisting"](example, pages, verbose=False)
        self.role = None
        result = []
        for page_id, (page, gist) in enumerate(zip(state["pages"], state["shortened_pages"])):
            indices = [int(re.match(r"Frame (\d+): ", line).group(1)) for line in page]
            result.append({"page_id": page_id, "frame_indices": indices,
                           "original": "\n".join(page), "gist": gist})
        if sorted(index for page in result for index in page["frame_indices"]) != list(range(len(observations))):
            raise ValueError("ReadAgent pagination lost or duplicated a frame")
        return result

    def lookup(self, pages, question):
        numbered_gists = "\n".join(f"<Page {page['page_id']}>\n{page['gist']}" for page in pages)
        # The official prompt accepts a question and optional answer choices.
        # FindingDory questions have no choices, so the last field is empty.
        prompt = self.namespace["prompt_lookup_template"].format(numbered_gists, question, "")
        self.role = "readagent_lookup"
        response = self._query_model(prompt).strip()
        self.role = None
        try:
            start, end = response.index("["), response.index("]")
        except ValueError:
            start, end = len(response), 0
        ids = []
        if start < end:
            for token in response[start + 1:end].split(","):
                if token.strip().isnumeric():
                    index = int(token)
                    if 0 <= index < len(pages) and index not in ids:
                        ids.append(index)
        if not ids:
            raise ValueError("ReadAgent did not choose a valid lookup page")
        # The published lookup prompt allows at most six pages. Some providers
        # nevertheless return a longer ordered list; enforce that bound without
        # adding a second model-selection policy or using evaluation labels.
        return ids[:6], response

    @staticmethod
    def expanded_gist_memory(episode, pages, selected_ids):
        """Official expansion: substitute looked-up originals for their gists.

        Original page text is supplied separately as cited evidence with images.
        This ordered context preserves every unselected gist, matching the
        notebook's expanded_shortened_pages operation.
        """
        selected = set(selected_ids)
        return {"readagent_pages": [
            {"page_id": page["page_id"],
             **({"lookup_evidence_id": f"{episode}_readagent_page_{page['page_id']}"}
                if page["page_id"] in selected else {"gist": page["gist"]})}
            for page in pages]}

