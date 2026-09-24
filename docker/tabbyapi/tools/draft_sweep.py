#!/usr/bin/env python3
"""Decode speed and draft acceptance vs num_draft_tokens x draft_confidence, sampled output.

Runs inside the qwen38-exl3-tabby image (SCRIPT=draft_sweep.py run_engine_bench.sh <name>),
TabbyAPI stopped. One model load; one Cache/Generator per draft length (Generators must not
share a cache), and per confidence value its own online calibrator, kept across samples as a
running server would keep it. Dynamic drafting on, the
qwen38 preset's sampling (temperature 1.0, top_k 20, top_p 0.95). Drafting does not change
the output distribution (the fork accepts a drafted token only if it matches the target's
sample), so this only measures speed.

Workloads: code (write a function), prose (short story), tool (pi-style edit tool call on a
file shown in the context). Prints one SAMPLE line per run and a SUMMARY per cell: median and
range of decode tok/s and acceptance over REPS samples.

Env: NDTS=3,4,5,6,7  CONFS=0.4,0.5,0.6,0.7,0.8  REPS=10  NTOK=320  WORKLOADS=code,prose,tool
"""
import os, time, statistics, collections
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler
from exllamav3.generator.draft_confidence import DraftConfidenceCalibrator

MODEL = os.environ.get("MODEL", "/models/qwen3.8-flash-next")
NDTS = [int(x) for x in os.environ.get("NDTS", "3,4,5,6,7").split(",")]
CONFS = [float(x) for x in os.environ.get("CONFS", "0.4,0.5,0.6,0.7,0.8").split(",")]
REPS = int(os.environ.get("REPS", "10"))
NTOK = int(os.environ.get("NTOK", "320"))
WORKLOADS = os.environ.get("WORKLOADS", "code,prose,tool").split(",")
print("CONFIG", f"NDTS={NDTS} CONFS={CONFS} REPS={REPS} NTOK={NTOK} WORKLOADS={WORKLOADS}", flush=True)

FILE = '''import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Settings:
    """Runtime settings, loaded from a JSON file."""
    host: str = "127.0.0.1"
    port: int = 8080
    workers: int = 4
    allowed_origins: list[str] = field(default_factory=list)
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: Path) -> "Settings":
        data = json.loads(path.read_text())
        settings = cls(**data)
        if settings.port < 1 or settings.port > 65535:
            raise ValueError(f"invalid port {settings.port}")
        log.info("loaded settings from %s", path)
        return settings

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2))


def merge(base: Settings, override: dict) -> Settings:
    values = dict(base.__dict__)
    for key, value in override.items():
        if key not in values:
            log.warning("unknown setting %s", key)
            continue
        values[key] = value
    return Settings(**values)
'''
TOOLS = '''# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "read", "description": "Read the contents of a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
{"type": "function", "function": {"name": "edit", "description": "Edit a file by replacing exact text. oldText must match the file exactly, including whitespace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "oldText": {"type": "string"}, "newText": {"type": "string"}}, "required": ["path", "oldText", "newText"]}}}
{"type": "function", "function": {"name": "bash", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>'''
NOTHINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPTS = {
    "code": "<|im_start|>user\nWrite a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example.<|im_end|>\n" + NOTHINK,
    "prose": "<|im_start|>user\nWrite a vivid 350-word short story about a lighthouse keeper on a remote island in Alaska who discovers something unexpected washed ashore after a storm.<|im_end|>\n" + NOTHINK,
    "tool": "<|im_start|>system\nYou are a coding agent working in the user's repository.\n\n" + TOOLS + "<|im_end|>\n"
            "<|im_start|>user\nIn src/app/settings.py, make Settings.load also validate that workers is at least 1 and log_level is one of DEBUG, INFO, WARNING, ERROR, and make merge() validate the merged result the same way. Use the edit tool.<|im_end|>\n"
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"src/app/settings.py\"}}\n</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n" + FILE + "</tool_response><|im_end|>\n" + NOTHINK,
}

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)
model = Model.from_config(config)
dm = Model.from_config(config, component="mtp")
dm.load(progressbar=False)
model.load(progressbar=False, max_chunk_size=8192, max_batch_size=4)
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
ids = {k: tok.encode(v, add_bos=False) for k, v in PROMPTS.items()}

def run(gen, w, seed):
    torch.manual_seed(seed)
    job = Job(input_ids=ids[w], max_new_tokens=NTOK, stop_conditions=[],
              sampler=ComboSampler(temperature=1.0, top_k=20, top_p=0.95))
    gen.enqueue(job); t0 = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if r.get("stage") == "streaming" and t0 is None: t0 = time.perf_counter()
    dt = time.perf_counter() - t0
    tot = job.accepted_draft_tokens + job.rejected_draft_tokens
    return (NTOK - 1) / dt, 100 * job.accepted_draft_tokens / max(tot, 1)

res = collections.defaultdict(list)
for ndt in NDTS:
    cache = Cache(model, max_num_tokens=16384, max_batch_size=4, max_history=ndt, **qkw)
    dcache = Cache(dm, max_num_tokens=16384, max_batch_size=4, max_history=ndt, **qkw)
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                    max_batch_size=4, max_chunk_size=8192, num_draft_tokens=ndt,
                    dynamic_draft_tokens=True, draft_confidence=CONFS[0])
    cals = {c: DraftConfidenceCalibrator(c) for c in CONFS}
    for c in CONFS:  # warm-up: this cache shape, and each calibrator
        gen.draft_calibrator = cals[c]
        for w in WORKLOADS: run(gen, w, 0)
    for rep in range(REPS):
        for c in CONFS:
            gen.draft_calibrator = cals[c]
            for w in WORKLOADS:
                tps, acc = run(gen, w, 1000 + rep)
                res[(ndt, c, w)].append((tps, acc))
                print(f"SAMPLE ndt={ndt} conf={c} {w:<5} rep={rep} {tps:6.1f} tok/s accept {acc:4.0f}%", flush=True)
    del gen, cache, dcache
    torch.cuda.empty_cache()

def fmt(v):
    return f"{statistics.median(v):5.1f} [{min(v):5.1f}-{max(v):5.1f}]"
print("SUMMARY decode tok/s median [range], acceptance % median")
for w in WORKLOADS:
    for ndt in NDTS:
        print(f"SUMMARY {w:<5} ndt={ndt} " + "  ".join(
            f"c{c}: {fmt([t for t, _ in res[(ndt, c, w)]])} {statistics.median([a for _, a in res[(ndt, c, w)]]):3.0f}%"
            for c in CONFS), flush=True)
print("DONE", flush=True)
