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

Workloads edit and rewrite (for prompt-lookup drafting, patch_exllamav3_pld.py) put a real
source file from the image (exllamav3/generator/draft_confidence.py, ~110 lines) in context:
edit asks for two changes with the edit tool (multi-line oldText quoted from the file),
rewrite for the whole file back with the write tool. They stop at <|im_end|> (at most
NTOK_LONG tokens) and report tok/s over the tokens actually generated.

PLDS=0,1 runs every cell with prompt-lookup drafting off and on (the generator module's flag,
toggled in process; start the container with -e EXL3_PLD=1 so the cache gets the longer
rollback history for both). EXL3_PLD_MAX / _MIN_MATCH / _NGRAM as in the patch.
VARIANTS="mtp:PLD=0;pld7:PLD=1,PLD_MAX=7;pld15:PLD=1,PLD_MAX=15,BSZN=16" instead names
in-process variants (module globals: PLD, PLD_MAX, MIN_MATCH, BSZN = the fused decode MoE's
row limit from patch_exllamav3_bszn16.py). Load with EXL3_PLD_MAX / EXL3_MOE_BSZN_MAX at the
largest value used, so the buffers are sized for it.

Env: NDTS=3,4,5,6,7  CONFS=0.4,0.5,0.6,0.7,0.8  REPS=10  NTOK=320  NTOK_LONG=1200
     WORKLOADS=code,prose,tool  PLDS=0
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
NTOK_LONG = int(os.environ.get("NTOK_LONG", "1200"))
PLDS = [int(x) for x in os.environ.get("PLDS", "0").split(",")]
VARIANTS = [(v.split(":")[0], dict(kv.split("=") for kv in v.split(":")[1].split(",") if kv))
            for v in os.environ.get("VARIANTS", "").split(";") if v] or [(str(p), {"PLD": p}) for p in PLDS]
import exllamav3.modules.block_sparse_mlp as _bsm, exllamav3.modules.mlp as _mlpm
def apply_variant(v):
    for k, val in v.items():
        if k == "PLD":
            if hasattr(G, "_PLD"): G._PLD = val not in ("0", 0)
        elif k == "PLD_MAX": G._PLD_MAX = int(val)
        elif k == "MIN_MATCH": G._PLD_MIN_MATCH = int(val)
        elif k == "BSZN": _bsm.MAX_BSZN = _mlpm.MAX_BSZN = int(val)
        else: raise ValueError(k)
import exllamav3.generator.generator as G
print("CONFIG", f"NDTS={NDTS} CONFS={CONFS} REPS={REPS} NTOK={NTOK} WORKLOADS={WORKLOADS} PLDS={PLDS} "
      f"PLD_MAX={getattr(G, '_PLD_MAX', None)} MIN_MATCH={getattr(G, '_PLD_MIN_MATCH', None)} "
      f"NGRAM={getattr(G, '_PLD_NGRAM', None)}", flush=True)
if any(PLDS): assert hasattr(G, "_PLD"), "image without patch_exllamav3_pld.py"

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
import exllamav3.generator.draft_confidence as _dc
REAL_PATH = "exllamav3/generator/draft_confidence.py"
REAL = open(_dc.__file__).read()
# edit / rewrite: rendered with the model's own chat template (tool calls as
# <function=...><parameter=...> blocks with raw multi-line values, as TabbyAPI serves them)
import json as _json, jinja2 as _jinja2
def _raise(m): raise RuntimeError(m)
_env = _jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
_env.globals["raise_exception"] = _raise
_env.filters["tojson"] = lambda x, indent=None, **k: _json.dumps(x, ensure_ascii=False, indent=indent)
_TPL = _env.from_string(open(os.path.join(MODEL, "chat_template.jinja")).read())
def _fn(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": list(props)}}}
PI_TOOLS = [
    _fn("read", "Read the contents of a file.", path="Path to the file"),
    _fn("edit", "Edit a file by replacing exact text. oldText must match the file exactly, including whitespace.",
        path="Path to the file", oldText="Exact text to replace", newText="Replacement text"),
    _fn("write", "Write a file, replacing its whole content.", path="Path to the file", content="New file content"),
    _fn("bash", "Run a shell command.", command="The command"),
]
def _agent(task):
    msgs = [{"role": "system", "content": "You are a coding agent working in the user's repository."},
            {"role": "user", "content": task},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {"path": REAL_PATH}}}]},
            {"role": "tool", "content": REAL}]
    return _TPL.render(messages=msgs, tools=PI_TOOLS, add_generation_prompt=True, enable_thinking=False)

PROMPTS = {
    "code": "<|im_start|>user\nWrite a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example.<|im_end|>\n" + NOTHINK,
    "prose": "<|im_start|>user\nWrite a vivid 350-word short story about a lighthouse keeper on a remote island in Alaska who discovers something unexpected washed ashore after a storm.<|im_end|>\n" + NOTHINK,
    "tool": "<|im_start|>system\nYou are a coding agent working in the user's repository.\n\n" + TOOLS + "<|im_end|>\n"
            "<|im_start|>user\nIn src/app/settings.py, make Settings.load also validate that workers is at least 1 and log_level is one of DEBUG, INFO, WARNING, ERROR, and make merge() validate the merged result the same way. Use the edit tool.<|im_end|>\n"
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"src/app/settings.py\"}}\n</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n" + FILE + "</tool_response><|im_end|>\n" + NOTHINK,
    "edit": _agent("In " + REAL_PATH + ", make two changes with the edit tool (one call per change; oldText must "
                   "quote the whole method being changed): rename decay_step() to age_step(), and make estimate() "
                   "return 0.0 when no populated bin is at or below the score instead of using the nearest bin above."),
    "rewrite": _agent("Rewrite " + REAL_PATH + " with the write tool: the complete file, unchanged except that every "
                      "docstring is shortened to a single line."),
}
LONG = {"edit", "rewrite"}

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)
model = Model.from_config(config)
dm = Model.from_config(config, component="mtp")
qkw = dict(layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
# Caches must exist before the load; one pair per draft length (max_history = ndt)
caches = {ndt: (Cache(model, max_num_tokens=16384, max_batch_size=4, max_history=ndt, **qkw),
                Cache(dm, max_num_tokens=16384, max_batch_size=4, max_history=ndt, **qkw)) for ndt in NDTS}
dm.load(progressbar=False)
model.load(progressbar=False, max_chunk_size=8192, max_batch_size=4)
ids = {k: tok.encode(v, add_bos=False, encode_special_tokens=True) for k, v in PROMPTS.items()}

IM_END = tok.single_id("<|im_end|>")
DUMP = os.environ.get("DUMP", "")
ROUNDSTAT = os.environ.get("ROUNDSTAT", "0") == "1"
roundstat = collections.defaultdict(list); last_draft = []
def _track(kind, fn):
    def f(*a, **k):
        d = fn(*a, **k)
        if d is not None: last_draft.append((kind, d.shape[-1]))
        return d
    return f
run_dump = []
def run(gen, w, seed):
    run.dump = run_dump; run.variant = getattr(run, "variant", "")
    torch.manual_seed(seed)
    job = Job(input_ids=ids[w], max_new_tokens=NTOK_LONG if w in LONG else NTOK,
              stop_conditions=[IM_END] if w in LONG else [],
              sampler=ComboSampler(temperature=1.0, top_k=20, top_p=0.95))
    gen.enqueue(job); t0 = None
    while gen.num_remaining_jobs():
        ta = time.perf_counter(); acc0 = job.accepted_draft_tokens; last_draft.clear()
        for r in gen.iterate():
            if r.get("error"): raise RuntimeError(r["error"])
            if r.get("stage") == "streaming" and t0 is None: t0 = time.perf_counter()
        if ROUNDSTAT and t0 is not None and last_draft:
            kind, width = last_draft[-1]
            roundstat[(w, run.variant, kind, width)].append((time.perf_counter() - ta, job.accepted_draft_tokens - acc0))
    dt = time.perf_counter() - t0
    tot = job.accepted_draft_tokens + job.rejected_draft_tokens
    run.ntok = job.new_tokens
    if DUMP:
        seq = job.sequences[0].sequence_ids.torch().view(-1).tolist()
        run.dump.append({"w": w, "seed": seed, "pld": getattr(G, "_PLD", False), "variant": run.variant, "prompt_len": len(ids[w][0]), "ids": seq})
    return (job.new_tokens - 1) / dt, 100 * job.accepted_draft_tokens / max(tot, 1)

res = collections.defaultdict(list)
for ndt in NDTS:
    cache, dcache = caches[ndt]
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                    max_batch_size=4, max_chunk_size=8192, num_draft_tokens=ndt,
                    dynamic_draft_tokens=True, draft_confidence=CONFS[0])
    if ROUNDSTAT:
        if hasattr(gen, "_pld_round_drafts"): gen._pld_round_drafts = _track("pld", gen._pld_round_drafts)
        _mtp = gen.iterate_draftmodel_mtp_gen
        gen.iterate_draftmodel_mtp_gen = lambda *a, **k: (lambda d: d if d is None or last_draft else (last_draft.append(("mtp", d.shape[-1])) or d))(_mtp(*a, **k))
    cals = {(c, pl): DraftConfidenceCalibrator(c) for c in CONFS for pl, _ in VARIANTS}
    for c in CONFS:  # warm-up: this cache shape, and each calibrator
        for pl, var in VARIANTS:
            apply_variant(var); run.variant = "warmup"
            gen.draft_calibrator = cals[(c, pl)]
            for w in WORKLOADS: run(gen, w, 0)
    for rep in range(REPS):
        for c in CONFS:
            for pl, var in VARIANTS:
                apply_variant(var); run.variant = pl
                gen.draft_calibrator = cals[(c, pl)]
                for w in WORKLOADS:
                    tps, acc = run(gen, w, 1000 + rep)
                    res[(ndt, c, pl, w)].append((tps, acc))
                    print(f"SAMPLE ndt={ndt} conf={c} v={pl:<6} {w:<7} rep={rep} {tps:6.1f} tok/s accept {acc:4.0f}% "
                          f"tokens {run.ntok}", flush=True)
    del gen

def fmt(v):
    return f"{statistics.median(v):5.1f} [{min(v):5.1f}-{max(v):5.1f}]"
print("SUMMARY decode tok/s median [range], acceptance % median")
for w in WORKLOADS:
    for ndt in NDTS:
        for pl, _ in VARIANTS:
            print(f"SUMMARY {w:<7} ndt={ndt} v={pl:<6} " + "  ".join(
                f"c{c}: {fmt([t for t, _ in res[(ndt, c, pl, w)]])} {statistics.median([a for _, a in res[(ndt, c, pl, w)]]):3.0f}%"
                for c in CONFS), flush=True)
if ROUNDSTAT:
    print("ROUND workload kind drafts: rounds, median ms, mean accepted drafts")
    for (w, vn, kind, width), v in sorted(roundstat.items()):
        print(f"ROUND {w:<7} {vn:<6} {kind} {width:2d}: {len(v):5d} {1000 * statistics.median(t for t, _ in v):6.1f} ms "
              f"{sum(a for _, a in v) / len(v):5.2f}", flush=True)
if DUMP:
    _json.dump(run_dump, open(DUMP, "w"))
print("DONE", flush=True)
