#!/usr/bin/env python3
"""Same client benchmark for any OpenAI-compatible server on the Spark (TabbyAPI, vLLM).

Cold prefill ladder (unique salt per prompt, TTFT with max_tokens=1) and single-stream
decode (400 tokens, thinking off, greedy and sampled) on three prompt classes.
Samples /proc/meminfo MemAvailable in the background. Run on the Spark host:

  PORT=18300 LABEL=tabby python3 compare_bench.py
  PORT=8888 MODEL=qwen3.8-flash-next LABEL=mia python3 compare_bench.py
  SIZES=8000,32000 REPS=2 DECODE_REPS=3 ...
"""
import json, os, random, threading, time, urllib.request

PORT = os.environ.get("PORT", "18300")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
LABEL = os.environ.get("LABEL", PORT)
SIZES = [int(s) for s in os.environ.get("SIZES", "8000,16000,32000,64000,128000").split(",")]
REPS = int(os.environ.get("REPS", "2"))
DECODE_REPS = int(os.environ.get("DECODE_REPS", "3"))
SRC = open(os.path.expanduser(os.environ.get(
    "PROMPT_SRC", "~/models/Qwen3.8-Flash-Next-EXL3/qbench_prompts.md"))).read()
OUT = os.environ.get("OUT", f"compare_{LABEL}.jsonl")

mem = {"min": 1 << 60, "stop": False}
def memwatch():
    while not mem["stop"]:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                mem["min"] = min(mem["min"], int(line.split()[1]))
        time.sleep(0.5)
def mem_now():
    return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 2**20

def filler(tokens, seed):
    random.seed(seed)
    out = [f"Document set {seed}-{random.random()}\n"]
    chars = int(tokens * 3.6)  # ~3.6 chars/token on this text; the real count comes from usage
    while sum(map(len, out)) < chars:
        a = random.randrange(0, len(SRC) - 2000)
        out.append(SRC[a:a + 2000])
    return "".join(out)[:chars]

def ask(messages, max_tokens, sampling):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, **sampling}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = None; text = ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                piece = "".join(delta.get(k) or "" for k in ("content", "reasoning_content", "reasoning"))
                if piece or delta.get("tool_calls"):
                    now = time.time()
                    first = first or now
                    last = now
                    text += piece
    return t0, first, last, usage, text

def log(rec):
    rec.update(label=LABEL, t=time.strftime("%H:%M:%S"))
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

GREEDY = {"temperature": 0.0}
SAMPLED = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
DECODE = {
    "code": "Write a Python function that parses an nginx access log line into a dict with ip, "
            "timestamp, method, path, status, bytes. Include type hints, a docstring and a short test.",
    "devops": "Explain what a Kubernetes readiness probe is and how it differs from a liveness probe, "
              "then give a complete Deployment YAML for an nginx container with both probes.",
    "prose": "Write a 350-word short story about a lighthouse keeper who finds a message in a bottle.",
}

threading.Thread(target=memwatch, daemon=True).start()
log({"phase": "idle", "mem_available_gib": round(mem_now(), 2)})
ask([{"role": "user", "content": "Say hi."}], 4, GREEDY)  # first-request compiles, outside the numbers

for size in SIZES:
    for rep in range(REPS):
        seed = int(time.time() * 1000) ^ (size << 4) ^ rep
        msgs = [{"role": "user", "content": filler(size, seed) + "\n\nSummarize the above in one sentence."}]
        mem["min"] = 1 << 60
        t0, first, _, usage, _ = ask(msgs, 1, GREEDY)
        ttft = first - t0
        n = usage["prompt_tokens"]
        log({"phase": "prefill", "target": size, "rep": rep, "prompt_tokens": n, "ttft_s": round(ttft, 3),
             "prefill_tok_s": round(n / ttft, 1), "mem_min_gib": round(mem["min"] / 2**20, 2)})

for mode, sampling in (("greedy", GREEDY), ("sampled", SAMPLED)):
    for name, prompt in DECODE.items():
        for rep in range(DECODE_REPS):
            t0, first, last, usage, text = ask([{"role": "user", "content": prompt}], 400, sampling)
            n = usage["completion_tokens"]
            rate = (n - 1) / (last - first) if last > first else 0.0
            log({"phase": "decode", "mode": mode, "prompt": name, "rep": rep, "tokens": n,
                 "ttft_s": round(first - t0, 3), "decode_tok_s": round(rate, 1), "head": text[:60]})

mem["stop"] = True
log({"phase": "done", "mem_available_gib": round(mem_now(), 2)})
