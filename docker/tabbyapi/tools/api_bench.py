#!/usr/bin/env python3
"""End-to-end check through the running TabbyAPI, from the Spark's host.

Prints TTFT and decode rate for: a warm-up, a cold ~20k-token conversation, a
follow-up turn adding ~600 tokens (prefix cache), and a 400-token code answer.
Compare with TabbyAPI's own per-request log lines (docker logs qwen38-tabby).

  python3 docker/tabbyapi/tools/api_bench.py
  PORT=18300 MODEL=qwen3.8-flash-next MODEL_DIR=~/models/Qwen3.8-Flash-Next-EXL3 python3 ...
"""
import json, os, time, random, urllib.request
URL = f"http://127.0.0.1:{os.environ.get('PORT', '18300')}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
SRC = open(os.path.join(os.path.expanduser(os.environ.get("MODEL_DIR", "~/models/Qwen3.8-Flash-Next-EXL3")),
                        "qbench_prompts.md")).read()
def filler(chars, seed):
    random.seed(seed); out = [f"Session {seed} {random.random()}\n"]
    while sum(map(len, out)) < chars:
        a = random.randrange(0, len(SRC) - 2000); out.append(SRC[a:a + 2000])
    return "".join(out)[:chars]
def ask(messages, max_tokens, label):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece and first is None: first = time.time()
                text += piece
    t1 = time.time()
    n = usage["completion_tokens"] if usage else None
    rate = f"{(n - 1) / (t1 - first):.1f} tok/s" if n and first and t1 > first else "-"
    print(f"{label:<34} prompt={usage and usage['prompt_tokens']} TTFT={first - t0:5.2f}s  gen={n} @ {rate}", flush=True)
    return text
ask([{"role": "user", "content": "Say hi."}], 8, "warm-up")
conv = [{"role": "user", "content": filler(80000, 1) + "\n\nSummarize the above in one sentence."}]
reply = ask(conv, 64, "cold ~20k conversation")
conv += [{"role": "assistant", "content": reply}, {"role": "user", "content": filler(2400, 2) + "\n\nAnd this part, in one sentence?"}]
ask(conv, 64, "follow-up turn, ~600 new tokens")
ask([{"role": "user", "content": "/no_think Write a Python function that parses an nginx access log line into a dict with ip, timestamp, method, path, status, bytes. Include type hints and a docstring."}], 400, "code answer (decode speed)")
