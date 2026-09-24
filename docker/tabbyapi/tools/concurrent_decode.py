#!/usr/bin/env python3
"""Decode with N sessions at once through the running TabbyAPI (pi runs up to three).

Sends N requests together (code, prose, tool-call prompts in turn), each generating
MAXTOK tokens with the model's default sampling, REPS times. Prints per-request decode
rate and the aggregate tok/s per round. Compare with TabbyAPI's own log lines.

  python3 docker/tabbyapi/tools/concurrent_decode.py
  N=3 REPS=3 MAXTOK=400 python3 ...
"""
import json, os, time, threading, statistics, urllib.request
URL = f"http://127.0.0.1:{os.environ.get('PORT', '18300')}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
N = int(os.environ.get("N", "3")); REPS = int(os.environ.get("REPS", "3"))
MAXTOK = int(os.environ.get("MAXTOK", "400"))
PROMPTS = [
    "/no_think Write a Python function that parses an nginx access log line into a dict with ip, timestamp, method, path, status, bytes. Include type hints and a docstring.",
    "/no_think Write a vivid 350-word short story about a lighthouse keeper on a remote island in Alaska who finds something unexpected after a storm.",
    "/no_think Explain how Kubernetes horizontal pod autoscaling decides when to scale, with the formula, two pitfalls, and an example HPA YAML.",
]
def one(i, out):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
                       "max_tokens": MAXTOK, "min_tokens": MAXTOK, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if first is None and any((c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content"))
                                     for c in d.get("choices", [])):
                first = time.time()
    out[i] = (usage["completion_tokens"], first, time.time())
rounds = []
for rep in range(REPS):
    out = {}
    ts = [threading.Thread(target=one, args=(i, out)) for i in range(N)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    toks = sum(v[0] for v in out.values())
    span = max(v[2] for v in out.values()) - min(v[1] for v in out.values())
    per = [(v[0] - 1) / (v[2] - v[1]) for v in out.values()]
    rounds.append(toks / span)
    print(f"round {rep}: {N} x {MAXTOK} tokens, aggregate {toks / span:6.1f} tok/s, per request "
          + " ".join(f"{p:5.1f}" for p in per), flush=True)
print(f"RESULT N={N}: aggregate median {statistics.median(rounds):.1f} tok/s "
      f"[{min(rounds):.1f}-{max(rounds):.1f}]", flush=True)
