#!/usr/bin/env python3
"""Thinking-on turns through the running TabbyAPI, for decode checks of the thinking phase.

Sends REPS rounds of the prompts below one at a time (thinking on, the template's default;
the server's default sampling, i.e. the qwen38 preset), MAXTOK tokens at most. The numbers
come from the server's [decode-stats] lines (patch_exllamav3_decode_stats.py); with
EXL3_TRIAL_AB=1 consecutive requests alternate arms, so on the Spark host:

  python3 docker/tabbyapi/tools/api_think.py
  docker logs --since 15m qwen38-tabby 2>&1 | python3 docker/tabbyapi/tools/decode_report.py

Env: REPS=6  MAXTOK=700  PORT=18300
"""
import json, os, time, urllib.request

URL = f"http://127.0.0.1:{os.environ.get('PORT', '18300')}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
REPS = int(os.environ.get("REPS", "6"))
MAXTOK = int(os.environ.get("MAXTOK", "700"))
PROMPTS = [
    "This function should return the k most frequent words, ties broken alphabetically, but the output order is "
    "sometimes wrong. Find the bug and fix it.\n\n```python\nfrom collections import Counter\n\n"
    "def top_k_words(text: str, k: int) -> list[str]:\n    counts = Counter(text.lower().split())\n"
    "    ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)\n"
    "    return [w for w, _ in ranked[:k]]\n```",
    "Write a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, "
    "path, status, bytes. Include a docstring, type hints, and a short usage example.",
    "A service behind a load balancer returns intermittent 502 errors only under load, and only on requests that "
    "take longer than 60 seconds. What are the likely causes and how would you confirm each one?",
]


def one(prompt):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAXTOK}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        u = json.load(r)["usage"]
    return u["completion_tokens"], time.time() - t0


for rep in range(REPS):
    for i, p in enumerate(PROMPTS):
        n, s = one(p)
        print(f"rep {rep} prompt {i}: {n} tokens in {s:.1f} s", flush=True)
