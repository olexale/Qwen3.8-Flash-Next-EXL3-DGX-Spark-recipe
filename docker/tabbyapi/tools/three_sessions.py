#!/usr/bin/env python3
"""Three long conversations, interleaved, through the running TabbyAPI (T1 check).

Opens sessions A, B, C with ~TOKENS-token first turns, then ROUNDS rounds of short
follow-up turns (~600 new tokens each) in the order A, B, C. Prints TTFT per turn and
the lowest MemAvailable seen. Whether each follow-up hit the prefix cache is in
TabbyAPI's log lines ("N cached"): docker logs qwen38-tabby | grep cached

  python3 docker/tabbyapi/tools/three_sessions.py
  TOKENS=100000 ROUNDS=2 python3 ...
"""
import json, os, time, random, threading, urllib.request
URL = f"http://127.0.0.1:{os.environ.get('PORT', '18300')}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
TOKENS = int(os.environ.get("TOKENS", "100000"))
ROUNDS = int(os.environ.get("ROUNDS", "2"))
SRC = open(os.path.join(os.path.expanduser(os.environ.get("MODEL_DIR", "~/models/Qwen3.8-Flash-Next-EXL3")),
                        "qbench_prompts.md")).read()
def filler(chars, seed):
    random.seed(seed); out = [f"Session {seed} {random.random()}\n"]
    while sum(map(len, out)) < chars:
        a = random.randrange(0, len(SRC) - 2000); out.append(SRC[a:a + 2000])
    return "".join(out)[:chars]
low = {"avail": 1 << 60}
def watch():
    while True:
        with open("/proc/meminfo") as f:
            m = {l.split(":")[0]: int(l.split()[1]) for l in f}
        low["avail"] = min(low["avail"], m["MemAvailable"]); time.sleep(0.5)
threading.Thread(target=watch, daemon=True).start()
def ask(messages, label):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": 48, "temperature": 0.0,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
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
    print(f"{label:<24} prompt={usage and usage['prompt_tokens']:>7} TTFT={first - t0:6.2f}s", flush=True)
    return text
convs = {}
for i, s in enumerate("ABC"):
    convs[s] = [{"role": "user", "content": filler(TOKENS * 4, 100 + i) + "\n\nSummarize the above in one sentence."}]
    convs[s].append({"role": "assistant", "content": ask(convs[s], f"{s} turn 1 (cold)")})
for rnd in range(ROUNDS):
    for i, s in enumerate("ABC"):
        convs[s].append({"role": "user", "content": filler(2400, 1000 + 10 * rnd + i) + "\n\nAnd this part, in one sentence?"})
        convs[s].append({"role": "assistant", "content": ask(convs[s], f"{s} turn {rnd + 2} (follow-up)")})
with open("/proc/meminfo") as f:
    total = int(next(l for l in f if l.startswith("MemTotal")).split()[1])
print(f"MEM lowest MemAvailable {low['avail']/1024**2:.1f} GiB -> in use {(total - low['avail'])/1024**2:.1f} GiB "
      f"of {total/1024**2:.1f} GiB", flush=True)
