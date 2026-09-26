#!/usr/bin/env python3
"""Three harnesses in parallel through the running TabbyAPI (Plan C, C2 multi-harness check).

One thread per harness, each running SESSIONS sessions of TURNS turns one after another:

  pi      a stable ~5k-token system prompt + four tools (the same in every session); turn 1's
          user message carries a ~180-token block that later turns do not repeat (as pi did
          on 2026-09-25); later turns send ~600-token tool results (the filler quotes chat
          markers, as a file with a chat template in it would)
  cc      a system prompt that starts with the date, working directory and git status (new in
          every session, as Claude Code does), ~3k tokens, the same tools, same turn shape
  chat    a one-line system prompt and short questions

The assistant's reasoning and text are sent back each turn. Content is fixed by SEED, so a
fresh server with and without the patch sees the same prompts. Prints per request TTFT and
decode rate (min_tokens forces OUT tokens), per harness medians, and the largest GPU memory
of TabbyAPI's process (nvidia-smi --query-compute-apps). Resume positions and anchors are in
TabbyAPI's log with EXL3_PREFIX_DIAG=1.

  python3 docker/tabbyapi/tools/multi_harness.py
  SESSIONS=2 TURNS=4 OUT=128 SEED=1 python3 ...
"""
import json, os, random, statistics, subprocess, threading, time, urllib.request
URL = f"http://127.0.0.1:{os.environ.get('PORT', '18300')}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
SESSIONS = int(os.environ.get("SESSIONS", "2")); TURNS = int(os.environ.get("TURNS", "4"))
OUT = int(os.environ.get("OUT", "128")); SEED = os.environ.get("SEED", "1")
SRC = open(os.path.join(os.path.expanduser(os.environ.get("MODEL_DIR", "~/models/Qwen3.8-Flash-Next-EXL3")),
                        "qbench_prompts.md")).read()

def filler(chars, seed):
    rnd = random.Random(f"{SEED}/{seed}"); out = []
    while sum(map(len, out)) < chars:
        a = rnd.randrange(0, len(SRC) - 1000); out.append(SRC[a:a + 1000])
    return "".join(out)[:chars]

def tool(name, desc, params):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {p: {"type": "string", "description": p} for p in params},
        "required": params[:1]}}}
TOOLS = [tool("read", "Read a file. Output is truncated to 2000 lines or 50KB.", ["path", "offset", "limit"]),
         tool("bash", "Execute a bash command in the current working directory.", ["command", "timeout"]),
         tool("edit", "Edit a file by replacing exact text.", ["path", "oldText", "newText"]),
         tool("write", "Write content to a file, creating parent directories.", ["path", "content"])]

peak = {"mib": 0}
def watch():
    while True:
        try:
            q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout
            vals = [int(l.split(",")[1]) for l in q.splitlines() if l.strip() and l.split(",")[1].strip().isdigit()]
            # TabbyAPI's process: the largest compute app (the voice stack's are far smaller)
            if vals: peak["mib"] = max(peak["mib"], max(vals))
        except Exception:
            pass
        time.sleep(1)
threading.Thread(target=watch, daemon=True).start()

lock = threading.Lock(); rows = []
def ask(harness, label, messages, tools):
    body = {"model": MODEL, "messages": messages, "max_tokens": OUT, "min_tokens": OUT,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    if tools: body["tools"] = tools
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None; text = ""; reasoning = ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("content") or ""); rpiece = (delta.get("reasoning_content") or "")
                if first is None and (piece or rpiece or delta.get("tool_calls")): first = time.time()
                text += piece; reasoning += rpiece
    t1 = time.time(); n = usage["completion_tokens"] if usage else 0
    ttft = (first or t1) - t0; tps = (n - 1) / (t1 - first) if first and n > 1 and t1 > first else 0.0
    with lock:
        rows.append((harness, label, ttft, tps))
        print(f"{harness:<5} {label:<10} prompt={usage and usage['prompt_tokens']:>6} TTFT={ttft:5.2f}s "
              f"decode={tps:5.1f} tok/s", flush=True)
    msg = {"role": "assistant", "content": text}
    if reasoning: msg["reasoning_content"] = reasoning
    return msg

def pi_like(s):
    return [{"role": "system", "content": "You are an expert coding assistant.\n" + filler(19000, "pi-sys")}], \
        filler(700, f"pi-block-{s}") + "\n\n", TOOLS

def cc_like(s):
    head = (f"Today's date: 2026-09-{10 + s:02d}. Working directory: /home/dev/repo-{s}-{SEED}\n"
            f"git status: branch feature-{s}, {3 + s} files modified\n")
    return [{"role": "system", "content": head + filler(12000, "cc-sys")}], "", TOOLS

def chat_like(s):
    return [{"role": "system", "content": "You are a helpful assistant."}], "", None

def harness(name, make):
    for s in range(SESSIONS):
        msgs, block, tools = make(s)
        for t in range(TURNS):
            if name == "chat":
                new = {"role": "user", "content": f"Question {s}.{t}: " + filler(300, f"chat-{s}-{t}")}
            elif t == 0:
                new = {"role": "user", "content": block + "Task: " + filler(200, f"{name}-task-{s}")}
            else:
                new = {"role": "tool", "tool_call_id": f"call_{t}", "content": filler(2400, f"{name}-tool-{s}-{t}")}
            reply = ask(name, f"s{s + 1} t{t + 1}", msgs + [new], tools)
            # pi sends turn 1's message without the block from turn 2 on
            if name == "pi" and t == 0:
                new = {"role": "user", "content": new["content"][len(block):]}
            if name != "chat":  # the next turn answers a tool call
                reply["tool_calls"] = [{"id": f"call_{t + 1}", "type": "function",
                                        "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})}}]
            msgs = msgs + [new, reply]

threads = [threading.Thread(target=harness, args=a) for a in (("pi", pi_like), ("cc", cc_like), ("chat", chat_like))]
for th in threads: th.start()
for th in threads: th.join()
for h in ("pi", "cc", "chat"):
    r = [x for x in rows if x[0] == h]
    t1 = [x[2] for x in r if x[1].endswith("t1")]; tn = [x[2] for x in r if not x[1].endswith("t1")]
    print(f"SUMMARY {h:<5} TTFT turn 1 median {statistics.median(t1):.2f}s, later turns median "
          f"{statistics.median(tn):.2f}s, decode median {statistics.median([x[3] for x in r]):.1f} tok/s", flush=True)
print(f"MEM TabbyAPI peak GPU memory {peak['mib'] / 1024:.2f} GiB", flush=True)
