#!/usr/bin/env python3
"""Decode speed of pi-style edit / rewrite turns through the running TabbyAPI.

The conversation: a coding-agent system prompt with pi's tools (read, edit, write, bash), a
task, an assistant `read` tool call, the tool result (a real ~110-line source file from the
repo: tools/concurrent_decode.py by default), then the model's turn. TabbyAPI
renders it with the model's chat template, so tool calls are <function=...><parameter=...>
blocks with raw multi-line values (oldText quotes the file verbatim). Thinking off
(chat_template_kwargs), default sampling. TabbyAPI buffers tool calls until they parse, so
the decode rate is TabbyAPI's own ("N tokens generated at X T/s" in `docker logs`; run on the
Spark host). Reports it per request and the median per task.

  python3 docker/tabbyapi/tools/api_edit.py            REPS=5 TASKS=edit,rewrite
"""
import json, os, re, subprocess, time, statistics, urllib.request
CONTAINER = os.environ.get("TABBY_CONTAINER", "qwen38-tabby")
URL = f"http://127.0.0.1:{os.environ.get('PORT', '18300')}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
REPS = int(os.environ.get("REPS", "5")); TASKS = os.environ.get("TASKS", "edit,rewrite").split(",")
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.environ.get("FILE", os.path.join(HERE, "concurrent_decode.py"))
REL = os.path.relpath(PATH, os.path.join(HERE, "../../.."))
SRC = open(PATH).read()
def fn(n, d, **pr):
    return {"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in pr.items()},
        "required": list(pr)}}}
TOOLS = [fn("read", "Read the contents of a file.", path="Path to the file"),
         fn("edit", "Edit a file by replacing exact text. oldText must match the file exactly, including whitespace.",
            path="Path to the file", oldText="Exact text to replace", newText="Replacement text"),
         fn("write", "Write a file, replacing its whole content.", path="Path to the file", content="New file content"),
         fn("bash", "Run a shell command.", command="The command")]
TASK = {
    "edit": f"In {REL}, make two changes with the edit tool (one call per change; oldText must quote the whole "
            "function or statement being changed): make one() retry the request once on a network error, and "
            "put the three PROMPTS in reverse order.",
    "rewrite": f"Rewrite {REL} with the write tool: the complete file, unchanged except that every comment and "
               "docstring is removed.",
}
def one(task):
    msgs = [{"role": "system", "content": "You are a coding agent working in the user's repository."},
            {"role": "user", "content": TASK[task]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": json.dumps({"path": REL})}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": SRC}]
    body = json.dumps({"model": MODEL, "messages": msgs, "tools": TOOLS, "max_tokens": 3000, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    first = last = None; usage = None; n = 0
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 1))
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content"):
                    now = time.time(); first = first or now; last = now; n += 1
    toks = usage["completion_tokens"] if usage else n
    time.sleep(0.5)
    log = subprocess.run(["docker", "logs", "--since", since, CONTAINER], capture_output=True, text=True)
    m = re.findall(r"(\d[\d,]*) tokens generated at\s+([\d.]+) T/s", log.stdout + log.stderr)
    return toks, float(m[-1][1]) if m else float("nan")
res = {t: [] for t in TASKS}
for rep in range(REPS):
    for t in TASKS:
        toks, tps = one(t)
        res[t].append(tps)
        print(f"{t:<8} rep {rep}: {toks} tokens, {tps:6.1f} tok/s", flush=True)
for t in TASKS:
    v = res[t]
    print(f"RESULT {t}: median {statistics.median(v):.1f} tok/s [{min(v):.1f}-{max(v):.1f}]", flush=True)
