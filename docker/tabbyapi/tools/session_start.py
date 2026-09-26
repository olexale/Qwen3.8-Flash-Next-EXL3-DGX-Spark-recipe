#!/usr/bin/env python3
"""Time to first token at the start of pi-like sessions, through the running TabbyAPI (Plan C, C2).

Each rep builds a fresh ~5k-token system prompt (a nonce first, so nothing from earlier reps
is cached), laid out like pi's: filler, then the working directory ~380 tokens before the
end, then a fixed tail; plus four pi-like tools. Then, in order:

  s1 turn 1   cwd A, the user message with a ~180-token block that pi sends on turn 1 only
  s1 turn 2   the same message without that block, an assistant tool call and its result
  s2 turn 1   cwd A again, another task (a repeat session in the same project)
  s3 turn 1   cwd B (a second project: shares the prefix up to the working directory)
  s4 turn 1   cwd C (a third project)

Prints TTFT per request and the median per step over REPS. Where each request resumed is in
TabbyAPI's log with EXL3_PREFIX_DIAG=1: docker logs qwen38-tabby | grep prefix-diag

  python3 docker/tabbyapi/tools/session_start.py
  REPS=5 python3 ...
"""
import json, os, random, statistics, time, urllib.request, uuid
URL = f"http://127.0.0.1:{os.environ.get('PORT', '18300')}/v1/chat/completions"
MODEL = os.environ.get("MODEL", "qwen3.8-flash-next")
REPS = int(os.environ.get("REPS", "3"))
SRC = open(os.path.join(os.path.expanduser(os.environ.get("MODEL_DIR", "~/models/Qwen3.8-Flash-Next-EXL3")),
                        "qbench_prompts.md")).read()

def filler(chars, seed):
    rnd = random.Random(seed); out = []
    while sum(map(len, out)) < chars:
        a = rnd.randrange(0, len(SRC) - 1000); out.append(SRC[a:a + 1000])
    return "".join(out)[:chars]

def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
        "required": req}}}

TOOLS = [
    tool("read", "Read the contents of a file. Supports text files and images. Output is truncated to "
         "2000 lines or 50KB. Use offset/limit for large files.",
         {"path": ("string", "Path to the file to read (relative or absolute)"),
          "offset": ("number", "Line number to start reading from (1-indexed)"),
          "limit": ("number", "Maximum number of lines to read")}, ["path"]),
    tool("bash", "Execute a bash command in the current working directory. Returns stdout and stderr. "
         "Output is truncated to the last 2000 lines or 50KB.",
         {"command": ("string", "Bash command to execute"),
          "timeout": ("number", "Timeout in seconds (optional, no default timeout)")}, ["command"]),
    tool("edit", "Edit a file by replacing exact text. The oldText must match exactly (including "
         "whitespace). Use this for precise, surgical edits.",
         {"path": ("string", "Path to the file to edit (relative or absolute)"),
          "oldText": ("string", "Exact text to find and replace (must match exactly)"),
          "newText": ("string", "New text to replace the old text with")}, ["path", "oldText", "newText"]),
    tool("write", "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
         "Automatically creates parent directories.",
         {"path": ("string", "Path to the file to write (relative or absolute)"),
          "content": ("string", "Content to write to the file")}, ["path", "content"]),
]

def system(nonce, cwd):
    head = f"Session prefix {nonce}\n" + filler(15000, nonce)
    tail = filler(1500, "tail")
    return f"{head}\n\nCurrent working directory: {cwd}\n\n{tail}"

def ask(messages, label):
    body = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS, "max_tokens": 16,
                       "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                    first = time.time()
    ttft = (first or time.time()) - t0
    print(f"{label:<12} prompt={usage and usage['prompt_tokens']:>6} TTFT={ttft:5.2f}s", flush=True)
    return ttft

steps = {}
for rep in range(REPS):
    nonce = uuid.uuid4().hex
    block = "<context>\n" + filler(700, "block" + nonce) + "\n</context>\n\n"
    task = "List the Python files in this project and tell me which one is the entry point."
    s1 = [{"role": "system", "content": system(nonce, "/home/dev/project-a")}]
    steps.setdefault("s1 turn 1", []).append(ask(s1 + [{"role": "user", "content": block + task}], "s1 turn 1"))
    turn2 = s1 + [
        {"role": "user", "content": task},
        {"role": "assistant", "content": "", "reasoning_content": "I should list the files first.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "bash", "arguments": json.dumps({"command": "ls *.py"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": filler(9000, "ls" + nonce)},
    ]
    steps.setdefault("s1 turn 2", []).append(ask(turn2, "s1 turn 2"))
    for name, cwd, t in (("s2 turn 1", "/home/dev/project-a", "Explain what the build script does."),
                         ("s3 turn 1", "/home/dev/project-b", "Find the failing test and explain why it fails."),
                         ("s4 turn 1", "/home/dev/project-c", "Summarize the README in two sentences.")):
        msgs = [{"role": "system", "content": system(nonce, cwd)}, {"role": "user", "content": block + t}]
        steps.setdefault(name, []).append(ask(msgs, name))
print("\nmedian TTFT over", REPS, "reps: " + ", ".join(
    f"{k} {statistics.median(v):.2f}s [{min(v):.2f}-{max(v):.2f}]" for k, v in steps.items()), flush=True)
