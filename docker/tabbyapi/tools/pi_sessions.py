#!/usr/bin/env python3
"""Scripted real pi sessions against the running TabbyAPI, for real-traffic A/B checks.

Wall time is almost all decode (85-95% of each turn), so a pass costs what the model writes: at
~50 tok/s, 3,000 tokens a minute. The default task, review, is the quick keep/drop check (~8 min,
~12k thinking / ~2.5k text / ~13k tool-call tokens, edits included, same start every run); snake
builds from scratch and its length varies (one turn took 9 min). Iterate on an idea with the
engine tools; use this before keeping or dropping it.

Runs the tasks in pi_sessions/prompts.md through the owner's own pi (its system prompt, tools,
extensions and provider, so the requests look like the owner's), each task as one pi session
in a fresh folder, turns one after another with `pi -p --session-id`. Sessions run one at a
time (--jobs 1) or several at once (--jobs 3, like the owner's parallel pi sessions). Every
turn gets a line in <out>/manifest.jsonl with the label and its UTC start/end, so the server's
[decode-stats] lines can be matched to the run afterwards. Run on the machine that has pi
(the Mac); nothing else should use the server while it runs.

  python3 pi_sessions.py run --label pld-on                        # review, ~8 min
  python3 pi_sessions.py run --label pld-on --jobs 3 --tasks review,review,review   # ~10 min
  python3 pi_sessions.py run --label pld-on --tasks snake          # long: ~20 min, varies
  ssh gx10-b2fe.local 'docker logs qwen38-tabby 2>&1' > tabby.log
  python3 pi_sessions.py lines tabby.log --arm pld-off=A --arm pld-on=B | python3 decode_report.py

`lines` keeps the [decode-stats] lines that finished inside a session of the manifest and sets
their arm to the session's label (or to A/B with --arm, which decode_report.py compares with a
bootstrap CI). TabbyAPI logs the clock in UTC without a date, so keep a run's log separate
from other days' logs.
"""
import argparse, datetime, json, os, re, shutil, subprocess, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "pi_sessions", "prompts.md")
FIXTURES = os.path.join(HERE, "pi_sessions", "fixtures")
DEFAULT_OUT = os.path.expanduser("~/scratch/pi-sessions")
_lock = threading.Lock()


def load_tasks(path):
    """{name: {"fixture": dir or None, "turns": [text, ...]}} from prompts.md"""
    tasks, name = {}, None
    for block in re.split(r"^## task (\S+)\s*$", open(path).read(), flags=re.M)[1:]:
        if name is None:
            name = block
            continue
        fixture = re.match(r"\s*fixture: (\S+)\s*\n", block)
        body = block[fixture.end():] if fixture else block
        turns = [t.strip() for t in re.split(r"^---\s*$", body, flags=re.M) if t.strip()]
        tasks[name] = {"fixture": fixture.group(1) if fixture else None, "turns": turns}
        name = None
    return tasks


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def log(msg):
    with _lock:
        print(f"{utc_now():%H:%M:%S}Z {msg}", flush=True)


def run_session(args, run_dir, task_name, task, rep):
    sdir = os.path.join(run_dir, f"{task_name}-{rep}")
    work = os.path.join(sdir, "work")
    if task["fixture"]:
        shutil.copytree(os.path.join(FIXTURES, task["fixture"]), work)
    else:
        os.makedirs(work)
    sid = str(uuid.uuid4())
    for i, text in enumerate(task["turns"], 1):
        cmd = [args.pi, "-p", "--mode", "json", "--session-dir", os.path.join(sdir, "pi"),
               "--session-id", sid] + args.pi_arg + [text]
        start = utc_now()
        log(f"{args.label} {task_name}-{rep} turn {i}/{len(task['turns'])}")
        with open(os.path.join(sdir, f"turn{i}.jsonl"), "w") as out, \
             open(os.path.join(sdir, f"turn{i}.err"), "w") as err:
            try:
                code = subprocess.run(cmd, cwd=work, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                      timeout=args.turn_timeout).returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        end = utc_now()
        rec = {"label": args.label, "jobs": args.jobs, "task": task_name, "rep": rep, "turn": i,
               "session": sid, "start": start.isoformat(), "end": end.isoformat(),
               "seconds": round((end - start).total_seconds(), 1), "exit": code, "timed_out": timed_out}
        if i == len(task["turns"]):
            t = subprocess.run(["node", "--test"], cwd=work, capture_output=True, text=True, timeout=120)
            rec["tests_pass"] = t.returncode == 0
        with _lock, open(os.path.join(args.out, "manifest.jsonl"), "a") as m:
            m.write(json.dumps(rec) + "\n")
        log(f"{args.label} {task_name}-{rep} turn {i} done in {rec['seconds']} s, exit {code}"
            + (" TIMEOUT" if timed_out else "")
            + (f", tests {'pass' if rec['tests_pass'] else 'FAIL'}" if "tests_pass" in rec else ""))
        if timed_out or code:
            return False
    return True


def cmd_run(args):
    tasks = load_tasks(PROMPTS)
    names = args.tasks.split(",")
    for n in names:
        if n not in tasks:
            sys.exit(f"unknown task {n!r}; prompts.md has {', '.join(tasks)}")
    run_dir = os.path.join(args.out, f"{utc_now():%Y%m%dT%H%M%SZ}-{args.label}-j{args.jobs}")
    os.makedirs(run_dir)
    # a task listed twice in --tasks runs as two sessions: snake-1a, snake-1b
    sessions = []
    for rep in range(1, args.reps + 1):
        for i, n in enumerate(names):
            copy = "abcdefghij"[names[:i].count(n)] if names.count(n) > 1 else ""
            sessions.append((n, f"{rep}{copy}"))
    log(f"{len(sessions)} sessions, {args.jobs} at a time, in {run_dir}")
    with ThreadPoolExecutor(args.jobs) as pool:
        ok = list(pool.map(lambda s: run_session(args, run_dir, s[0], tasks[s[0]], s[1]), sessions))
    log(f"finished: {sum(ok)}/{len(ok)} sessions completed every turn")
    return 0 if all(ok) else 1


def seconds_of_day(hms):
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def cmd_lines(args):
    windows = {}
    for line in open(args.manifest):
        r = json.loads(line)
        if args.label and r["label"] not in args.label:
            continue
        w = windows.setdefault(r["session"], [r["label"], r["start"], r["end"]])
        w[1], w[2] = min(w[1], r["start"]), max(w[2], r["end"])
    arm_of = dict(a.split("=", 1) for a in args.arm)
    spans = []
    for label, s, e in windows.values():
        s, e = datetime.datetime.fromisoformat(s), datetime.datetime.fromisoformat(e)
        spans.append((label, seconds_of_day(f"{s:%H:%M:%S}"),
                      (seconds_of_day(f"{e:%H:%M:%S}") + args.slack) % 86400))

    def inside(t, s, e):  # a session may run past midnight UTC
        return s <= t <= e if s <= e else t >= s or t <= e

    head = re.compile(r"\[decode-stats\] at (\S+) arm (\S+)")
    files = [open(f, errors="replace") for f in args.logs] or [sys.stdin]
    kept = 0
    for f in files:
        for line in f:
            m = head.search(line)
            if not m:
                continue
            labels = {lb for lb, s, e in spans if inside(seconds_of_day(m.group(1)), s, e)}
            if len(labels) != 1:
                continue  # outside every session, or inside sessions of two labels
            label = labels.pop()
            sys.stdout.write(line[:m.start(2)] + arm_of.get(label, label) + line[m.end(2):])
            kept += 1
    print(f"pi_sessions: kept {kept} [decode-stats] lines from {len(spans)} sessions", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the scripted sessions")
    r.add_argument("--label", required=True, help="what this run measures, e.g. pld-on")
    r.add_argument("--tasks", default="review",
                   help="comma-separated task names (default: review); a name twice runs two sessions")
    r.add_argument("--reps", type=int, default=1, help="run the task list this many times")
    r.add_argument("--jobs", type=int, default=1, help="sessions at once (1 = sequential)")
    r.add_argument("--turn-timeout", type=int, default=900, help="seconds before a turn is killed")
    r.add_argument("--out", default=DEFAULT_OUT, help=f"results folder (default {DEFAULT_OUT})")
    r.add_argument("--pi", default="pi")
    r.add_argument("--pi-arg", action="append", default=[], help="extra pi argument (repeatable)")
    ln = sub.add_parser("lines", help="[decode-stats] lines of the manifest's sessions, arm = label")
    ln.add_argument("logs", nargs="*", help="TabbyAPI logs (default stdin)")
    ln.add_argument("--manifest", default=os.path.join(DEFAULT_OUT, "manifest.jsonl"))
    ln.add_argument("--label", action="append", help="only these labels (repeatable)")
    ln.add_argument("--arm", action="append", default=[], help="LABEL=ARM, e.g. pld-on=B")
    ln.add_argument("--slack", type=float, default=5, help="seconds after a session's end still counted")
    args = ap.parse_args()
    if args.cmd == "run":
        os.makedirs(args.out, exist_ok=True)
        sys.exit(cmd_run(args))
    cmd_lines(args)


if __name__ == "__main__":
    main()
