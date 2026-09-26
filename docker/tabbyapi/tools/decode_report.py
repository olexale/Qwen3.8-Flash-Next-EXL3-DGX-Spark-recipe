#!/usr/bin/env python3
"""Per-arm decode report from TabbyAPI's [decode-stats] lines (patch_exllamav3_decode_stats.py).

    { cat logs/qwen38-tabby-*.log; docker logs qwen38-tabby 2>&1; } | python3 decode_report.py
    python3 decode_report.py logs/qwen38-tabby-*.log --since 14:00 --min-think 32

Per arm (A = candidate off, B = on, "-" = no trial): requests, per-request thinking / text /
tool-call tok/s (median, IQR), pooled tok/s (phase tokens / phase time over all requests),
draft acceptance and tokens per round in each phase, TTFT. With both arms present: the
B-vs-A difference of median and pooled thinking tok/s in %, with a bootstrap 95% CI
(requests resampled within each arm), and the same for tool calls.

A request counts for a phase's per-request rate only with at least --min-think (thinking) or
--min-tool (tool, text) tokens in that phase, so a few tokens split across one round do not
dominate the medians. Only standard library; no text or ids are read (the lines hold none).
"""
import argparse, random, re, statistics, sys

PHASES = ("thinking", "text", "tool")
_NUM = r"([0-9.]+)"
_RE_HEAD = re.compile(r"\[decode-stats\] at (\S+) arm (\S+)")
_RE_PROMPT = re.compile(r"prompt (\d+), ttft " + _NUM + " s")
_RE_TOK = re.compile(r"tokens (\d+) \(thinking (\d+), text (\d+), tool (\d+)\)")
_RE_TIME = re.compile(r"time thinking " + _NUM + " s, text " + _NUM + " s, tool " + _NUM + " s")
_RE_CNT = re.compile(r"(thinking|text|tool) rounds (\d+), drafted (\d+), accepted (\d+), "
                     r"lookup (\d+) \(drafted (\d+), accepted (\d+)\)")


def parse_line(line):
    """One [decode-stats] line as a dict, or None if the line is not one (or is an error line)"""
    h = _RE_HEAD.search(line)
    if not h:
        return None
    p, t, tm = _RE_PROMPT.search(line), _RE_TOK.search(line), _RE_TIME.search(line)
    if not (p and t and tm):
        return None
    r = {"clock": h.group(1), "arm": h.group(2), "prompt": int(p.group(1)), "ttft": float(p.group(2)),
         "tokens": int(t.group(1)), "tok": dict(zip(PHASES, map(int, t.groups()[1:]))),
         "time": dict(zip(PHASES, map(float, tm.groups()))), "cnt": {}}
    for m in _RE_CNT.finditer(line):
        r["cnt"][m.group(1)] = dict(zip(("rounds", "drafted", "accepted", "lookup", "lookup_drafted",
                                         "lookup_accepted"), map(int, m.groups()[1:])))
    tail = line.rstrip().rsplit(" | ", 1)[-1]
    if not _RE_CNT.search(tail) and not tail.startswith("time "):
        r["extra"] = tail
    return r


def rate(r, phase):
    s = r["time"][phase]
    return r["tok"][phase] / s if s > 0 else None


def pooled(rs, phase):
    tok = sum(r["tok"][phase] for r in rs)
    s = sum(r["time"][phase] for r in rs)
    return tok / s if s > 0 else None


def quartiles(xs):
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0], xs[0], xs[0]
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[0], statistics.median(xs), q[2]


def eligible(rs, phase, min_tok):
    return [r for r in rs if r["tok"][phase] >= min_tok and r["time"][phase] > 0]


def bootstrap_diff(a, b, stat, iters=2000, seed=0):
    """(point, lo, hi) of 100 * (stat(b) / stat(a) - 1), resampling each arm's requests"""
    if not a or not b:
        return None
    rnd = random.Random(seed)
    sa, sb = stat(a), stat(b)
    if not sa or sb is None:
        return None
    ds = []
    for _ in range(iters):
        ra = [rnd.choice(a) for _ in a]
        rb = [rnd.choice(b) for _ in b]
        x, y = stat(ra), stat(rb)
        if x and y is not None:
            ds.append(100.0 * (y / x - 1.0))
    ds.sort()
    if not ds:
        return None
    return 100.0 * (sb / sa - 1.0), ds[int(0.025 * len(ds))], ds[min(int(0.975 * len(ds)), len(ds) - 1)]


def summarize(rs, min_think=32, min_tool=16):
    out = {"requests": len(rs), "with_thinking": sum(1 for r in rs if r["tok"]["thinking"] > 0)}
    for ph in PHASES:
        el = eligible(rs, ph, min_think if ph == "thinking" else min_tool)
        q = quartiles([rate(r, ph) for r in el])
        c = [r["cnt"].get(ph) for r in rs if r["cnt"].get(ph)]
        rounds = sum(x["rounds"] for x in c)
        drafted = sum(x["drafted"] for x in c)
        acc = sum(x["accepted"] for x in c)
        out[ph] = {
            "n": len(el), "q": q, "pooled": pooled(rs, ph), "tokens": sum(r["tok"][ph] for r in rs),
            "accept": acc / drafted if drafted else None,
            "per_round": sum(r["tok"][ph] for r in rs) / rounds if rounds else None,
            "lookup_rounds": sum(x["lookup"] for x in c),
        }
    out["ttft"] = quartiles([r["ttft"] for r in rs])
    return out


def _f(x, fmt="{:.1f}"):
    return "-" if x is None else fmt.format(x)


def report(rs, min_think=32, min_tool=16, file=sys.stdout):
    arms = sorted({r["arm"] for r in rs})
    by = {a: [r for r in rs if r["arm"] == a] for a in arms}
    for a in arms:
        s = summarize(by[a], min_think, min_tool)
        print(f"arm {a}: {s['requests']} requests, {s['with_thinking']} with thinking, "
              f"TTFT median {_f(s['ttft'] and s['ttft'][1], '{:.2f}')} s", file=file)
        for ph in PHASES:
            x = s[ph]
            q = x["q"]
            print(f"  {ph:8s} tok/s median {_f(q and q[1])} [IQR {_f(q and q[0])}-{_f(q and q[2])}] "
                  f"(n {x['n']}), pooled {_f(x['pooled'])}, tokens {x['tokens']}, "
                  f"accept {_f(x['accept'] and 100 * x['accept'])}%, per round {_f(x['per_round'], '{:.2f}')}, "
                  f"lookup rounds {x['lookup_rounds']}", file=file)
    if "A" in by and "B" in by:
        for ph, mt in (("thinking", min_think), ("tool", min_tool)):
            a, b = eligible(by["A"], ph, mt), eligible(by["B"], ph, mt)
            med = bootstrap_diff(a, b, lambda rs, ph=ph: statistics.median([rate(r, ph) for r in rs]))
            pool = bootstrap_diff(by["A"], by["B"], lambda rs, ph=ph: pooled(rs, ph))
            fm = lambda d: "-" if d is None else f"{d[0]:+.1f}% [95% CI {d[1]:+.1f}, {d[2]:+.1f}]"
            print(f"B vs A, {ph}: median {fm(med)}, pooled {fm(pool)}", file=file)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", help="log files (default: stdin)")
    ap.add_argument("--since", help="only lines at or after this HH:MM[:SS] clock (same day logs)")
    ap.add_argument("--min-think", type=int, default=32)
    ap.add_argument("--min-tool", type=int, default=16)
    a = ap.parse_args(argv)
    lines = []
    if a.files:
        for f in a.files:
            with open(f, errors="replace") as fh:
                lines += fh.readlines()
    else:
        lines = sys.stdin.readlines()
    rs = [r for r in map(parse_line, lines) if r]
    if a.since:
        rs = [r for r in rs if r["clock"] >= a.since]
    if not rs:
        print("no [decode-stats] lines")
        return 1
    report(rs, a.min_think, a.min_tool)
    return 0


if __name__ == "__main__":
    sys.exit(main())
