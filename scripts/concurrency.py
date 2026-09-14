#!/usr/bin/env python3
"""Concurrency with a chosen prompt size and generation length. Two aggregates are reported:
'window' (total tokens over the earliest first token to the latest last token, the recipe's
definition, which charges other streams' prefills to the first stream) and 'steady' (total
tokens over the window from the LAST stream's first token to the FIRST stream's last token,
where every stream is decoding)."""
import json, sys, threading, time, urllib.request
U = "http://127.0.0.1:8899"
FILLER = ("The maintenance log for reactor bay seven records a pressure excursion at "
          "oh four hundred hours, followed by a manual override and a return to nominal. ")
_s = [int(time.time()) % 100000]
def prompt(ntok):
    _s[0] += 1
    return "Record %d. " % _s[0] + FILLER * max(1, int(ntok * 4 / len(FILLER))) + "\n\nSummarize in one sentence."
def one(p, maxgen, out):
    body = json.dumps({"model": "Qwen3.8-Flash-Next", "prompt": p, "max_tokens": maxgen, "temperature": 0,
                       "seed": 0, "stream": True, "stream_options": {"include_usage": True}}).encode()
    r = urllib.request.Request(U + "/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); tf = None; usage = None; n = 0
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "): continue
            c = raw[6:].strip()
            if c == b"[DONE]": break
            d = json.loads(c)
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text") and tf is None: tf = time.perf_counter()
    tl = time.perf_counter()
    out.append({"gen": usage["completion_tokens"], "t0": t0, "tf": tf, "tl": tl})
# usage: concurrency.py <tag> <streams, e.g. 1,2,4,8> <prompt tokens> <new tokens> [rounds]
tag, Ns, ctx, maxgen = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]), int(sys.argv[4])
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 2
res = {}
for N in Ns:
    rows = []
    for r in range(rounds):
        out, th = [], []
        for i in range(N):
            t = threading.Thread(target=one, args=(prompt(ctx), maxgen, out)); t.start(); th.append(t)
        for t in th: t.join()
        tot = sum(o["gen"] for o in out)
        window = max(o["tl"] for o in out) - min(o["tf"] for o in out)
        steady_w = min(o["tl"] for o in out) - max(o["tf"] for o in out)
        # tokens emitted inside the steady window, assuming a constant rate per stream
        steady_tok = sum(o["gen"] * max(0.0, steady_w) / (o["tl"] - o["tf"]) for o in out)
        per = [round((o["gen"] - 1) / (o["tl"] - o["tf"]), 2) for o in out]
        rows.append({"window": round(tot / window, 2), "steady": round(steady_tok / steady_w, 2) if steady_w > 0 else None,
                     "steady_s": round(steady_w, 2), "per_stream": per})
        print(f"[{tag} ctx={ctx} gen={maxgen} N={N} r={r}] window={rows[-1]['window']} steady={rows[-1]['steady']} "
              f"(steady window {rows[-1]['steady_s']}s) per_stream={per}", flush=True)
    res[N] = rows
    w = sorted(x["window"] for x in rows); s = sorted((x["steady"] or 0) for x in rows)
    print(f"SUMMARY {tag} ctx={ctx} gen={maxgen} N={N}: window_p50={w[len(w)//2]} steady_p50={s[len(s)//2]}", flush=True)
json.dump(res, open(f"concurrency.{tag}.json", "w"), indent=1)
