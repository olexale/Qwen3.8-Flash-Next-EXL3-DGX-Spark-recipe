"""patch_exllamav3_decode_stats.py: phase split, round accounting, the log line, and
tools/decode_report.py's parsing of it."""
import contextlib, io, sys, types
from exllamav3.generator import job as jm

sys.path.insert(0, "/tools")
import decode_report as dr

# Token ids: 1 <think>, 2 </think>, 3 <tool_call>, 10+ other
MARKS = {"<think>": 1, "</think>": 2, "<tool_call>": 3}


def _run(ds, rounds):
    """rounds: [(token ids, drafted, lookup, now)]"""
    for toks, drafted, lookup, now in rounds:
        for t in toks:
            jm._ds_token(ds, t, MARKS)
        jm._ds_round(ds, len(toks), drafted, lookup, now)


def test_think_on_detection():
    assert jm._ds_new([10, 11, 1, 12], MARKS, 0.0)["think_on"]           # generation prompt opens <think>
    assert not jm._ds_new([10, 1, 12, 2, 12], MARKS, 0.0)["think_on"]    # empty think block: thinking off
    assert not jm._ds_new([10, 11, 12], MARKS, 0.0)["think_on"]          # no think markers


def test_phases_and_time_split():
    ds = jm._ds_new([10, 1], MARKS, 100.0)
    _run(ds, [
        ([20, 21, 22, 23], 5, False, 101.0),     # thinking, 3 of 5 drafts accepted
        ([24, 2, 30, 31], 5, False, 102.0),      # 2 thinking (incl. </think>), 2 text
        ([32, 3, 40, 41, 42], 7, True, 103.0),   # 1 text, 4 tool (a lookup round)
        ([43], 0, False, 103.5),                 # no-draft round
    ])
    assert ds["think_end"] == 6 and ds["tool_start"] == 10, ds
    assert ds["tok"] == {"thinking": 6, "text": 3, "tool": 5}, ds["tok"]
    t = ds["time"]
    assert abs(t["thinking"] - 1.5) < 1e-9 and abs(t["text"] - 0.7) < 1e-9 and abs(t["tool"] - 1.3) < 1e-9, t
    # Round counters go to the phase of the round's first token
    assert ds["cnt"]["thinking"] == [2, 10, 6, 0, 0, 0], ds["cnt"]
    assert ds["cnt"]["text"] == [1, 7, 4, 1, 7, 4], ds["cnt"]
    assert ds["cnt"]["tool"] == [1, 0, 0, 0, 0, 0], ds["cnt"]


def test_thinking_off_and_tool_marker_inside_thinking():
    ds = jm._ds_new([10], MARKS, 0.0)
    _run(ds, [([3, 20, 21], 2, False, 1.0)])
    assert ds["tok"] == {"thinking": 0, "text": 0, "tool": 3}, ds["tok"]
    # A <tool_call> inside thinking does not start the tool phase
    ds = jm._ds_new([1], MARKS, 0.0)
    _run(ds, [([20, 3, 21, 2, 30], 4, False, 1.0)])
    assert ds["tool_start"] is None and ds["tok"] == {"thinking": 4, "text": 1, "tool": 0}, ds


def test_zero_token_round_only_moves_clock():
    ds = jm._ds_new([1], MARKS, 0.0)
    jm._ds_round(ds, 0, 5, False, 2.0)
    assert ds["t"] == 2.0 and ds["cnt"]["thinking"][0] == 0


def test_line_round_trip():
    ds = jm._ds_new([10, 1], MARKS, 100.0)
    _run(ds, [([20, 21, 22, 23], 5, False, 101.0), ([24, 2, 30, 31], 5, False, 102.0),
              ([32, 3, 40, 41, 42], 7, True, 103.0)])
    ds["extra"] = {"spec rounds": 3}
    line = jm._ds_line(ds, "B", 5403, 0.42, "14:02:03")
    assert "[decode-stats] at 14:02:03 arm B | prompt 5403, ttft 0.42 s | tokens 13 (thinking 6, text 3, tool 4)" in line, line
    r = dr.parse_line("14:02:03.100 INFO:     " + line + "\n")
    assert r["arm"] == "B" and r["prompt"] == 5403 and r["tokens"] == 13, r
    assert r["tok"] == {"thinking": 6, "text": 3, "tool": 4}, r
    assert abs(r["time"]["thinking"] - 1.5) < 1e-3 and abs(r["time"]["tool"] - 0.8) < 1e-3, r
    assert r["cnt"]["thinking"] == {"rounds": 2, "drafted": 10, "accepted": 6, "lookup": 0,
                                    "lookup_drafted": 0, "lookup_accepted": 0}, r["cnt"]
    assert r["cnt"]["text"]["lookup"] == 1 and r.get("extra") == "spec rounds 3", r
    assert dr.parse_line("[prefix-diag] prompt 90 | kv-matched 0") is None
    assert dr.parse_line("[decode-stats] error: KeyError('x')") is None


def test_emit_via_job():
    tok = types.SimpleNamespace(single_id=lambda s: MARKS[s])
    ids = [10, 11, 1]

    class _Ids:
        def __len__(self): return len(ids)
        def torch_slice(self, a, b):
            import torch
            return torch.tensor([ids[a:b]])

    job = types.SimpleNamespace(sequences=[types.SimpleNamespace(sequence_ids=_Ids())],
                                generator=types.SimpleNamespace(tokenizer=tok),
                                time_first_token=50.0, time_enqueue=49.5)
    jm._ds_ids = None
    for t in (20, 21, 2, 30):
        jm._ds_on_token(job, t)
    jm._ds_round(job._ds, 4, 3, False, 51.0)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        jm._ds_emit(job)
        jm._ds_emit(job)   # emitted once
    lines = out.getvalue().splitlines()
    assert len(lines) == 1 and "arm - | prompt 3, ttft 0.50 s | tokens 4 (thinking 3, text 1, tool 0)" in lines[0], lines


def _rec(arm, think_tok, think_s, tool_tok=0, tool_s=0.0):
    return {"clock": "10:00:00", "arm": arm, "prompt": 1000, "ttft": 0.5, "tokens": think_tok + tool_tok,
            "tok": {"thinking": think_tok, "text": 0, "tool": tool_tok},
            "time": {"thinking": think_s, "text": 0.0, "tool": tool_s},
            "cnt": {"thinking": {"rounds": 10, "drafted": 40, "accepted": 30, "lookup": 0,
                                 "lookup_drafted": 0, "lookup_accepted": 0}}}


def test_report_bootstrap():
    a = [_rec("A", 300, 300 / (50 + i % 5), 100, 1.0) for i in range(30)]
    b = [_rec("B", 300, 300 / (60 + i % 5), 100, 1.0) for i in range(30)]
    d = dr.bootstrap_diff(dr.eligible(a, "thinking", 32), dr.eligible(b, "thinking", 32),
                          lambda rs: dr.statistics.median([dr.rate(r, "thinking") for r in rs]))
    assert abs(d[0] - 100 * (62 / 52 - 1)) < 0.01 and d[1] > 10 and d[2] < 30, d
    s = dr.summarize(a)
    assert s["thinking"]["n"] == 30 and abs(s["thinking"]["accept"] - 0.75) < 1e-9, s
    out = io.StringIO()
    dr.report(a + b, file=out)
    assert "B vs A, thinking: median +19.2%" in out.getvalue(), out.getvalue()
