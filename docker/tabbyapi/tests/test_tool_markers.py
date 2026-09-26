"""patch_tabbyapi_tool_markers.py: `<tool_call>` opens a call only when `<function=` follows and
it is outside a fenced code block; quoted markers stay text; reasoning_effort aliases."""
import sys, types
sys.path.insert(0, "/app")
from endpoints.OAI.utils.stream_parser import GuardedTagStreamParser, TagStreamParser

CALL = "<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hosts\n</parameter>\n</function>\n</tool_call>"


def parser(cls=GuardedTagStreamParser, **kw):
    return cls(reasoning_start="<think>", reasoning_end="</think>",
               tool_start="<tool_call>", tool_end="</tool_call>", **kw)


def run(text, chunked, cls=GuardedTagStreamParser, **kw):
    p = parser(cls, **kw)
    out = {"reasoning": "", "content": "", "tool": ""}
    for chunk in (list(text) if chunked else [text]):
        for ch, t in p.feed(chunk):
            out[ch] += t
    for ch, t in p.finish():
        out[ch] += t
    return out


def both(text, **kw):
    """Whole text and one character at a time must agree."""
    a, b = run(text, False, **kw), run(text, True, **kw)
    assert a == b, (a, b)
    return a


def test_real_call_after_text():
    o = both("Reading it.\n\n" + CALL)
    assert o["content"] == "Reading it.\n\n" and o["tool"] == CALL, o


def test_real_call_in_reasoning():
    o = both("I should read it.\n" + CALL, start_in_reasoning=True)
    assert o["tool"] == CALL and o["reasoning"] == "I should read it.\n", o


def test_marker_in_prose_stays_text():
    t = "The literal string `<tool_call>` opens a call. DONE."
    o = both(t)
    assert o["content"] == t and o["tool"] == "", o


def test_marker_then_whitespace_then_prose():
    t = "Write <tool_call>\n  and then the function header."
    assert both(t)["content"] == t


def test_call_inside_backtick_fence_stays_text():
    t = "Like this:\n```xml\n" + CALL + "\n```\nThat is all."
    o = both(t)
    assert o["content"] == t and o["tool"] == "", o


def test_call_inside_tilde_fence_stays_text():
    t = "~~~\n" + CALL + "\n~~~\n"
    assert both(t)["content"] == t


def test_call_after_closed_fence_is_a_call():
    pre = "```xml\n<tool_call>\n<function=x>\n</function>\n</tool_call>\n```\nNow for real:\n"
    o = both(pre + CALL)
    assert o["content"] == pre and o["tool"] == CALL, o


def test_nested_fence_does_not_close_outer():
    # ```` opens; the inner ``` lines neither close it nor open anything
    t = "````md\n```xml\n" + CALL + "\n```\n" + CALL + "\n````\n"
    o = both(t)
    assert o["content"] == t and o["tool"] == "", o


def test_indented_four_spaces_is_not_a_fence():
    pre = "    ```\n"
    o = both(pre + CALL)
    assert o["tool"] == CALL, o


def test_closing_fence_needs_same_char_and_length():
    t = "```\n~~~\n" + CALL + "\n```\n"
    assert both(t)["tool"] == ""


def test_fence_open_in_reasoning_does_not_carry_into_content():
    o = both("Example:\n```xml\n</think>\n\nCalling now.\n" + CALL, start_in_reasoning=True)
    assert o["tool"] == CALL, o
    assert o["reasoning"] == "Example:\n```xml\n", o


def test_fence_in_reasoning_guards_reasoning_marker():
    t = "```\n" + CALL + "\n```\n"
    o = both(t + "</think>\n\nok", start_in_reasoning=True)
    assert o["reasoning"] == t and o["tool"] == "" and o["content"] == "\n\nok", o


def test_stray_tool_end_stays_text():
    t = "Close it with </tool_call> when done."
    o = both(t)
    assert o["content"] == t and o["tool"] == "", o


def test_stream_ends_after_marker():
    o = both("Answer: <tool_call>")
    assert o["content"] == "Answer: <tool_call>" and o["tool"] == "", o


def test_stream_ends_inside_header_prefix():
    o = both("x <tool_call>\n<func")
    assert o["content"] == "x <tool_call>\n<func", o


def test_two_calls():
    o = both(CALL + "\n" + CALL)
    # the newline between calls is content, as in the stock parser
    assert o["tool"] == CALL + CALL and o["content"] == "\n", o
    assert run(CALL + "\n" + CALL, False, cls=TagStreamParser) == o


def test_rejected_marker_then_real_call():
    pre = "The `<tool_call>` tag starts a call:\n"
    o = both(pre + CALL)
    assert o["content"] == pre and o["tool"] == CALL, o


def test_tool_calls_in_reasoning_off_keeps_text():
    o = both("r " + CALL, start_in_reasoning=True, tool_calls_in_reasoning=False)
    assert o["reasoning"] == "r " + CALL and o["tool"] == "", o


def test_valid_traffic_same_as_stock():
    """Where the stock parser was right, the guarded one gives the same channels."""
    cases = [
        ("Let me check.\n" + CALL, {}),
        ("plan it</think>\n\nOK.\n" + CALL, {"start_in_reasoning": True}),
        ("think " + CALL + " more</think>\n\nanswer", {"start_in_reasoning": True}),
        ("<think>\nhm\n</think>\n\n" + CALL + "\n" + CALL, {}),
        ("Just an answer with ``` a fence\n```\ncode\n```\nend", {}),
    ]
    for text, kw in cases:
        for chunked in (False, True):
            assert run(text, chunked, **kw) == run(text, chunked, cls=TagStreamParser, **kw), text


def test_unpatched_parser_still_there():
    o = run("`<tool_call>` x", False, cls=TagStreamParser)
    assert o["tool"] == "<tool_call>` x", o  # the stock behaviour this patch fixes


def test_upstream_parser_tests_pass():
    """TabbyAPI's own TagStreamParser tests still pass (the stock class is unchanged)."""
    import unittest
    sys.path.insert(0, "/app/tests")
    import test_stream_parser
    r = unittest.TextTestRunner(stream=open("/dev/null", "w")).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_stream_parser))
    assert r.wasSuccessful(), (r.failures + r.errors)[:2]


def _vars(effort):
    from endpoints.OAI.types.chat_completion import ChatCompletionRequest
    from endpoints.OAI.utils.chat_completion import resolve_template_vars
    c = types.SimpleNamespace(template_vars_default={}, template_vars_force={})
    data = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], reasoning_effort=effort)
    return resolve_template_vars(data, c).get("reasoning_effort")


def test_effort_alias():
    assert _vars("high") == "xhigh"
    assert _vars("max") == "xhigh"
    assert _vars("minimal") == "low"
    assert _vars("medium") == "medium"
    assert _vars("xhigh") == "xhigh"
    assert _vars(None) is None
