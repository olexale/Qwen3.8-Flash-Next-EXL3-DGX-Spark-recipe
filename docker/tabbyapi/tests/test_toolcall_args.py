"""patch_tabbyapi_toolcall_args.py: tool-call arguments reach the client as the model wrote
them, typed only where the tool's schema asks for it, and render back to the same text."""
import json, os, pathlib, sys
sys.path.insert(0, "/app")
from endpoints.OAI.utils.tools import parse_toolcalls

# pi's tools, as pi declares them (strings, plus numeric read offsets and a bash timeout)
def _fn(name, **props):
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {
        "type": "object", "properties": props, "required": list(props)}}}
S, I, N = {"type": "string"}, {"type": "integer"}, {"type": "number"}
TOOLS = [_fn("read", path=S, offset=I, limit=I), _fn("write", path=S, content=S),
         _fn("edit", path=S, oldText=S, newText=S), _fn("bash", command=S, timeout=N),
         _fn("flags", enabled={"type": "boolean"}, opts={"type": "object"},
             maybe={"type": ["string", "null"]}, any_of={"anyOf": [I, {"type": "null"}]})]

def raw_call(name, **params):
    """A tool call exactly as the model writes it (and as the chat template renders it)."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"

def args_of(raw, tools=TOOLS):
    calls = parse_toolcalls(raw, "qwen3_coder", tools=tools)
    assert len(calls) == 1, calls
    return json.loads(calls[0].function.arguments)

PKG = '{\n  "name": "demo",\n  "version": "1.0.0"\n}\n'
PY = "def f():\n    return 1\n"

def test_write_json_file_stays_text():
    a = args_of(raw_call("write", path="package.json", content=PKG))
    assert a["content"] == PKG, repr(a["content"])

def test_write_keeps_final_newline():
    assert args_of(raw_call("write", path="a.py", content=PY))["content"] == PY

def test_write_keeps_leading_blank_line_and_trailing_blank_lines():
    c = "\n\nx = 1\n\n\n"
    assert args_of(raw_call("write", path="a.py", content=c))["content"] == c

def test_edit_keeps_indentation_of_first_line():
    a = args_of(raw_call("edit", path="a.py", oldText="    return 1\n", newText="        return 2"))
    assert a["oldText"] == "    return 1\n" and a["newText"] == "        return 2", a

def test_string_params_that_look_like_json_stay_strings():
    for c in ("42", "true", "null", "[1, 2]", "3.5"):
        v = args_of(raw_call("write", path="f", content=c))["content"]
        assert v == c and isinstance(v, str), (c, v)

def test_empty_string():
    assert args_of(raw_call("write", path="f", content=""))["content"] == ""

def test_typed_params_are_converted():
    a = args_of(raw_call("read", path="a.py", offset="10", limit=" 50 "))
    assert a == {"path": "a.py", "offset": 10, "limit": 50}, a
    assert args_of(raw_call("bash", command="ls", timeout="2.5"))["timeout"] == 2.5
    f = args_of(raw_call("flags", enabled="True", opts='{"a": 1}', maybe="null", any_of="7"))
    assert f == {"enabled": True, "opts": {"a": 1}, "maybe": "null", "any_of": 7}, f

def test_unparsable_typed_value_falls_back_to_text():
    assert args_of(raw_call("read", path="a", offset="ten"))["offset"] == "ten"

def test_unknown_param_or_tool_or_no_tools_stays_text():
    assert args_of(raw_call("write", path="f", content="1", extra="2"))["extra"] == "2"
    assert args_of(raw_call("mystery", n="5"))["n"] == "5"
    assert args_of(raw_call("read", path="a", limit="5"), tools=None)["limit"] == "5"

def test_value_without_template_newlines():
    raw = "<tool_call>\n<function=read>\n<parameter=path>a.py</parameter>\n</function>\n</tool_call>"
    assert args_of(raw)["path"] == "a.py"

def test_two_calls_in_one_answer():
    raw = raw_call("read", path="a.py") + "\n" + raw_call("write", path="b.py", content=PY)
    calls = parse_toolcalls(raw, "qwen3_coder", tools=TOOLS)
    assert [c.function.name for c in calls] == ["read", "write"]
    assert json.loads(calls[1].function.arguments)["content"] == PY

async def test_round_trip_through_the_chat_template():
    """Parse the model's text, render the assistant turn back with the model's template via
    TabbyAPI's renderer: the tool call must come back byte for byte (else the next turn's
    prompt no longer matches what the model generated)."""
    tpl_path = pathlib.Path(os.environ.get("TEMPLATE", "/tests_data/chat_template.jinja"))
    if not tpl_path.exists():
        raise AssertionError(f"chat template not mounted at {tpl_path}")
    from common.templating import PromptTemplate
    tpl = await PromptTemplate.from_file(tpl_path)
    cases = [raw_call("write", path="package.json", content=PKG),
             raw_call("write", path="a.py", content="\n\nx = 1\n\n"),
             raw_call("edit", path="a.py", oldText="    return 1\n", newText="    return 2"),
             raw_call("write", path="VERSION", content="42"),
             raw_call("read", path="a.py", offset="10", limit="50")]
    for raw in cases:
        call = parse_toolcalls(raw, "qwen3_coder", tools=TOOLS)[0]
        msgs = [{"role": "user", "content": "do it"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                 "function": {"name": call.function.name, "arguments": json.loads(call.function.arguments)}}]}]
        out = await tpl.render({"messages": msgs, "tools": TOOLS, "add_generation_prompt": False})
        # the system prompt carries an example <tool_call>; take the one in the assistant turn
        a = out.index("<tool_call>", out.rindex("<|im_start|>assistant"))
        got = out[a:out.index("</tool_call>", a) + len("</tool_call>")]
        assert got == raw, f"\nmodel wrote: {raw!r}\nrendered:    {got!r}"
