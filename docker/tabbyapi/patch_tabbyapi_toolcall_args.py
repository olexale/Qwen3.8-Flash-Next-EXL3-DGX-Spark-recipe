"""Tool-call arguments as the model wrote them (TABBY_TOOLCALL_ARGS=1, on in the image).

The model writes tool calls in the qwen3_coder format:

    <tool_call>
    <function=write>
    <parameter=content>
    ...file text...
    </parameter>
    </function>
    </tool_call>

TabbyAPI 2186cdb's parser (endpoints/OAI/utils/toolcall_formats/qwen3_coder.py) turns each
value into a JSON argument with `.strip()` and then `json.loads` whenever the text parses,
ignoring the tool's schema. So a `write` of package.json reached the client as an object instead
of the file's text, every written file lost its final newline, an `edit` whose oldText started
with indentation lost it (matching a shorter string), and a file containing `42` or `true`
became a number or a boolean. The chat template also renders those values back differently on
the next turn, so the prompt no longer matched what the model generated (a prefix-cache miss
inside the tool call).

With the patch, a value loses exactly the one newline the template puts after the opening tag
and the one before the closing tag. It is converted with json.loads only when the request's
tool schema declares that parameter with a non-string type (integer, number, boolean, object,
array, null), and stays text when the schema allows a string, has no type, or does not list the
parameter. That is what vLLM's parser for this format does. Round trip: parse, then render with
the model's template, gives back the model's text (tests/test_toolcall_args.py).

TABBY_TOOLCALL_ARGS=0 restores the old parsing.

Usage: python3 patch_tabbyapi_toolcall_args.py [tabbyapi dir, default /app]
Run once at image build time; exits non-zero if an anchor is missing.
"""
import pathlib, sys
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/app")

HELPERS = '''

# --- tool-call arguments as written (patch_tabbyapi_toolcall_args.py) ---
import os as _ta_os
_ARGS_FIX = _ta_os.environ.get("TABBY_TOOLCALL_ARGS", "1") != "0"


def _param_schema(tools, func_name, key):
    """The JSON schema of parameter `key` of tool `func_name` in the request, or None."""
    for spec in tools or []:
        fn = getattr(spec, "function", None)
        if fn is None and isinstance(spec, dict):
            fn = spec.get("function")
        if fn is None:
            continue
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        if name != func_name:
            continue
        params = fn.get("parameters") if isinstance(fn, dict) else getattr(fn, "parameters", None)
        props = (params or {}).get("properties") or {}
        s = props.get(key)
        return s if isinstance(s, dict) else None
    return None


def _schema_types(schema):
    types = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    for k in ("anyOf", "oneOf"):
        for sub in schema.get(k) or []:
            if isinstance(sub, dict):
                types |= _schema_types(sub)
    return types


def _typed_value(raw, schema):
    """A parameter value as the model wrote it, typed by the tool's schema."""
    v = raw
    if v.startswith("\\n"):
        v = v[1:]
    if v.endswith("\\n"):
        v = v[:-1]
    types = _schema_types(schema) if schema else set()
    if not types or "string" in types:
        return v
    s = v.strip()
    if "boolean" in types and s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return v
'''

edits = {
    "endpoints/OAI/utils/toolcall_formats/qwen3_coder.py": [
        ("def parse_toolcalls(text: str) -> list[ToolCall]:\n",
         "def parse_toolcalls(text: str, tools=None) -> list[ToolCall]:\n"),
        ("                key = pm.group(1).strip()\n"
         "                val = pm.group(2).strip()\n"
         "                val = coerce_param_value(val)\n",
         "                key = pm.group(1).strip()\n"
         "                if _ARGS_FIX:\n"
         "                    val = _typed_value(pm.group(2), _param_schema(tools, func_name, key))\n"
         "                else:\n"
         "                    val = pm.group(2).strip()\n"
         "                    val = coerce_param_value(val)\n"),
    ],
    "endpoints/OAI/utils/tools.py": [
        ("def parse_toolcalls(tool_calls_str: str, tool_format: str) -> List[ToolCall]:\n",
         "def parse_toolcalls(tool_calls_str: str, tool_format: str, tools=None) -> List[ToolCall]:\n"),
        ("        return parser.parse_toolcalls(tool_calls_str)\n",
         "        # Parsers that type arguments by the request's tool schemas take them\n"
         "        # (patch_tabbyapi_toolcall_args.py)\n"
         "        import inspect\n"
         "        if \"tools\" in inspect.signature(parser.parse_toolcalls).parameters:\n"
         "            return parser.parse_toolcalls(tool_calls_str, tools=tools)\n"
         "        return parser.parse_toolcalls(tool_calls_str)\n"),
    ],
    "endpoints/OAI/utils/chat_completion.py": [
        ("def _parse_tool_calls(\n    text: str,\n    tool_format: str,\n    label: str,\n) -> list:\n",
         "def _parse_tool_calls(\n    text: str,\n    tool_format: str,\n    label: str,\n    tools=None,\n) -> list:\n"),
        ("    parsed = parse_toolcalls(text, tool_format)\n",
         "    parsed = parse_toolcalls(text, tool_format, tools=tools)\n"),
        ("                    generation[\"delta_tool_calls\"] = _parse_tool_calls(\n"
         "                        full_tool, tool_format, label\n"
         "                    )\n",
         "                    generation[\"delta_tool_calls\"] = _parse_tool_calls(\n"
         "                        full_tool, tool_format, label, tools=params.tools\n"
         "                    )\n"),
        ("            generation[\"tool_calls\"] = _parse_tool_calls(full_tool, tool_format, label)\n",
         "            generation[\"tool_calls\"] = _parse_tool_calls(\n"
         "                full_tool, tool_format, label, tools=params.tools\n"
         "            )\n"),
    ],
}
for rel, reps in edits.items():
    p = root / rel
    s = p.read_text()
    for old, new in reps:
        if s.count(old) != 1:
            sys.exit(f"patch_tabbyapi_toolcall_args: anchor not found exactly once in {rel}: {old[:60]!r}")
        s = s.replace(old, new)
    if rel.endswith("qwen3_coder.py"):
        s += HELPERS
    p.write_text(s)
    print("patched", rel)
