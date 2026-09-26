"""Quoted tool markers stay text; reasoning_effort aliases (TABBY_TOOL_MARKERS=1, TABBY_EFFORT_ALIAS=1).

TabbyAPI 2186cdb's TagStreamParser moves everything after a `<tool_call>` to the tool channel,
wherever it appears. Measured against the served model (2026-09-26):

- The model mentions the marker in prose ("the literal string `<tool_call>` means ...") with
  tools in the request: the answer is cut at the marker and the rest is lost.
- The model documents the format inside a ```xml fence: the example goes to the tool channel,
  the parser finds no call in it, and the response says finish_reason "tool_calls" with no
  tool calls. Without tools in the request the fenced example disappears the same way.
- A stray `</tool_call>` outside a call also went to the tool channel.

The same bugs, and the fixes below, are in vLLM's qwen3 parser (blazux/qwen3.8-Flash-DGX
patches 12 and 13). For the qwen3_coder tool format, with TABBY_TOOL_MARKERS=1:

- `<tool_call>` opens a call only when `<function=` follows (after whitespace), which is the
  only thing the qwen3_coder parser can parse. Otherwise the marker and what follows stay text.
- Inside a fenced code block (CommonMark: a line of >= 3 backticks or tildes with at most three
  spaces of indent opens it; a line of the same character, at least as long, with nothing but
  whitespace after it closes it) `<tool_call>` stays text. Fence state is kept separately for
  reasoning and content and reset at each reasoning tag.
- A `</tool_call>` outside a call stays text.
- Requests without tools never enter tool mode.
- finish_reason is "tool_calls" only when a call was parsed. Tool-channel text that parses to
  no call is returned as content instead of being dropped.

Not covered (same as vLLM's fix): a fence the model opens and never closes keeps later real
calls in that channel as text.

TABBY_EFFORT_ALIAS=1: the model's chat template accepts reasoning_effort xhigh, medium and low
and raises on anything else, so a client sending "high" (Claude Code's default) got HTTP 400.
high and max map to xhigh, minimal to low.

Usage: python3 patch_tabbyapi_tool_markers.py [tabbyapi dir, default /app]
Run once at image build time; exits non-zero if an anchor is missing.
"""
import pathlib, sys
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/app")

PARSER = '''

# --- quoted tool markers stay text (patch_tabbyapi_tool_markers.py) ---
import os as _tm_os

TOOL_MARKERS = _tm_os.environ.get("TABBY_TOOL_MARKERS", "1") != "0"
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


class GuardedTagStreamParser(TagStreamParser):
    """
    TagStreamParser for formats whose calls start with `tool_start` followed by `header`
    (qwen3_coder: `<tool_call>` then `<function=`). A tool start opens a call only when the
    header follows and it is not inside a fenced code block; otherwise it stays text. A tool
    end outside a call stays text.
    """

    _MAX_HOLD_AFTER_START = 256

    def __init__(self, *args, header: str = "<function=", **kwargs):
        super().__init__(*args, **kwargs)
        self.header = header
        self._after_start = None  # text after an unconfirmed tool start, or None
        self._line = {CONTENT: "", REASONING: ""}
        self._fence = {CONTENT: None, REASONING: None}  # (char, length) of the open fence

    def feed(self, text: str) -> List[Tuple[str, str]]:
        self.saw_tag = False
        events = []
        self._pending += text
        self._pump(events)
        return _merge_events(events)

    def finish(self) -> List[Tuple[str, str]]:
        events = []
        while self._after_start is not None:
            # The stream ended before the header: the marker was text
            self._reject_start(events)
            self._pump(events)
        self._route(self._pending, events)
        self._pending = ""
        return _merge_events(events)

    def _pump(self, events: list):
        while self._pending:
            if self._after_start is not None:
                self._after_start += self._pending
                self._pending = ""
                rest = self._after_start.lstrip()
                if rest.startswith(self.header):
                    self.in_tool = True
                    events.append((TOOL, self.tool_start))
                    self._pending = self._after_start
                    self._after_start = None
                    continue
                if self.header.startswith(rest) and len(self._after_start) < self._MAX_HOLD_AFTER_START:
                    break  # could still become the header
                self._reject_start(events)
                continue

            match = self._tag_re.search(self._pending) if self._tag_re else None
            if not match:
                hold = self._partial_tag_len()
                emit_len = len(self._pending) - hold
                self._route(self._pending[:emit_len], events)
                self._pending = self._pending[emit_len:]
                break

            i, j = match.span()
            tag = match[0]
            self._route(self._pending[:i], events)
            self._pending = self._pending[j:]
            self.saw_tag = True
            tool_tags_active = not (self.in_reasoning and not self.tool_calls_in_reasoning)
            if not self.in_tool and tool_tags_active and tag == self.tool_start:
                if self._in_fence():
                    self._route(tag, events)
                else:
                    self._after_start = ""
                continue
            if not self.in_tool and tool_tags_active and tag == self.tool_end:
                self._route(tag, events)
                continue
            self._handle_tag(tag, events)

    def _reject_start(self, events: list):
        """The pending tool start was text: route it, rescan what followed it."""
        self._route(self.tool_start, events)
        self._pending = self._after_start + self._pending
        self._after_start = None

    def _handle_tag(self, tag: str, events: list):
        if not self.in_tool and tag in (self.reasoning_start, self.reasoning_end):
            self._line[REASONING] = ""
            self._fence[REASONING] = None
        super()._handle_tag(tag, events)

    def _route(self, text: str, events: list):
        if text and not self.in_tool:
            self._track(REASONING if self.in_reasoning else CONTENT, text)
        super()._route(text, events)

    def _in_fence(self) -> bool:
        return self._fence[REASONING if self.in_reasoning else CONTENT] is not None

    def _track(self, channel: str, text: str):
        """Follow fenced code blocks line by line in the channel's emitted text."""
        *lines, self._line[channel] = (self._line[channel] + text).split("\\n")
        for line in lines:
            m = _FENCE_RE.match(line)
            if not m:
                continue
            run, rest = m.group(1), m.group(2)
            fence = self._fence[channel]
            if fence is None:
                if run[0] == "~" or "`" not in rest:
                    self._fence[channel] = (run[0], len(run))
            elif run[0] == fence[0] and len(run) >= fence[1] and not rest.strip():
                self._fence[channel] = None
'''

CC_IMPORT_OLD = "from endpoints.OAI.utils.stream_parser import ("
CC_IMPORT_NEW = (
    "from endpoints.OAI.utils.stream_parser import GuardedTagStreamParser, TOOL_MARKERS  # tool markers\n"
    "from endpoints.OAI.utils.tools import canonical_format_name as _tm_canonical  # tool markers\n"
    "from endpoints.OAI.utils.stream_parser import ("
)

USE_TOOL_OLD = '''        use_tool = params.tool_choice != "none" and bool(t_tool_start)
'''
USE_TOOL_NEW = '''        use_tool = params.tool_choice != "none" and bool(t_tool_start)
        # Tool markers: no tools in the request, no tool mode
        if TOOL_MARKERS and not params.tools:
            use_tool = False
'''

CTOR_OLD = '''        parser = TagStreamParser(
            reasoning_start=mc.reasoning_start_token if use_think else None,'''
CTOR_NEW = '''        guarded = TOOL_MARKERS and _tm_canonical(tool_format) == "qwen3_coder"
        parser = (GuardedTagStreamParser if guarded else TagStreamParser)(
            reasoning_start=mc.reasoning_start_token if use_think else None,'''

STREAM_OLD = '''                generation["delta_tool_calls"] = ""
                if finish_reason and full_tool:
                    generation["delta_tool_calls"] = _parse_tool_calls(
                        full_tool, tool_format, label, tools=params.tools
                    )
                    generation["finish_reason"] = "tool_calls"
'''
STREAM_NEW = '''                generation["delta_tool_calls"] = ""
                if finish_reason and full_tool:
                    generation["delta_tool_calls"] = _parse_tool_calls(
                        full_tool, tool_format, label, tools=params.tools
                    )
                    if generation["delta_tool_calls"] or not TOOL_MARKERS:
                        generation["finish_reason"] = "tool_calls"
                    else:
                        # Tool markers: nothing parsed, so it was text
                        xlogger.warning(f"{label}: tool-channel text parsed to no call; returned as content")
                        generation["delta_content"] += full_tool
                        full_content += full_tool
                        generation["delta_tool_calls"] = ""
'''

NONSTREAM_OLD = '''            generation["tool_calls"] = _parse_tool_calls(
                full_tool, tool_format, label, tools=params.tools
            )
            if full_tool:
                generation["finish_reason"] = "tool_calls"
            return generation
'''
NONSTREAM_NEW = '''            generation["tool_calls"] = _parse_tool_calls(
                full_tool, tool_format, label, tools=params.tools
            )
            if full_tool and (generation["tool_calls"] or not TOOL_MARKERS):
                generation["finish_reason"] = "tool_calls"
            elif full_tool:
                # Tool markers: nothing parsed, so it was text
                xlogger.warning(f"{label}: tool-channel text parsed to no call; returned as content")
                generation["content"] = (full_content + full_tool) or None
            return generation
'''

EFFORT_OLD = '''    request_vars.update(data.template_vars)

    return {
        **container.template_vars_default,
        **request_vars,
        **container.template_vars_force,
    }
'''
EFFORT_NEW = '''    request_vars.update(data.template_vars)

    merged = {
        **container.template_vars_default,
        **request_vars,
        **container.template_vars_force,
    }
    # Effort alias (patch_tabbyapi_tool_markers.py): the template takes xhigh, medium, low
    if _tm_os.environ.get("TABBY_EFFORT_ALIAS", "1") != "0":
        effort = merged.get("reasoning_effort")
        if isinstance(effort, str) and effort.lower() in _EFFORT_ALIAS:
            merged["reasoning_effort"] = _EFFORT_ALIAS[effort.lower()]
    return merged


import os as _tm_os
_EFFORT_ALIAS = {"high": "xhigh", "max": "xhigh", "minimal": "low"}
'''


def patch(path, pairs):
    text = path.read_text()
    for old, new in pairs:
        if text.count(old) != 1:
            sys.exit(f"patch_tabbyapi_tool_markers: anchor not found once in {path}:\n{old}")
        text = text.replace(old, new)
    path.write_text(text)


sp = root / "endpoints/OAI/utils/stream_parser.py"
text = sp.read_text()
if "class TagStreamParser" not in text or "def _merge_events" not in text:
    sys.exit(f"patch_tabbyapi_tool_markers: TagStreamParser not found in {sp}")
sp.write_text(text + PARSER)

patch(root / "endpoints/OAI/utils/chat_completion.py", [
    (CC_IMPORT_OLD, CC_IMPORT_NEW),
    (USE_TOOL_OLD, USE_TOOL_NEW),
    (CTOR_OLD, CTOR_NEW),
    (STREAM_OLD, STREAM_NEW),
    (NONSTREAM_OLD, NONSTREAM_NEW),
    (EFFORT_OLD, EFFORT_NEW),
])
print("patch_tabbyapi_tool_markers: applied")
