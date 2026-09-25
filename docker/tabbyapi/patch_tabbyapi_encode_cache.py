"""Tokenize only the new part of a long chat prompt.

Every chat turn re-renders and re-tokenizes the whole conversation; at ~115k tokens the
tokenizer alone takes ~0.16 s of each follow-up's time to first token (TabbyAPI's own
"first token" figure starts after it, at job enqueue). This keeps the token ids of the
last few long prompts and, for a new prompt that shares a prefix with one of them,
reuses the ids up to the last "<|im_start|>" inside the shared prefix and encodes only the
rest. With special tokens encoded (as TabbyAPI does for prompts) the tokenizer splits the
text at special tokens before anything else, so the pieces on either side of one encode
independently and the result equals encoding the whole prompt. Anything unusual (images,
BOS, a marker that is not one token, short prompts) takes the plain path.

TABBY_ENCODE_CACHE=0 at run time disables it; TABBY_ENCODE_CACHE=verify also encodes the
whole prompt and logs any mismatch (and returns the full encoding).

Usage: python3 patch_tabbyapi_encode_cache.py [TabbyAPI dir, default /app]
Run once at image build time; exits non-zero if the anchor is missing.
"""
import pathlib, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/app")

HELPER = '''"""Prefix-reusing prompt tokenization (patch_tabbyapi_encode_cache.py)."""
import os
import re

import torch

MARKER = "<|im_start|>"
MIN_CHARS = 16384      # shorter prompts tokenize in a few ms; not worth the bookkeeping
MAX_ENTRIES = 8
_entries = []          # most recent first: (text, ids (1, n) tensor, [(char_pos, tok_pos)])
_marker_id = {}


def _common_prefix_len(a: str, b: str) -> int:
    lo, hi = 0, min(len(a), len(b))
    if a[:hi] == b[:hi]:
        return hi
    while lo < hi:  # largest m with a[:m] == b[:m]; slice compares run in C
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _markers(tokenizer, text: str, ids: torch.Tensor):
    mid = _marker_id.get(id(tokenizer))
    if mid is None:
        return None
    chars = [m.start() for m in re.finditer(re.escape(MARKER), text)]
    toks = (ids[0] == mid).nonzero().flatten().tolist()
    return list(zip(chars, toks)) if len(chars) == len(toks) else None


def encode_prompt(tokenizer, prompt: str, add_bos: bool, embeddings) -> torch.Tensor:
    def full():
        return tokenizer.encode(prompt, add_bos = add_bos, encode_special_tokens = True,
                                embeddings = embeddings)

    mode = os.environ.get("TABBY_ENCODE_CACHE", "1")
    if mode == "0" or embeddings or add_bos or len(prompt) < MIN_CHARS:
        return full()
    if id(tokenizer) not in _marker_id:
        m = tokenizer.encode(MARKER, encode_special_tokens = True)
        _marker_id[id(tokenizer)] = int(m[0, 0]) if m.shape[-1] == 1 else None

    best = None  # (char_pos, tok_pos, ids)
    for text, ids, marks in _entries:
        n = _common_prefix_len(text, prompt)
        for c, t in reversed(marks):
            if c + len(MARKER) <= n:
                if best is None or c > best[0]:
                    best = (c, t, ids)
                break
    if best is None:
        ids = full()
    else:
        c, t, ids0 = best
        rest = tokenizer.encode(prompt[c:], add_bos = False, encode_special_tokens = True)
        ids = torch.cat((ids0[:, :t], rest), dim = -1)
        if mode == "verify":
            ref = full()
            if not torch.equal(ref, ids):
                print(f"TABBY_ENCODE_CACHE mismatch: {ref.shape[-1]} vs {ids.shape[-1]} tokens", flush = True)
                ids = ref

    marks = _markers(tokenizer, prompt, ids)
    if marks:
        _entries.insert(0, (prompt, ids, marks))
        del _entries[MAX_ENTRIES:]
    return ids
'''

rel = "backends/exllamav3/model.py"
old = """        input_ids = [
            self.tokenizer.encode(
                prompt,
                add_bos=add_bos_token,
                encode_special_tokens=True,
                embeddings=mm_embeddings_content,
            )
            for prompt in prompts
        ]
"""
new = """        # patch_tabbyapi_encode_cache.py: reuse the ids of a shared conversation prefix
        from common.encode_cache import encode_prompt

        input_ids = [
            encode_prompt(self.tokenizer, prompt, add_bos_token, mm_embeddings_content)
            for prompt in prompts
        ]
"""
p = root / rel
s = p.read_text()
if s.count(old) != 1:
    sys.exit(f"patch_tabbyapi_encode_cache: anchor not found exactly once in {rel}")
p.write_text(s.replace(old, new))
(root / "common" / "encode_cache.py").write_text(HELPER)
print("patched", rel, "+ common/encode_cache.py")
