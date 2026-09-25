#!/usr/bin/env python3
"""patch_tabbyapi_encode_cache.py: cached vs full prompt tokenization on growing chats.

Renders conversations with the model's chat template turn by turn (varied content: prose,
code, CJK, emoji, whitespace runs, text that looks like markup) and checks every turn's
cached encoding against encoding the whole prompt. No GPU, no model load; runs in the image:

  docker run --rm -v $MODEL:/m:ro -v $MODEL/config.json.native:/m/config.json:ro \\
    -v $PWD/docker/tabbyapi:/t:ro --entrypoint sh qwen38-exl3-tabby:latest -c \\
    'mkdir -p /tmp/app/backends/exllamav3 /tmp/app/common && cp /app/backends/exllamav3/model.py \\
     /tmp/app/backends/exllamav3/ && python3 /t/patch_tabbyapi_encode_cache.py /tmp/app && \\
     PYTHONPATH=/tmp/app python3 /t/tools/encode_cache_test.py'
Env: CONVS=12 TURNS=8
"""
import os, random, time
import jinja2, torch
from exllamav3 import Config, Tokenizer
import common.encode_cache as EC

M = "/m"
tok = Tokenizer.from_config(Config.from_directory(M))
tpl = jinja2.Environment().from_string(open(os.path.join(M, "chat_template.jinja")).read())
SRC = open(os.path.join(M, "qbench_prompts.md")).read()
EXTRA = ["def f(x):\n\treturn x  # tab\n", "  \n\n   ", "日本語のテキストと中文混合。", "emoji 🚀🔥👍🏽 ",
         "<think>not a real tag</think>", "</s> <|endoftext|> literal", "ÄÖÜ ß café naïve", "​ "]

def chunk(rng, n):
    out = []
    while sum(map(len, out)) < n:
        if rng.random() < 0.3: out.append(rng.choice(EXTRA))
        else:
            a = rng.randrange(0, len(SRC) - 600); out.append(SRC[a:a + rng.randrange(20, 600)])
    return "".join(out)

CONVS = int(os.environ.get("CONVS", "12")); TURNS = int(os.environ.get("TURNS", "8"))
checked = bad = hits = 0; t_full = t_cached = 0.0
for c in range(CONVS):
    rng = random.Random(c)
    msgs = [{"role": "system", "content": chunk(rng, 2000)}] if c % 2 else []
    msgs.append({"role": "user", "content": chunk(rng, rng.choice((20000, 60000, 200000)))})
    for t in range(TURNS):
        prompt = tpl.render(messages=msgs, add_generation_prompt=True)
        a = time.perf_counter(); ref = tok.encode(prompt, encode_special_tokens=True); b = time.perf_counter()
        n0 = len(EC._entries); ids = EC.encode_prompt(tok, prompt, False, None); d = time.perf_counter()
        t_full += b - a; t_cached += d - b; checked += 1
        same = torch.equal(ref, ids); bad += not same
        if not same: print(f"MISMATCH conv {c} turn {t}: {ref.shape[-1]} vs {ids.shape[-1]}", flush=True)
        msgs.append({"role": "assistant", "content": chunk(rng, rng.randrange(50, 3000))})
        msgs.append({"role": "user", "content": chunk(rng, rng.randrange(50, 4000))})
print(f"RESULT {checked} prompts checked, {bad} mismatches; tokenize full {t_full:.2f}s, cached {t_cached:.2f}s")
