# Interactive Qwen benchmark

Current 16:9 presentation of the measured **Qwen3.8-Flash-Next EXL3** serving envelope on **one NVIDIA DGX Spark**. It is not a live model demo and does not run inference in the browser.

## View

[Open the current benchmark](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/)

The viewer now uses a fixed 1920x1080 presentation stage that scales as one unit, so the landscape composition no longer breaks or clips when the browser is narrow. Desktop and landscape-mobile views keep the same card layout.

On phones:

- portrait mode shows a small **Best in landscape** prompt instead of reflowing the benchmark into a broken vertical layout;
- **Fullscreen** attempts to enter fullscreen and request landscape orientation where the browser supports it;
- swipe left/right on the benchmark to change scenes;
- the bottom controls become a compact horizontally scrollable strip;
- all seven scenes remain the same 16:9 design used for desktop captures.

The page was rebuilt on 2026-09-14 so the default scenes no longer lead with the superseded Sep 7/8 measurements. Historical data remains in the main repository README for provenance; this viewer intentionally leads with the current Sep 13/14 envelope.

Current headline results:

- direct ExLlamaV3 1.5.0: **58.8 tok/s** single stream
- vLLM + vllm-exl3: **157.6 tok/s** steady aggregate at 8 short-prompt streams
- full **262,144-token configured context** on the 3.05 bpw pack
- vLLM MTP k=3: roughly **50–53 tok/s** below the 163,840-prompt acceptance cliff
- 196K-token cached-prefix TTFT: **1.56 s**, versus **178.72 s** cold
- 4.05 bpw with the packed n-gram table on NVMe: **262K boots**, **954,453-token KV pool**, **18–20 GiB** available memory

## Scenes

- `?scene=overview` — current one-Spark headline numbers
- `?scene=speed` — current MTP k=3 / k=2 / no-draft single-stream sweep
- `?scene=context` — context curve and the 163,840 prompt-token MTP acceptance cliff
- `?scene=scale` — steady-state concurrency scaling
- `?scene=cache` — 196K-token prefix-cache TTFT
- `?scene=revision` — 4.05 bpw resident vs NVMe n-gram-table mode
- `?scene=engines` — direct ExLlamaV3 vs vLLM + vllm-exl3 serving tradeoff

Legacy scene names such as `sweep`, `long`, `cliff`, and `native` still map to their current replacements so old shared links do not break.

Add `&clean=1` for a controls-free view. `?autoplay=1` cycles through the current scenes. Keyboard: arrow keys change scenes, **Space** plays/pauses, **F** toggles fullscreen, and **H** hides or restores controls.

## Engine framing

Direct ExLlamaV3 is the leaner and faster measured single-user path on this pack. For production/API use, another serving layer such as TabbyAPI is normally added.

The vLLM path exists for the complete serving stack: `vllm-exl3` integrates EXL3 into vLLM so the recipe can use the OpenAI-compatible API, reasoning/tool-call parsers, structured output, prefix caching, batching/concurrency, and broader vLLM tooling out of the box.

## Data and methodology

`docs/benchmark-data.json` is the current presentation dataset. It intentionally contains the current Sep 13/14 serving envelope rather than the old headline sweep. The main README remains the detailed source of methodology, historical comparisons, limitations, and exact runtime notes.

Important boundaries:

- The **163,840** number is a **prompt-token MTP acceptance cliff**, not the configured context ceiling. The configured model context remains 262,144.
- Decode excludes TTFT.
- Cold prefill uses unique prefixes and was checked against vLLM's own prefix-cache counters.
- Concurrency scenes use **steady aggregate** throughput over the interval where every stream is decoding; this avoids the old window metric that made vLLM look like it plateaued at two streams.
- The 4.05 bpw NVMe mode is a deliberate memory/speed trade: the packed n-gram table stays file-backed, enabling full context and much more KV headroom at some decode cost.
- This page presents recorded measurements only. It does not establish new quality results.

## Render PNGs locally

Optional setup:

```sh
python -m pip install playwright
python -m playwright install chromium
python scripts/render_benchmark.py
```

This renders all seven scenes to `benchmark-renders/` and reports JavaScript errors or unexpected network requests. No model, CUDA install, or DGX Spark is required to render the presentation.

## GitHub Pages

The root `index.html` redirects into `docs/`, preserving query parameters. The repository works with GitHub Pages configured from either `main / (root)` or `main /docs`; both roots include `.nojekyll`.

## Credits

Pack, codec and kernels: Turboderp / ExLlamaV3. Engine: vLLM. EXL3/vLLM integration and measurements: Victor Cruz (@ViC305). See the main repository and `vllm-exl3` notices for upstream attribution and licensing boundaries.
