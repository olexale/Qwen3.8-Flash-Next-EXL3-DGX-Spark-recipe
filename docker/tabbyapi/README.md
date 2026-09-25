# TabbyAPI in Docker (native exllamav3 path)

OpenAI-compatible API, with image input, for Qwen3.8-Flash-Next EXL3 on one
DGX Spark. The image runs [TabbyAPI](https://github.com/theroyallab/tabbyAPI)
on the tuned [vcruz305/exllamav3 `329e051`](https://github.com/vcruz305/exllamav3/commit/329e051)
fork with the same settings as `scripts/exl3_native/tuning/run-qwen38-exl3.sh`.

Everything below runs **on the Spark**, from the repo root.

## Once: download the pack

```bash
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 3.05bpw_h5_ng5 \
  --local-dir ~/models/Qwen3.8-Flash-Next-EXL3
```

About 80 GB. Use the pack as downloaded: the native path does not need
`prepare_pack.sh`, which rewrites the pack for vLLM only.

## Start and stop

From the repo root. Both scripts read `.env` (copy `.env.sample` if you have
none):

```bash
./start_tabby.sh --build     # first time, or after changing anything in docker/tabbyapi
./start_tabby.sh             # later starts
./stop_tabby.sh              # stop; the log is saved to logs/
docker logs -f qwen38-tabby  # follow the log (Ctrl+C stops following, not the server)
```

`start_tabby.sh` refuses to start while something else holds the GPU (stop
vLLM with `./stop_vllm.sh` first), starts the container, and follows the log
until the API answers, about a minute. `./start_tabby.sh --no-launch` prints
the `docker run` command without running it.

Settings from `.env`:

| Variable | Default | |
|---|---|---|
| `MODEL_DIR` | `~/models/Qwen3.8-Flash-Next-EXL3` | shared with vLLM |
| `BIND` | `0.0.0.0` | shared with vLLM; `127.0.0.1` keeps it on the Spark |
| `PORT` | `18300` | shared with vLLM |
| `SERVED_NAME` | `qwen3.8-flash-next` | shared with vLLM; the model name clients see |
| `TABBY_RESTART` | `unless-stopped` | Docker restart policy |
| `TABBY_CPUSET` | `5-9,15-19` | the fast cores (about +2 tok/s) |
| `TABBY_CONFIG` | none | a `config.yml` to use instead of the image's, no rebuild |
| `TABBY_CACHE_VOLUME` | `qwen38-tabby-cache` | Docker volume for kernel tuning results; empty to not keep them |
| `TABBY_IMAGE`, `TABBY_CONTAINER` | `qwen38-exl3-tabby:latest`, `qwen38-tabby` | |

What the script and image take care of:

- **Restarts.** After a crash, and after a reboot, Docker starts the container
  again. `./stop_tabby.sh` (or `docker stop`) ends that.
- **Page cache.** On GB10, cached file pages count against GPU memory, and a
  load fails if the pack is still cached from the last one. The container
  drops the pack's pages itself before every load, so restarts work unattended.
- **A pack prepared for vLLM.** `prepare_pack.sh` rewrites `config.json` and the
  index; when their `*.native` originals are there, the script mounts those
  instead. The pack on disk is not changed.
- **Read-only model, no privileges.** The model is mounted read-only and the
  server runs as an unprivileged user.

Installed versions: `docker run --rm --entrypoint cat qwen38-exl3-tabby /app/BUILD_INFO`.

## Check that it works

From the Spark, with the default `PORT=18300` (from another machine, use
`http://<spark-ip>:18300`):

```bash
curl -s localhost:18300/health
curl -s localhost:18300/v1/models
```

Text:

```bash
curl -s localhost:18300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "max_tokens": 400
}'
```

Image (a local file, sent as base64):

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s localhost:18300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}}
  ]}],
  "max_tokens": 300
}'
```

An `http(s)://` URL works in `image_url` too; the server downloads it.

For any OpenAI client library, use base URL `http://<spark-ip>:18300/v1`. There
is no API key, so pass any string where the client requires one.

## Change settings

**Settings in `config.yml`** (context length, cache, draft, vision, auth): edit
`docker/tabbyapi/config.yml`, then `./stop_tabby.sh && ./start_tabby.sh --build`.
To try a change without rebuilding, point `TABBY_CONFIG` in `.env` at your
copy of the file.

**GB10 settings** (environment variables): set them in `.env`; `start_tabby.sh`
passes them to the container.

| Variable | Default | What it does |
|---|---|---|
| `EXL3_DRAFT_CONFIDENCE` | `0.6` | when to stop drafting early (the launcher's `-dc`) |
| `EXL3_GR_INT8` | `1` | mixer weights stored int8 (faster, 1.6 GB less memory) |
| `EXL3_MOE_COOP_WIDE` | `1` | wide MoE kernel tile for GB10 |
| `EXL3_INT8_GEMV` | `0` | int8 GEMV off (slower on GB10) |
| `EXL3_MTP_HEAD_N` | `65536` | draft uses a 64K-column slice of the output head |
| `EXL3_NGRAM_STREAM` | `1` | read the ~30 GiB n-gram table from NVMe as needed; `0` loads all of it into RAM (no faster, about 30 GiB more memory) |
| `EXL3_GR_COLLAPSE` | `1` | the fused hyper-connection collapse kernel (`patch_exllamav3_gr_collapse.py`); `0` = the fork's torch code, ~18% slower long prompts |
| `EXL3_MOE_FUSED_UNIFORM` | `1` | the fused MoE kernel (`patch_exllamav3_fused_moe.py`); `0` restores the fork's per-expert path (about 2x slower short prompts and concurrent decode) |
| `EXL3_QSA_STAGE` | `1` | sparse attention in prefill dequantizes the 8-bit K/V once per layer (`patch_exllamav3_qsa_stage.py`, bit-identical); `0` = dequantize per gathered tile, ~8% slower long prompts |

What each one is worth is in the main [README](../../README.md#the-native-engine-tuned-for-gb10).

## What to expect

Measured 2026-09-25 through the API:

| | |
|---|---|
| Memory in use while serving | about 72 GiB of device memory plus 4 GiB host with three ~115k-token conversations cached; system total ~79.5 GiB |
| Prefill, cold prompt | about 1,280–1,330 tok/s (24k-token prompt: 19 s; 115k: 87 s) |
| Cold short prompt, ~600 tokens | about 1.1 s to first token |
| Follow-up turn, cached history + ~850 new tokens | about 1.5 s on a 25k conversation, 1.6–1.9 s on 115k |
| Conversations kept in the prefix cache | three at full length (262,144 tokens each) |
| Decode, 400-token code answer, the model's default sampling | about 54 tok/s (48–56), one session |
| Decode, three sessions at once | about 54 tok/s together, ~18–20 each |
| First request after start | under 2 s; only the very first start after a new image pays ~20 s of kernel tuning |

What keeps it fast, so keep these when you edit:

- `patch_exllamav3_fused_moe.py`: the exllamav3 fork (`785f206`) switched off
  its fused MoE kernel for packs like this one, so every expert ran as its own
  launch. The patch turns it back on (`EXL3_MOE_FUSED_UNIFORM=1`, the image
  default): short prompts and follow-up turns ~2x faster, long prompts ~20%,
  three concurrent sessions decode 2.1x faster (26 → 55 tok/s together).
- `patch_exllamav3_gr_collapse.py`: a CUDA kernel for the hyper-connection
  stream collapse that the fork runs in torch during prefill (four ~335 MB fp32
  temporaries per call on an 8k chunk). Bit-identical output, long prompts ~18%
  faster (`EXL3_GR_COLLAPSE=0` turns it off).
- `patch_exllamav3_qsa_stage.py`: the sparse attention kernel dequantized each
  gathered 8-bit K/V tile once per query row; the patch dequantizes the
  sequence once per layer. Bit-identical, the kernel 2x, long prompts ~8%
  faster (`EXL3_QSA_STAGE=0` turns it off).
- `chunk_size: 8192` in `config.yml` (TabbyAPI's default of 2048 is much slower;
  16384 is 1.7% faster still but needs 1.6 GiB more).
- `patch_exllamav3_checkpoints.py` (recurrent-state checkpoints stay on the GPU
  instead of being copied to RAM).
- The `qwen38` sampler preset (without it TabbyAPI samples untruncated and fewer
  drafted tokens are accepted).
- The `qwen38-tabby-cache` volume that `start_tabby.sh` mounts (kernel tuning
  results; without it every start re-tunes for ~20 s).

The draft settings (5 tokens, confidence 0.6) were re-checked on sampled code,
prose and tool-call output: no other combination was clearly faster.
Tried, no help: a 128-row tile for the fused MoE kernel (register spills, slower), other MoE group widths, `EXL3_MOE_COOP_WIDE=0`, no vision tower, the n-gram table in
RAM, other fused row limits, chunk 4096. Details:
[OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md#results-2026-09-24).

## Access and safety

- The API has **no key**, and `BIND=0.0.0.0` opens it to the whole network.
  Anyone who can reach the port can use it, including TabbyAPI's admin
  endpoints (unload the model, load another from `/models`). To keep it on the
  Spark only, set `BIND=127.0.0.1`. To require a key, set
  `disable_auth: false` in `config.yml`; the key is then in
  `docker exec qwen38-tabby cat /app/api_tokens.yml`.
- With auth off, any web page open in a browser on your network can also call
  the API. `network.allowed_origins: []` in `config.yml` blocks that, but
  it also blocks web UIs that call the API from the browser.
- The container runs as an unprivileged user and cannot change the host or the
  model files. It still shares the GB10's memory with the whole machine, so if
  the model runs out of memory the Spark itself can still freeze.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `start_tabby.sh` says the GPU is in use | the vLLM container (or something else) is running; `./stop_vllm.sh` |
| `start_tabby.sh` says TabbyAPI failed to start and Docker keeps retrying | read `docker logs qwen38-tabby`, then `./stop_tabby.sh` to end the retries |
| load fails with not enough memory, while `free -g` shows plenty available | something else is using memory, or the image predates the page-cache entrypoint; `./start_tabby.sh --build` |
| `could not select device driver "" with capabilities: [[gpu]]` | the NVIDIA container runtime is missing; it ships with DGX OS |
| build fails at the `EXL3_DRAFT_CONFIDENCE` step | TabbyAPI's code changed at `TABBYAPI_REF`; keep the pinned commit |
| `ERROR: TabbyAPI requires exllamav3 1.5.1` | `TABBYAPI_REF` was moved past `2186cdb`; the fork reports 1.5.0 |
| images are ignored or rejected | check that `vision: true` is in `config.yml` and that `preprocessor_config.json` is in the model folder |
