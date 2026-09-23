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

What each one is worth is in the main [README](../../README.md#the-native-engine-tuned-for-gb10).

## What to expect

Measured 2026-09-23 through the API, one request at a time:

| | |
|---|---|
| Memory in use while serving | about 65 GB (about 59 GiB of it the model and KV cache) |
| Prefill, cold prompt | about 840 tok/s (24k-token prompt: 29 s to first token) |
| Follow-up turn, cached history + ~850 new tokens | about 3.6 s to first token |
| Decode, 400-token code answer, the model's default sampling | about 56 tok/s, 66% of drafted tokens accepted |
| First request after start | about 8 s extra (one-time kernel setup) |

Three things in the image are there for speed, so keep them when you edit:
`chunk_size: 8192` in `config.yml` (larger prefill chunks; TabbyAPI's default
of 2048 prefills at about 460 tok/s), `patch_exllamav3_checkpoints.py` (keeps
the recurrent-state checkpoints taken every 2,048 tokens on the GPU instead of
copying them to RAM), and the `qwen38` sampler preset (the model's own
sampling defaults for requests that set none; without it TabbyAPI samples
untruncated and fewer drafted tokens are accepted).

Short prompts still take about 3 s to the first token. That is the engine's
MoE prefill for this pack: its expert weights do not qualify for exllamav3's
fused prefill kernel, so small prefills run one expert at a time. Settings
cannot change that.

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
