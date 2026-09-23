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

## Build the image

```bash
docker build -t qwen38-exl3-tabby docker/tabbyapi
```

The build compiles the exllamav3 CUDA extension for sm_121, so the first build
takes a while. Rebuild after you change `Dockerfile` or `config.yml`. The
installed versions are in the image:

```bash
docker run --rm --entrypoint cat qwen38-exl3-tabby /app/BUILD_INFO
```

## Start

```bash
bash scripts/exl3_native/tuning/drop-model-cache.sh ~/models/Qwen3.8-Flash-Next-EXL3
docker run -d --name qwen38-tabby --gpus all \
  --cpuset-cpus 5-9,15-19 \
  --security-opt no-new-privileges \
  -p 5000:5000 \
  -v ~/models/Qwen3.8-Flash-Next-EXL3:/models/Qwen3.8-Flash-Next-EXL3:ro \
  qwen38-exl3-tabby
docker logs -f qwen38-tabby      # Ctrl+C stops following, not the server
```

- **Run `drop-model-cache.sh` before every start.** On GB10, cached file pages
  count against GPU memory, and the load fails if they are still there. The
  script needs no root.
- `--cpuset-cpus 5-9,15-19` keeps the server on the fast cores (about +2 tok/s).
- `:ro` mounts the model read-only, so the container cannot change it.
- To start it again after a reboot, add `--restart unless-stopped`. The
  page-cache step is then skipped, which is fine right after a reboot.

The model loads first and the API starts after it. The server is ready when
the log shows `Serving OAI API on http://0.0.0.0:5000`.

## Check that it works

From the Spark (from another machine, use `http://<spark-ip>:5000`):

```bash
curl -s localhost:5000/health
curl -s localhost:5000/v1/models
```

Text:

```bash
curl -s localhost:5000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next-EXL3",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "max_tokens": 400
}'
```

Image (a local file, sent as base64):

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s localhost:5000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next-EXL3",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}}
  ]}],
  "max_tokens": 300
}'
```

An `http(s)://` URL works in `image_url` too; the server downloads it.

For any OpenAI client library, use base URL `http://<spark-ip>:5000/v1`. There
is no API key, so pass any string where the client requires one.

## Stop, restart, remove

```bash
docker stop qwen38-tabby         # stop, frees the GPU memory
docker start qwen38-tabby        # start the same container again (run drop-model-cache.sh first)
docker rm -f qwen38-tabby        # remove it; needed before a new `docker run` with the same name
```

After you rebuild the image, run `docker rm -f qwen38-tabby` and then the
Start commands again. `docker start` keeps using the old image.

## Change settings

**Settings in `config.yml`** (context length, cache, draft, vision, auth): edit
`docker/tabbyapi/config.yml` and rebuild. To change them without rebuilding,
mount the file over the built-in one by adding this to `docker run`:

```bash
  -v "$PWD/docker/tabbyapi/config.yml:/app/config.yml:ro" \
```

**GB10 settings** (environment variables): pass `-e NAME=value` to `docker run`.

| Variable | Default | What it does |
|---|---|---|
| `EXL3_DRAFT_CONFIDENCE` | `0.6` | when to stop drafting early (the launcher's `-dc`) |
| `EXL3_GR_INT8` | `1` | mixer weights stored int8 (faster, 1.6 GB less memory) |
| `EXL3_MOE_COOP_WIDE` | `1` | wide MoE kernel tile for GB10 |
| `EXL3_INT8_GEMV` | `0` | int8 GEMV off (slower on GB10) |
| `EXL3_MTP_HEAD_N` | `65536` | draft uses a 64K-column slice of the output head |
| `EXL3_NGRAM_STREAM` | `0` | n-gram table streaming off |

What each one is worth is in the main [README](../../README.md#the-native-engine-tuned-for-gb10).

## Access and safety

- The API has **no key**, and `-p 5000:5000` opens it to the whole network.
  Anyone who can reach port 5000 can use it, including TabbyAPI's admin
  endpoints (unload the model, load another from `/models`). To keep it on the
  Spark only, use `-p 127.0.0.1:5000:5000`. To require a key, set
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
| load fails with not enough memory, while `free -g` shows plenty available | page cache; `docker rm -f qwen38-tabby`, run `drop-model-cache.sh`, start again |
| `docker: Error response ... name "/qwen38-tabby" is already in use` | `docker rm -f qwen38-tabby`, or `docker start qwen38-tabby` to reuse it |
| `could not select device driver "" with capabilities: [[gpu]]` | the NVIDIA container runtime is missing; it ships with DGX OS |
| build fails at the `EXL3_DRAFT_CONFIDENCE` step | TabbyAPI's code changed at `TABBYAPI_REF`; keep the pinned commit |
| `ERROR: TabbyAPI requires exllamav3 1.5.1` | `TABBYAPI_REF` was moved past `2186cdb`; the fork reports 1.5.0 |
| images are ignored or rejected | check that `vision: true` is in `config.yml` and that `preprocessor_config.json` is in the model folder |
