#!/usr/bin/env bash
# Run the patch tests (docker/tabbyapi/tests/test_*.py) inside a TabbyAPI image, on the Spark.
# No model is loaded, so this runs next to a serving TabbyAPI (it only opens a CUDA
# context, for importing exllamav3). The model's chat template is mounted for the
# tool-call round-trip tests.
#
#   docker/tabbyapi/tests/run_tests.sh                          # qwen38-exl3-tabby:latest
#   IMAGE=qwen38-exl3-tabby:new docker/tabbyapi/tests/run_tests.sh toolcall
#
# Arguments select tests by substring of "file::function". Exit status: 0 = all passed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
exec docker run --rm --gpus all \
    -v "$HERE:/tests:ro" \
    -v "$MODEL_DIR/chat_template.jinja:/tests_data/chat_template.jinja:ro" \
    -e TEMPLATE=/tests_data/chat_template.jinja \
    -w /app --entrypoint python3 "${IMAGE:-qwen38-exl3-tabby:latest}" /tests/run.py "$@"
