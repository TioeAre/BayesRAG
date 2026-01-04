#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# http://host.docker.internal:61217/v1
# http://host.docker.internal:12501/embeddings
# http://host.docker.internal:12502/rerank

# Qwen3-VL-32B-Instruct
VLLM_CACHE_ROOT="/cache/vllm" /opt/miniforge3/bin/conda run -n qwenvl --live-stream vllm serve /models/Qwen3-VL-32B-Instruct --served-model-name Qwen3-VL-32B-Instruct --tensor-parallel-size 2 --limit-mm-per-prompt.video 0 --async-scheduling --host 0.0.0.0 --port 12503

SGLANG_DG_CACHE_DIR="/cache/sglang" /opt/miniforge3/bin/conda run -n qwenvl --live-stream -m sglang.launch_server \
   --model-path /models/Qwen3-VL-32B-Instruct \
   --host 0.0.0.0 \
   --port 12503 \
   --tp 4

# python3 -m
# vllm.entrypoints.openai.api_server
# --model ${MODEL_PATH}
# --served-model-name ${MODEL_NAME}
# --tensor-parallel-size ${MLP_GPU_NUM}
# --enable-prefix-caching
# --enable-chunked-prefill
# --max-model-len 32768
# --port 8000
# --disable-log-requests

export http_proxy="http://127.0.0.1:61110"
export HTTP_PROXY="http://127.0.0.1:61110"
export https_proxy="http://127.0.0.1:61110"
export HTTPS_PROXY="http://127.0.0.1:61110"
