#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# ragflow 不支持 Qwen3-Reranker-4B
# VLLM_CACHE_ROOT="/cache/vllm" /opt/miniforge3/bin/conda run -n nomic --live-stream vllm serve /models/Qwen3-Reranker-4B --served-model-name Qwen3-Reranker-4B --host 0.0.0.0 --port 12502 --gpu_memory_utilization=0.5 --max-model-len 10240

VLLM_CACHE_ROOT="/cache/vllm" /opt/miniforge3/bin/conda run -n nomic --live-stream vllm serve /models/bge-reranker-v2-m3 --served-model-name bge-reranker-v2-m3 --host 0.0.0.0 --port 12502 --gpu_memory_utilization=0.15 --max-model-len 8192

# http://host.docker.internal:12502/rerank