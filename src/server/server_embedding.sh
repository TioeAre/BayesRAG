#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

VLLM_CACHE_ROOT="/cache/vllm" /opt/miniforge3/bin/conda run -n nomic --live-stream vllm serve /models/Qwen3-Embedding-4B --served-model-name Qwen3-Embedding-4B --task embed --host 0.0.0.0 --port 12501 --gpu_memory_utilization=0.2 --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
# (EngineCore_DP0 pid=3695502) INFO 12-12 16:45:38 [gpu_model_runner.py:2653] Model loading took 7.5387 GiB and 8.971304 seconds

# http://host.docker.internal:61217/v1
# http://host.docker.internal:12501/v1/embeddings
# http://host.docker.internal:12502/rerank