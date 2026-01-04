#!/bin/bash

export VLLM_CACHE_ROOT="/cache/vllm"

echo "starting Qwen3-Embedding-4B (Port 12501, GPU 0.3)..."

/opt/miniforge3/bin/conda run -n nomic --live-stream vllm serve \
    /models/Qwen3-Embedding-4B \
    --served-model-name Qwen3-Embedding-4B \
    --task embed \
    --host 0.0.0.0 \
    --port 12501 \
    --gpu_memory_utilization=0.3 \
    --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' &


sleep 20

# echo "starting Qwen3-Reranker-4B (Port 12502, GPU 0.6)..."

# /opt/miniforge3/bin/conda run -n nomic --live-stream vllm serve \
#     /models/Qwen3-Reranker-4B \
#     --served-model-name Qwen3-Reranker-4B \
#     --host 0.0.0.0 \
#     --port 12502 \
#     --gpu_memory_utilization=0.6 \
#     --max-model-len 10240

# echo "starting bge-reranker-v2-m3 (Port 12502, GPU 0.25)..."

# /opt/miniforge3/bin/conda run -n nomic --live-stream vllm serve \
#     /models/bge-reranker-v2-m3 \
#     --served-model-name bge-reranker-v2-m3 \
#     --host 0.0.0.0 \
#     --port 12502 \
#     --gpu_memory_utilization=0.2  &

# sleep 20

echo "starting mineru (Port 12503, GPU 0.25)..."

MINERU_MODEL_SOURCE=local MINERU_TOOLS_CONFIG_JSON="/models/mineru.json" /opt/miniforge3/bin/conda run -n nomic --live-stream mineru-openai-server --engine vllm --host 0.0.0.0 --port 12503 --gpu_memory_utilization=0.3
# mineru-api --host 0.0.0.0 --port 12502
# -b vlm-http-client -u http://127.0.0.1:30000

