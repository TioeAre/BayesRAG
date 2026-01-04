#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

BASE_MODEL_DIR="$DATA_ROOT/models"
CACHE_DIR="$DATA_ROOT/models/cache"

echo "$BASE_MODEL_DIR"

# # nomic-ai/colnomic-embed-multimodal-7b
# REPO_ID="nomic-ai/colnomic-embed-multimodal-7b"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget

# # nomic-ai/nomic-embed-multimodal-7b
# REPO_ID="nomic-ai/nomic-embed-multimodal-7b"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget

# # nomic-ai/colnomic-embed-multimodal-3b
# REPO_ID="nomic-ai/colnomic-embed-multimodal-3b"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# # hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget

# # timm/PE-Core-bigG-14-448
# REPO_ID="timm/PE-Core-bigG-14-448"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# # hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget

# # Qwen/Qwen3-Embedding-0.6B
# REPO_ID="Qwen/Qwen3-Embedding-0.6B"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# # hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget

# # Qwen/Qwen3-Embedding-4B
# REPO_ID="Qwen/Qwen3-Embedding-4B"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# # hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget

# # Qwen/Qwen3-Reranker-4B
# REPO_ID="Qwen/Qwen3-Reranker-4B"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# # hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget

# # Qwen/Qwen3-Reranker-0.6B
# REPO_ID="Qwen/Qwen3-Reranker-0.6B"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# # hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget

# # BAAI/bge-reranker-v2-m3
# REPO_ID="BAAI/bge-reranker-v2-m3"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# # hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget

# # Qwen/Qwen3-VL-32B-Instruct
# REPO_ID="Qwen/Qwen3-VL-32B-Instruct"
# MODEL_NAME="${REPO_ID##*/}"
# LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
# hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"
# # hfd.sh "$REPO_ID" --local-dir "$LOCAL_DIR" --tool wget


# openbmb/VisRAG-Ret
REPO_ID="openbmb/VisRAG-Ret"
MODEL_NAME="${REPO_ID##*/}"
LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"

# jinaai/jina-embeddings-v4
REPO_ID="jinaai/jina-embeddings-v4"
MODEL_NAME="${REPO_ID##*/}"
LOCAL_DIR="${BASE_MODEL_DIR}/${MODEL_NAME}"
hf download --cache-dir "$CACHE_DIR" --local-dir "$LOCAL_DIR" "$REPO_ID"