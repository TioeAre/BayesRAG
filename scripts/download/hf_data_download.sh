#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# git lfs install

echo "$DATA_ROOT"
BASE_DATA_DIR="$DATA_ROOT/data/rag"
CACHE_DIR="$DATA_ROOT/data/cache"


# yubo2333/MMLongBench-Doc
REPO_ID="yubo2333/MMLongBench-Doc"
DATA_NAME="${REPO_ID##*/}"
LOCAL_DIR="${BASE_DATA_DIR}/${DATA_NAME}"
# hf download --repo-type dataset --cache-dir "$CACHE_DIR"  --local-dir "$LOCAL_DIR" "$REPO_ID"
hfd.sh "$REPO_ID" --dataset --local-dir "$LOCAL_DIR" --tool wget
