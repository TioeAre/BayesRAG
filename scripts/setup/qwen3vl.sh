#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

conda create -n qwenvl python=3.13 -y
conda activate qwenvl

conda run -n qwenvl --live-stream pip install -U vllm
conda run -n qwenvl --live-stream pip install qwen-vl-utils==0.0.14 accelerate flashinfer-python
conda run -n qwenvl --live-stream pip install -U flash-attn --no-build-isolation