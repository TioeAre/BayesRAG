#!/usr/bin/bash

cd /projects/rag_method/ViDoRAG/ || exit

export TEST_BENCH_FULL=true
export EVAL_CONCURRENCY_LIMIT=50

export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=DEBUG

# modified in /projects/rag_method/ViDoRAG/llms/llm.py
# export GENERATE_MODEL_NAME="Qwen3-VL-32B-Instruct"
export GENERATE_MODEL_NAME="azure-gpt-4o-mini"  # azure-gpt-4o

export WRITE_RESULTS=true
export IF_ASK=true
# export RESULT_DIR_NAME="timestamp"
# export RESULT_DIR_NAME="Qwen3_Bench_results"
export RESULT_DIR_NAME="gpt_4o_mini_Bench_results"
export HF_ENDPOINT="https://hf-mirror.com"

/opt/miniforge3/bin/conda run -n vidorag --live-stream python eval/test_mmLongBench.py