#!/usr/bin/bash

export TEST_BENCH_FULL=true
export EVAL_CONCURRENCY_LIMIT=10

export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=DEBUG

export GENERATE_MODEL_NAME="azure-gpt-4o-mini"  # azure-gpt-4o

export WRITE_RESULTS=true
export IF_ASK=true
# export RESULT_DIR_NAME="timestamp"
# export RESULT_DIR_NAME="Qwen3_Bench_results"
export RESULT_DIR_NAME="gpt_4o_mini_Bench_results"

/opt/miniforge3/bin/conda run -n nomic --live-stream python ./eval_other_method/pure_model/test_docbench.py