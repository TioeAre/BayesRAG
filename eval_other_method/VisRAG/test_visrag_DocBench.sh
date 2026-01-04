#!/usr/bin/bash

# cd /projects/rag_method/UltraRAG || exit

# /opt/miniforge3/bin/conda run -n ultrarag --live-stream ultrarag build examples/build_image_corpus_DocBench.yaml
# /opt/miniforge3/bin/conda run -n ultrarag --live-stream ultrarag run examples/build_image_corpus_DocBench.yaml

# /opt/miniforge3/bin/conda run -n ultrarag --live-stream ultrarag build examples/corpus_index_DocBench.yaml
# /opt/miniforge3/bin/conda run -n ultrarag --live-stream ultrarag run examples/corpus_index_DocBench.yaml

# /opt/miniforge3/bin/conda run -n ultrarag --live-stream ultrarag build examples/eval_visrag_DocBench.yaml
# /opt/miniforge3/bin/conda run -n ultrarag --live-stream ultrarag run examples/eval_visrag_DocBench.yaml

export TEST_BENCH_FULL=true
export EVAL_CONCURRENCY_LIMIT=10

export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=DEBUG

# export GENERATE_MODEL_NAME="Qwen3-VL-32B-Instruct"
export GENERATE_MODEL_NAME="azure-gpt-4o-mini"  # azure-gpt-4o

export WRITE_RESULTS=true
export IF_ASK=true
# export RESULT_DIR_NAME="timestamp"
# export RESULT_DIR_NAME="Qwen3_Bench_results"
export RESULT_DIR_NAME="gpt_4o_mini_Bench_results"

export VISRAG_RAW_RESULTS_PATH="/projects/rag_method/UltraRAG/output/memory_test_DocBench_eval_visrag_DocBench_20251217_150853.json"

/opt/miniforge3/bin/conda run -n nomic --live-stream python ./eval_other_method/VisRAG/eval_visrag_docbench.py