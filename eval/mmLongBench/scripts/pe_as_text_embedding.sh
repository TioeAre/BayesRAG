#!/usr/bin/bash

# add vector
export TEST_BENCH_FULL=true
export TEST_RAG=false
export IF_ASK=false
export ADD_VECTOR=true
export MODEL_PA_BY_KG=false
export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=DEBUG
export TEXT_DATABASE="./database/pe_text_db"
export EMBEDDING_MODEL_NAME="PE-Core-bigG-14-448"
# /opt/miniforge3/bin/conda run -n bayesrag --live-stream python ./eval/mmLongBench/test_mmLongBench.py

# test
export TEST_BENCH_FULL=false
export TEST_RAG=true
export IF_ASK=true
export BAYES=false
export ADD_VECTOR=false
export MODEL_PA_BY_KG=false
export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=INFO
export TEXT_DATABASE="./database/pe_text_db"
export EMBEDDING_MODEL_NAME="PE-Core-bigG-14-448"
export RERANK_MODEL_NAME="Qwen3-Reranker-4B"

export BM25_K=100
export TEXT_K=100
export IMAGE_K=5
export SHORT_CUT_K=5

/opt/miniforge3/bin/conda run -n bayesrag --live-stream python ./eval/mmLongBench/test_mmLongBench.py