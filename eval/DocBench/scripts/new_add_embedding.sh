#!/usr/bin/bash

# add vector
export TEST_BENCH_FULL=true
export TEST_RAG=false
export IF_ASK=false
export ADD_VECTOR=true
export MODEL_PA_BY_KG=false
export PRINT_FILE=false
export RERANK=false
export PRINT_TERMINAL_LEVEL=INFO
export TEXT_DATABASE="./database/new_text_db_docbench"
export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export SHORT_CUT=false
export PARSE_RESULT_DIR="./storge/docbench"
export MINERU_BACKEND="vlm-http-client"

export ADD_VECTOR_TEXT_BATCH_SIZE=10
export ADD_VECTOR_IMAGE_BATCH_SIZE=20
export ADD_KG_PDF_BATCH_SIZE=20

/opt/miniforge3/bin/conda run -n nomic --live-stream python ./eval/DocBench/test_DocBench.py
