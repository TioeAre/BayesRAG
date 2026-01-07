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
export TEXT_DATABASE="./database/new_th_text_db"
export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export SHORT_CUT=false
export MINERU_BACKEND="vlm-http-client"

export ADD_VECTOR_TEXT_BATCH_SIZE=1
export ADD_VECTOR_IMAGE_BATCH_SIZE=20
export ADD_KG_PDF_BATCH_SIZE=20

/opt/miniforge3/bin/conda run -n bayesrag --live-stream python ./eval/mmLongBench/test_mmLongBench.py
