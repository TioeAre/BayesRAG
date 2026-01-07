#!/usr/bin/bash

# add vector
export TEST_BENCH_FULL=true
export TEST_RAG=false
export IF_ASK=false
export ADD_VECTOR=true
export MODEL_PA_BY_KG=false
export PRINT_FILE=false
export RERANK=false
export PRINT_TERMINAL_LEVEL=DEBUG
export TEXT_DATABASE="./database/new_th_text_db"
export IMAGE_DATABASE="./database/image_caption_db"
export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export SHORT_CUT=false

export PARSE_RESULT_DIR="./storge/mineru"
export MINERU_BACKEND="vlm-http-client"    # vlm-http-client pipeline

export ADD_VECTOR_TEXT_BATCH_SIZE=5
export ADD_VECTOR_IMAGE_BATCH_SIZE=5
export ADD_KG_PDF_BATCH_SIZE=5

/opt/miniforge3/bin/conda run -n bayesrag --live-stream python ./eval/mmLongBench/test_mmLongBench.py
