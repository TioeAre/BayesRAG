#!/usr/bin/bash
export EMBEDDING_MODEL_NAME="vllm-Qwen3-Embedding-4B"
# export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export RERANK_MODEL_NAME="vllm-bge-reranker-v2-m3"

export TEXT_DATABASE="./database/text_db_docbench"
export IMAGE_DATABASE="./database/image_db_docbench"
export SHORTCUT_DATABASE="./database/shortcut_db_docbench"

export MODEL_PA_BY_KG=true
# export MODEL_PA_BY_KG=false
export ONLY_ADD_KG=true
# export SHORT_CUT=false

export PARSE_RESULT_DIR="./storge/docbench"

export MONGO_URI="mongodb://admin:admin@localhost:27018/?authSource=admin&directConnection=true"
export NEO4J_URI="neo4j://localhost:7688"

export TEST_BENCH_FULL=true
export TEST_RAG=false
export IF_ASK=false
export ADD_VECTOR=true
export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=WARNING

export ADD_VECTOR_TEXT_BATCH_SIZE=150
export ADD_VECTOR_IMAGE_BATCH_SIZE=75
export ADD_KG_PDF_BATCH_SIZE=100

export ENABLE_THINK=false

# export OPENAI_LLM_MAX_TOKENS=3072
export OPENAI_LLM_MAX_TOKENS=1024
# export OPENAI_LLM_MAX_TOKENS=1024

export RAGANYTHING_LLM_MODEL_MAX_ASYNC=50

export SUMMARY_CONTEXT_SIZE=1024

/opt/miniforge3/bin/conda run -n nomic --live-stream python ./eval/DocBench/test_DocBench.py