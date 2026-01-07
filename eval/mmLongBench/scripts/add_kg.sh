#!/usr/bin/bash
export EMBEDDING_MODEL_NAME="vllm-Qwen3-Embedding-4B"
export RERANK_MODEL_NAME="vllm-bge-reranker-v2-m3"

export MODEL_PA_BY_KG=true
export ONLY_ADD_KG=true
# export SHORT_CUT=false

export PARSE_RESULT_DIR="./storge/mineru"

export MONGO_URI="mongodb://admin:admin@localhost:27017/?authSource=admin&directConnection=true"
export NEO4J_URI="neo4j://localhost:7687"

export TEST_BENCH_FULL=true
export TEST_RAG=false
export IF_ASK=false
export ADD_VECTOR=true
export PRINT_FILE=true
export PRINT_TERMINAL_LEVEL=INFO

export ADD_VECTOR_TEXT_BATCH_SIZE=200
export ADD_VECTOR_IMAGE_BATCH_SIZE=150
export ADD_KG_PDF_BATCH_SIZE=200

export ENABLE_THINK=false

# export OPENAI_LLM_MAX_TOKENS=3072
export OPENAI_LLM_MAX_TOKENS=2048
# export OPENAI_LLM_MAX_TOKENS=1024

export RAGANYTHING_LLM_MODEL_MAX_ASYNC=200

export SUMMARY_CONTEXT_SIZE=2048

/opt/miniforge3/bin/conda run -n bayesrag --live-stream python ./eval/mmLongBench/test_mmLongBench.py