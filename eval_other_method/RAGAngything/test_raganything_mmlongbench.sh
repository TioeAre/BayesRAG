#!/usr/bin/bash
export TEST_BENCH_FULL=true
export TEST_RAG=true
export IF_ASK=true
export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=DEBUG

export ENABLE_THINK=true
export OPENAI_LLM_MAX_TOKENS=4096
export RAGANYTHING_LLM_MODEL_MAX_ASYNC=20
export SUMMARY_CONTEXT_SIZE=2048
export RAGANYTHING_RERANK=true

export EVAL_CONCURRENCY_LIMIT=5

export EMBEDDING_MODEL_NAME="vllm-Qwen3-Embedding-4B"
export RERANK_MODEL_NAME="vllm-bge-reranker-v2-m3"

export MONGO_URI=""
export NEO4J_URI=""

# export IF_GPT4o=true
# export GPT4o_MODEL_NAME="azure-gpt-4o-mini"

export IF_GPT4o=false   # use qwen3-vl-32b

export LLM_MODEL_BASE_URL=""

export WRITE_RESULTS=true
# export RESULT_DIR_NAME="gpt_4o_mini_Bench_results"
export RESULT_DIR_NAME="Qwen3_Bench_results"

/opt/miniforge3/bin/conda run -n bayesrag --live-stream python ./eval_other_method/RAGAngything/test_ragangthing_mmlongbench.py