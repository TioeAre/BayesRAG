#!/usr/bin/bash
export TEST_BENCH_FULL=true
export TEST_RAG=true
export IF_ASK=true
export ADD_VECTOR=false
export ONLY_ADD_KG=false
export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=DEBUG

export ENABLE_THINK=true
export OPENAI_LLM_MAX_TOKENS=4096
export RAGANYTHING_LLM_MODEL_MAX_ASYNC=20
export SUMMARY_CONTEXT_SIZE=2048

export EVAL_CONCURRENCY_LIMIT=1

# export EMBEDDING_MODEL_NAME="vllm-Qwen3-Embedding-4B"
# export EMBEDDING_MODEL_NAME="PE-Core-bigG-14-448"
export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export RERANK_MODEL_NAME="vllm-bge-reranker-v2-m3"
# export RERANK_MODEL_NAME="Qwen3-Reranker-4B"

export TEXT_DATABASE="./database/new_th_text_db"
# export TEXT_DATABASE="./database/pe_text_db"
export IMAGE_DATABASE="./database/image_caption_db"
export SHORTCUT_DATABASE="./database/shortcut_db"
export MONGO_URI="mongodb://admin:admin@localhost:27017/?authSource=admin&directConnection=true"
export NEO4J_URI="neo4j://localhost:7687"

export BAYES=true   # or just use text reranker
export DShafer=true # or calculate P(B|A) by linear weight
export MODEL_PA_BY_KG=true  # or calculate P(A) by bbox
export MODEL_PA_BY_KG_WEIGHT=true   # or calculate P(A) by kg relations' frequency
export SHORT_CUT=true
export RERANK=true

export BM25_K=20   # 100
export TEXT_K=1024   # 100   retriever
export IMAGE_K=512
export SHORT_CUT_K=512

export RERANK_TOP_K_FOR_BAYES=100   # rerank in retriever
export COMB_TOP_K_THRESHOLD=15  # to add combinations, 20*20*20
# answering
export RERANK_TOP_K=5
export RERANK_BATCHSIZE=10
export TUPLE_RERANK_TOP_K_TEXT=15    # for bayes method
export TUPLE_RERANK_TOP_K_IMAGE=10
export TUPLE_RERANK_TOP_K_SHORTCUT=15

### calculate P(A) by bbox
export DISTANCE_THRESHOLD=2   # page_size
export EPSILON=0.1  # the score which (text, image, shortcut) is not related
### calculate P(B|A) by Dempster-Shafer
export D_S_ALPHA=0.7 # embedding 是正确的可信度
export D_S_BETA=0.6  # embedding 错误分类的可信度
export SCALING_FACTOR=0.1   # P(A) by KG, probability
### calculate P(B|A) by weight
export WEIGHT_TEXT_VECTOR=0.4
export WEIGHT_IMAGE=0.3
export WEIGHT_SHOERCUT=0.3
export RAGANYTHING_RERANK=false

export RERANK_THRESHOLD=0.1

# export GENERATE_MODEL_NAME="Qwen3-VL-32B-Instruct"
# export GENERATE_MODEL_NAME="azure-gpt-4o"
export GENERATE_MODEL_NAME="azure-gpt-4o-mini"  # azure-gpt-4o

export WRITE_RESULTS=true
# export RESULT_DIR_NAME="timestamp"
# export RESULT_DIR_NAME="Qwen3_Bench_results_test1"
# export RESULT_DIR_NAME="gpt_4o_mini_Bench_results"
export RESULT_DIR_NAME="gpt_4o_mini_Bench_results_final"

/opt/miniforge3/bin/conda run -n bayesrag --live-stream python ./eval/mmLongBench/test_mmLongBench.py
