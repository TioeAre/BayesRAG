# BayesRAG: Probabilistic Mutual Evidence Corroboration for Multimodal Retrieval-Augmented Generation

## System Overview

![framework](./assets/20260109_100720.png)

Our core idea is to treat different retrieval models for multimodal data, such as text and images, as a process analogous to multi-sensor fusion. When independent, heterogeneous modality-specific retrieval systems return results with the shared semantics, it's intuitive to increase the confidence of the retrieval results for this semantically consistent tuple.

## Setup

### Setup with conda

- Create environment

```bash
conda create -n bayesrag python=3.10  -y
conda activate bayesrag
```

- Install with pip

We use [langchain](https://www.langchain.com/) as the basic RAG framework and [RAG-Anything](https://github.com/HKUDS/RAG-Anything) for building the knowledge-graph.

```bash
conda run -n bayesrag --live-stream pip install git+https://github.com/illuin-tech/colpali.git

# langchain
conda run -n bayesrag --live-stream pip install langchain langchain-community langchain-text-splitters langchain-core langchain-experimental langchain_huggingface langchain_chroma langchain_openai

conda run -n bayesrag --live-stream pip install mineru open_clip_torch
conda run -n bayesrag --live-stream pip install pydantic lxml matplotlib tiktoken pypdf PyMuPDF pymupdf pillow rank_bm25 'raganything[all]' nest_asyncio pdf2image
conda run -n bayesrag --live-stream pip install sentence_transformers vllm
```

## Prepare data

We use [MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc), [Docbench](https://github.com/Anni-Zou/DocBench) to evaluate our framewrok. You can follow their official instructions to download the datasets and place them in the `./dataset` directory. The dataset structure is as follows:

```bash
.
├── dataset
│   ├── DocBench
│   │   ├── 0
│   │   ├── 1
│   │   ...
│   └── MMLongBench-Doc
│       ├── data
│       └── documents
├── eval
...
```

## Embedding and Reranker models

### Download pre-trained weights

We use [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) as text retrieval model, The OpenCLIP remaped version of [PE-Core-bigG-14-448](https://huggingface.co/timm/PE-Core-bigG-14-448) as image model and [colnomic-embed-multimodal-3b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-3b) to retrieve document screenshots.

After downloading the weights, you can modify the corresponding environment variables in the `.env` file to specify the model name and path.

```bash
export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export EMBEDDING_MODEL_PATH="./models/Qwen3-Embedding-4B"
export EMBEDDING_CUDA_DEVICE="cuda:0"

export IMAGE_EMBEDDING_MODEL_NAME="PE-Core-bigG-14-448"
export IMAGE_EMBEDDING_MODEL_PATH="./models/PE-Core-bigG-14-448/open_clip_pytorch_model.bin"
export IMAGE_EMBEDDING_CUDA_DEVICE="cuda:0"

export SHORT_CUT_MODEL_NAME="nomic-ai/colnomic-embed-multimodal-3b"
export SHORT_CUT_CUDA_DEVICE="cuda:0"

export RERANK_MODEL_NAME="bge-reranker-v2-m3"
export RERANK_MODEL_PATH="./models/bge-reranker-v2-m3"
export RERANK_CUDA_DEVICE="cuda:0"
```

### [vLLM](https://github.com/vllm-project/vllm) hosted

You can also use Embedding or Reranker that hosted by vLLM. You can specify urls through the following environment variables.

```bash
export EMBEDDING_BASE_URL_VEC="http://localhost:12501/v1"
export RERANKER_BASE_URL_VEC="http://localhost:12502/rerank"
```

**Note**: When using vllm, please change the model name that starting with `vllm-` like the following:

```bash
export EMBEDDING_MODEL_NAME="vllm-Qwen3-Embedding-4B"
export RERANK_MODEL_NAME="vllm-bge-reranker-v2-m3"
```

## Build database

The directory structure of the database is as follows:

```bash
.
├── database
│   ├── image_db_docbench
│   ├── image_db_mmlongbench
│   ├── mongot_docbench
│   ├── mongot_mmlongbench
│   ├── text_db_docbench
│   ├── text_db_mmlongbench
│   ├── raganything_db
│   │   ├── mongo_docbench
│   │   ├── mongo_mmlongbench
│   │   ├── neo4j_docbench
│   │   ├── neo4j_mmlongbench
│   │   └── neo4j_plugins
│   ├── shortcut_db_docbench
│   ├── shortcut_db_mmlongbench
│   ├── text_db_docbench
│   └── text_db_mmlongbench
├── dataset
...
```

### Start Server

We use [MongoDB](https://www.mongodb.com/), [Neo4j](https://neo4j.com/) to build knowledge-graph and [chroma](https://github.com/chroma-core/chroma) to build embedding vector database. You can start the service deployed by docker through the following command:

```bash
#!/usr/bin/bash
docker run --name ragmongo_mmlongbench --hostname localhost --userns=host --user root -e MONGODB_INITDB_ROOT_USERNAME=admin -e MONGODB_INITDB_ROOT_PASSWORD=admin -v ./database/raganything_db/mongo:/data/db -v ./database/mongo_keyfile:/data/configdb/keyfile -v ./database/mongot_mmlongbench:/data/mongot -p 27017:27017 -d mongodb/mongodb-atlas-local

docker run --name ragmongo_doc_bench --hostname localhost --userns=host --user root -e MONGODB_INITDB_ROOT_USERNAME=admin -e MONGODB_INITDB_ROOT_PASSWORD=admin -v ./database/raganything_db/mongo_docbench:/data/db -v ./database/mongo_keyfile:/data/configdb/keyfile -v ./database/mongot_docbench:/data/mongot -p 27018:27017 -d mongodb/mongodb-atlas-local

docker run --name ragneo4j_mmlongbench --hostname localhost --userns=host --user root -p 7474:7474 -p 7687:7687 -v ./database/raganything_db/neo4j_mmlongbench:/data -v ./database/raganything_db/neo4j_plugins:/plugins -e NEO4J_AUTH=none -e NEO4J_PLUGINS='["apoc"]' -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true -e NEO4J_apoc_import_file_use__neo4j__config=true -e NEO4J_dbms_security_procedures_unrestricted=apoc.\\\* neo4j

docker run --name ragneo4j_doc_bench --hostname localhost --userns=host --user root -p 7475:7474 -p 7688:7687 -v ./database/raganything_db/neo4j_docbench:/data -v ./database/raganything_db/neo4j_plugins:/plugins -e NEO4J_AUTH=none -e NEO4J_PLUGINS='["apoc"]' -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true -e NEO4J_apoc_import_file_use__neo4j__config=true -e NEO4J_dbms_security_procedures_unrestricted=apoc.\\\* neo4j
```

### Parse documents and add to vector database

We use [MinerU](https://github.com/opendatalab/MinerU) to parse documents. You can download the models or using vLLM by environment variables.

```bash
export MINERU_DEVICE_MODE=""
export MINERU_VIRTUAL_VRAM_SIZE=12
export MINERU_MODEL_SOURCE=local
export PARSE_RESULT_DIR="./storge/mineru"
export MINERU_TOOLS_CONFIG_JSON="./models/mineru.json"
export MINERU_TABLE_ENABLE=false
export MINERU_BACKEND="pipeline"    # vlm-http-client pipeline
export MINERU_SERVER_URL="https://example.com/"
```

You can using following scripts to construct database from PDFs.

```bash
#!/usr/bin/bash

export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export RERANK_MODEL_NAME="bge-reranker-v2-m3"

export TEXT_DATABASE="./database/text_db_mmlongbench"
export IMAGE_DATABASE="./database/image_db_mmlongbench"
export SHORTCUT_DATABASE="./database/shortcut_db_mmlongbench"

export MODEL_PA_BY_KG=true
export SHORT_CUT=true

export MONGO_URI="mongodb://admin:admin@localhost:27017/?authSource=admin&directConnection=true"
export NEO4J_URI="neo4j://localhost:7687"

export TEST_BENCH_FULL=true
export TEST_RAG=false
export IF_ASK=false
export ADD_VECTOR=true
export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=DEBUG

export ADD_VECTOR_TEXT_BATCH_SIZE=150
export ADD_VECTOR_IMAGE_BATCH_SIZE=75
export ADD_KG_PDF_BATCH_SIZE=100

export ENABLE_THINK=false

export OPENAI_LLM_MAX_TOKENS=4096
export RAGANYTHING_LLM_MODEL_MAX_ASYNC=50
export SUMMARY_CONTEXT_SIZE=2048

conda run -n bayesrag --live-stream python ./eval/mmLongBench/test_mmLongBench.py
```

```bash
#!/usr/bin/bash

export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export RERANK_MODEL_NAME="bge-reranker-v2-m3"

export TEXT_DATABASE="./database/text_db_docbench"
export IMAGE_DATABASE="./database/image_db_docbench"
export SHORTCUT_DATABASE="./database/shortcut_db_docbench"

export MODEL_PA_BY_KG=true
export SHORT_CUT=true

export MONGO_URI="mongodb://admin:admin@localhost:27018/?authSource=admin&directConnection=true"
export NEO4J_URI="neo4j://localhost:7688"

export TEST_BENCH_FULL=true
export TEST_RAG=false
export IF_ASK=false
export ADD_VECTOR=true
export PRINT_FILE=false
export PRINT_TERMINAL_LEVEL=DEBUG

export ADD_VECTOR_TEXT_BATCH_SIZE=150
export ADD_VECTOR_IMAGE_BATCH_SIZE=75
export ADD_KG_PDF_BATCH_SIZE=100

export ENABLE_THINK=false

export OPENAI_LLM_MAX_TOKENS=4096
export RAGANYTHING_LLM_MODEL_MAX_ASYNC=50
export SUMMARY_CONTEXT_SIZE=2048

conda run -n bayesrag --live-stream python ./eval/DocBench/test_DocBench.py
```

## Evaluation

You can run the following command to evaluate the performance on MMLongBench-Doc or DocBench.

```bash
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

export EVAL_CONCURRENCY_LIMIT=10

export EMBEDDING_MODEL_NAME="Qwen3-Embedding-4B"
export RERANK_MODEL_NAME="vllm-bge-reranker-v2-m3"

export TEXT_DATABASE="./database/text_db_mmlongbench"
export IMAGE_DATABASE="./database/image_db_mmlongbench"
export SHORTCUT_DATABASE="./database/shortcut_db_mmlongbench"
export MONGO_URI="mongodb://admin:admin@localhost:27017/?authSource=admin&directConnection=true"
export NEO4J_URI="neo4j://localhost:7687"
# or docbench
# export TEXT_DATABASE="./database/text_db_docbench"
# export IMAGE_DATABASE="./database/image_db_docbench"
# export SHORTCUT_DATABASE="./database/shortcut_db_docbench"
# export MONGO_URI="mongodb://admin:admin@localhost:27018/?authSource=admin&directConnection=true"
# export NEO4J_URI="neo4j://localhost:7688"

export BAYES=true   # or just use text reranker
export DShafer=true # or calculate P(B|A) by linear weight
export MODEL_PA_BY_KG=true  # or calculate P(A) by bbox
export MODEL_PA_BY_KG_WEIGHT=true   # or calculate P(A) by kg relations' frequency
export SHORT_CUT=true
export RERANK=true

export BM25_K=20
export TEXT_K=512
export IMAGE_K=512
export SHORT_CUT_K=512

export RERANK_TOP_K_FOR_BAYES=100   # rerank in retriever
export COMB_TOP_K_THRESHOLD=15
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
export D_S_ALPHA=0.7 # embedding confidence
export D_S_BETA=0.6  # embedding confidence of misclassification
export SCALING_FACTOR=0.1   # P(A) by KG, probability
### calculate P(B|A) by weight
export WEIGHT_TEXT_VECTOR=0.4
export WEIGHT_IMAGE=0.3
export WEIGHT_SHOERCUT=0.3
export RAGANYTHING_RERANK=false

export RERANK_THRESHOLD=0.1

export GENERATE_MODEL_NAME="gpt-4o-mini"  # azure-gpt-4o
export OPENAI_BASE_URL=""
export OPENAI_API_KEY=""
export JUDGE_AGENT_MODEL_NAME=""
export JUDGE_AGENT_BASE_URL=""
export JUDGE_AGENT_API_KEY=""

export WRITE_RESULTS=true
export RESULT_DIR_NAME="gpt_4o_mini_Bench_results"

conda run -n bayesrag --live-stream python ./eval/mmLongBench/test_mmLongBench.py
# or
# conda run -n bayesrag --live-stream python ./eval/DocBench/test_DocBench.py
```

Then, you can run the following command to evaluate the performance:

```bash
python ./eval/count_result/count_docbench.py
python ./eval/count_result/count_mmlongbench.py
```
