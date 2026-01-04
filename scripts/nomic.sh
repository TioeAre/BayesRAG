#!/usr/bin/bash

export http_proxy=127.0.0.1:61110
export https_proxy=127.0.0.1:61110

conda create -n nomic python=3.10  -y
conda activate nomic

conda run -n nomic --live-stream pip install git+https://github.com/illuin-tech/colpali.git

# langchain
conda run -n nomic --live-stream pip install -U langchain langchain-community langchain-text-splitters langchain-core langchain-experimental langchain_huggingface langchain_chroma langchain_openai

conda run -n nomic --live-stream pip install mineru[core] traceloop-sdk langfuse open_clip_torch pydantic lxml matplotlib tiktoken pypdf PyMuPDF pymupdf pillow sentence_transformers rank_bm25 vllm 'raganything[all]' nest_asyncio pdf2image
# flashinfer-python