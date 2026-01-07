#!/usr/bin/bash

conda create -n bayesrag python=3.10  -y
conda activate bayesrag

conda run -n bayesrag --live-stream pip install git+https://github.com/illuin-tech/colpali.git

# langchain
conda run -n bayesrag --live-stream pip install -U langchain langchain-community langchain-text-splitters langchain-core langchain-experimental langchain_huggingface langchain_chroma langchain_openai

conda run -n bayesrag --live-stream pip install mineru open_clip_torch
conda run -n bayesrag --live-stream pip install pydantic lxml matplotlib tiktoken pypdf PyMuPDF pymupdf pillow rank_bm25 'raganything[all]' nest_asyncio pdf2image traceloop-sdk langfuse
conda run -n bayesrag --live-stream pip install sentence_transformers vllm
# flashinfer-python