import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config
from langchain_core.documents import Document
from langchain_community.cross_encoders.base import BaseCrossEncoder  # type: ignore
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Tuple
import requests
from loguru import logger
from transformers import AutoTokenizer

# from traceloop.sdk.decorators import workflow, task
tokenizer = AutoTokenizer.from_pretrained(
    os.getenv("RERANK_MODEL_PATH", f"{project_config.DATA_ROOT}/models/{project_config.RERANK_MODEL_NAME}")
)


class VLLMRemoteCrossEncoder(BaseModel, BaseCrossEncoder):
    client: Any = None
    model_name: str = (
        project_config.RERANK_MODEL_NAME.split("vllm-")[1]
        if project_config.RERANK_MODEL_NAME.startswith("vllm")
        else project_config.RERANK_MODEL_NAME
    )
    # api_url: str = "http://localhost:12502/v1/rerank"
    api_url: str = project_config.RERANKER_BASE_URL_VEC
    api_key: str = "None"
    batch_size: int = project_config.RERANK_BATCHSIZE
    timeout: int = 600

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if hasattr(project_config, "RERANK_MODEL_NAME"):
            self.model_name = (
                project_config.RERANK_MODEL_NAME.split("vllm-")[1]
                if project_config.RERANK_MODEL_NAME.startswith("vllm")
                else project_config.RERANK_MODEL_NAME
            )
        self.batch_size = project_config.RERANK_BATCHSIZE
        logger.info(f"Initialized VLLM Remote Reranker at {self.api_url} with model {self.model_name}")

    def get_token_count(self, text: str) -> int:
        if not text:
            return 0
        if tokenizer:
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 3

    def truncate_to_token_limit(self, text: str, limit: int = 8000) -> str:
        if not tokenizer:
            return text[: limit * 3]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= limit:
            return text
        return tokenizer.decode(tokens[:limit])

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        if not text_pairs:
            return []

        query = text_pairs[0][0]
        raw_documents = [pair[1] for pair in text_pairs]
        all_scores = []

        TOKEN_LIMIT = 8192
        batches = []
        current_batch = []
        current_batch_tokens = 0

        safe_limit = TOKEN_LIMIT - self.get_token_count(query) - 200

        for doc in raw_documents:
            doc_tokens = self.get_token_count(doc)
            if doc_tokens > safe_limit:
                logger.warning(f"Single document exceeds {safe_limit} tokens. Truncating.")
                doc = self.truncate_to_token_limit(doc, safe_limit)
                doc_tokens = safe_limit
            if current_batch and (
                current_batch_tokens + doc_tokens > safe_limit or len(current_batch) >= self.batch_size
            ):
                batches.append(current_batch)
                current_batch = [doc]
                current_batch_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_batch_tokens += doc_tokens

        if current_batch:
            batches.append(current_batch)

        for batch_docs in batches:
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": batch_docs,
                "top_n": len(batch_docs),
            }

            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    logger.error(f"Reranker API Error: {response.status_code} | {response.text}")
                    response.raise_for_status()

                result_json = response.json()

                if "results" in result_json:
                    sorted_res = sorted(result_json["results"], key=lambda x: x["index"])
                    batch_scores = [item["relevance_score"] for item in sorted_res]
                    all_scores.extend(batch_scores)
                else:
                    logger.error(f"Unexpected API response format: {result_json}")

            except Exception as e:
                logger.error(f"Failed to process batch in reranker: {e}")
                raise e

        return all_scores

    def get_sorted_results_with_score(
        self, question: str, text_results: List[Document]
    ) -> List[Tuple[Document, float]]:
        if not text_results:
            return []

        if project_config.DEBUG:
            logger.debug(f"Reranking {len(text_results)} documents for query: {question[:50]}...")
        text_pairs = [(question, doc.page_content) for doc in text_results]
        scores = self.score(text_pairs)

        if len(scores) != len(text_results):
            logger.warning(f"Scores count ({len(scores)}) mismatch docs count ({len(text_results)}).")
            min_len = min(len(scores), len(text_results))
            scores = scores[:min_len]
            text_results = text_results[:min_len]
        scored_results = list(zip(text_results, scores))
        sorted_results = sorted(scored_results, key=lambda item: item[1], reverse=True)

        if project_config.DEBUG:
            if sorted_results:
                logger.debug(f"Top Rerank score: {sorted_results[0][1]}")

        return sorted_results
