import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config
from langchain_core.documents import Document
from langchain_community.cross_encoders.base import BaseCrossEncoder  # type: ignore
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Tuple
from loguru import logger
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BgeCrossEncoder(BaseModel, BaseCrossEncoder):
    client: Any = None  #: :meta private:
    tokenizer: Any = None
    token_false_id: Any = None
    token_true_id: Any = None
    max_length: Any = None
    prefix: Any = None
    suffix: Any = None
    prefix_tokens: Any = None
    suffix_tokens: Any = None
    model: Any = None
    model_name: str = "BAAI/bge-reranker-v2-m3"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=10)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        model_path = os.getenv(
            "RERANK_MODEL_PATH", f"{project_config.DATA_ROOT}/models/{project_config.RERANK_MODEL_NAME}"
        )

        if project_config.DEBUG:
            torch.cuda.reset_peak_memory_stats(device=f"{project_config.RERANK_CUDA_DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, device_map=f"{project_config.RERANK_CUDA_DEVICE}"
        ).eval()
        if project_config.DEBUG:
            peak_mem_gb = torch.cuda.max_memory_allocated(device=f"{project_config.RERANK_CUDA_DEVICE}") / (1024**3)
            logger.debug(f"峰值显存占用 (on {project_config.RERANK_CUDA_DEVICE}): {peak_mem_gb:.2f} GB")
            current_mem_gb = torch.cuda.memory_allocated(device=f"{project_config.RERANK_CUDA_DEVICE}") / (1024**3)
            logger.debug(f"当前显存占用 (on {project_config.RERANK_CUDA_DEVICE}): {current_mem_gb:.2f} GB")
        self.batch_size = project_config.RERANK_BATCHSIZE

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute similarity scores using a HuggingFace transformer model.

        Args:
            text_pairs: The list of text text_pairs to score the similarity. [[query, doc]]

        Returns:
            List of scores, one for each pair.
        """
        indexed_pairs = list(enumerate(text_pairs))  # [(0, (qestion, doc_str))]
        sorted_indexed_pairs = sorted(indexed_pairs, key=lambda item: len(item[1][1]))  # find the max tokens in docs
        final_scores = [0.0] * len(text_pairs)
        with torch.no_grad():
            for i in range(0, len(sorted_indexed_pairs), self.batch_size):
                batch_data = sorted_indexed_pairs[i : i + self.batch_size]
                original_indices = [item[0] for item in batch_data]
                batch_pairs_tuples = [item[1] for item in batch_data]
                inputs = self.tokenizer(
                    batch_pairs_tuples, padding=True, truncation=True, return_tensors="pt", max_length=4096
                ).to(self.model.device)
                batch_scores = (
                    self.model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                    .to("cpu")
                )
                for idx, score in zip(original_indices, batch_scores):
                    final_scores[idx] = score
            return final_scores

    def get_top_k_results(self, question: str, text_results: List[Document], k: int) -> List[Tuple[Document, float]]:

        # if project_config.DEBUG:
        #     question_token_count = len(self.tokenizer.encode(question, add_special_tokens=False))
        #     total_doc_tokens = 0
        #     for doc in text_results:
        #         total_doc_tokens += len(self.tokenizer.encode(doc.page_content, add_special_tokens=False))
        #     total_original_tokens = question_token_count + total_doc_tokens
        #     logger.debug(f"question tokens: {question_token_count}")
        #     logger.debug(f"docs tokens: {total_doc_tokens}")
        #     logger.debug(f"question + docs: {total_original_tokens}")

        # text_pairs = [(question, doc.page_content) for doc in text_results]

        # if not text_pairs:
        #     return []

        # scores = self.score(text_pairs)
        # scored_results = list(zip(text_results, scores))    # scores
        # sorted_results = sorted(scored_results, key=lambda item: item[1], reverse=True)
        sorted_results = self.get_sorted_results_with_score(question, text_results)
        top_k_results = sorted_results[:k]
        return top_k_results

    def get_sorted_results_with_score(
        self, question: str, text_results: List[Document]
    ) -> List[Tuple[Document, float]]:

        if project_config.DEBUG:
            question_token_count = len(self.tokenizer.encode(question, add_special_tokens=False))
            total_doc_tokens = 0
            for doc in text_results:
                total_doc_tokens += len(self.tokenizer.encode(doc.page_content, add_special_tokens=False))
            total_original_tokens = question_token_count + total_doc_tokens
            # logger.debug(f"question tokens: {question_token_count}")
            # logger.debug(f"docs tokens: {total_doc_tokens}")
            # logger.debug(f"question + docs: {total_original_tokens}")

        text_pairs = [(question, doc.page_content) for doc in text_results]
        scores = self.score(text_pairs)
        scored_results = list(zip(text_results, scores))
        sorted_results = sorted(scored_results, key=lambda item: item[1], reverse=True)

        if project_config.DEBUG:
            if sorted_results:
                logger.debug(f"Top Rerank score: {sorted_results[0][1]}")

        return sorted_results

        # inputs = self.tokenizer(text_pairs, padding=True, truncation=True, return_tensors='pt', max_length=4096).to(self.model.device)
        # scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().to("cpu")
        # # scores = self.score(text_pairs)
        # scored_results = list(zip(text_results, scores))
        # sorted_results = sorted(scored_results, key=lambda item: item[1], reverse=True)
        # top_k_results = sorted_results[:k]
        # return top_k_results
