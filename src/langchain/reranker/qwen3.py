import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config
from langchain_core.documents import Document
from langchain_community.cross_encoders.base import BaseCrossEncoder  # type: ignore
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Any, Dict, List, Tuple
import requests
from loguru import logger

# from traceloop.sdk.decorators import workflow, task


class Qwen3CrossEncoder(BaseModel, BaseCrossEncoder):
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
    model_name: str = "Qwen/Qwen3-Reranker-4B"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=10)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        model_path = f"{project_config.DATA_ROOT}/models/{project_config.RERANK_MODEL_NAME}"

        if project_config.DEBUG:
            torch.cuda.reset_peak_memory_stats(device=f"{project_config.RERANK_CUDA_DEVICE}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=f"{project_config.RERANK_CUDA_DEVICE}"
        ).eval()
        if project_config.DEBUG:
            peak_mem_gb = torch.cuda.max_memory_allocated(device=f"{project_config.RERANK_CUDA_DEVICE}") / (1024**3)
            logger.debug(f"峰值显存占用 (on {project_config.RERANK_CUDA_DEVICE}): {peak_mem_gb:.2f} GB")
            current_mem_gb = torch.cuda.memory_allocated(device=f"{project_config.RERANK_CUDA_DEVICE}") / (1024**3)
            logger.debug(f"当前显存占用 (on {project_config.RERANK_CUDA_DEVICE}): {current_mem_gb:.2f} GB")

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.batch_size = project_config.RERANK_BATCHSIZE

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)

        if project_config.DEBUG:
            total_model_tokens = inputs["input_ids"].numel()
            batch_shape = list(inputs["input_ids"].shape)  # [batch_size, sequence_length]
            # logger.debug(f"tensor shape: {batch_shape}")
            # logger.debug(f"total tokens in model: {total_model_tokens}")

        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute similarity scores using a HuggingFace transformer model.

        Args:
            text_pairs: The list of text text_pairs to score the similarity. [[query, doc]]

        Returns:
            List of scores, one for each pair.
        """
        # task = "Given a query, retrieve relevant passages that answer the query"
        # pairs = [self.format_instruction(instruction=task, query=query, doc=doc) for query, doc in text_pairs]
        # inputs = self.process_inputs(pairs)
        # scores = self.compute_logits(inputs)
        # return scores
        # processing in batch to avoid OOM
        task = "Given a query, retrieve relevant passages that answer the query"
        indexed_pairs = list(enumerate(text_pairs))  # [(0, (qestion, doc_str))]
        sorted_indexed_pairs = sorted(indexed_pairs, key=lambda item: len(item[1][1]))  # to reduce memory usage
        final_scores = [0.0] * len(text_pairs)
        with torch.no_grad():
            for i in range(0, len(sorted_indexed_pairs), self.batch_size):
                batch_data = sorted_indexed_pairs[i : i + self.batch_size]
                original_indices = [item[0] for item in batch_data]
                batch_pairs_tuples = [item[1] for item in batch_data]
                pairs = [self.format_instruction(task, query, doc) for query, doc in batch_pairs_tuples]
                inputs = self.process_inputs(pairs)
                scores = self.compute_logits(inputs)
                for idx, score in zip(original_indices, scores):
                    final_scores[idx] = score
        return final_scores

    # @task(name="get_top_k_results")
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
        # scores = self.score(text_pairs)
        # scored_results = list(zip(text_results, scores))
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
