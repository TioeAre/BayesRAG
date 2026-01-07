import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config
from src.utils.gpt import GPT
from src.langchain.models.models import RAGModels
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, Tuple, Optional
import asyncio, traceback, pickle
import base64
import datetime, time
import re
import copy
import json
import uuid


def print_config():
    try:
        config_details = ["\n" + "=" * 20 + " Project Configuration " + "=" * 20]
        for key, value in vars(project_config).items():
            if key.isupper() and not key.startswith("__"):
                if "KEY" in key or "SECRET" in key or "AUTH" in key:
                    value = f"***{str(value)[-4:]}"
                config_details.append(f"    {key:<30}: {value}")
        config_details.append("=" * 63 + "\n")
        logger.debug("\n".join(config_details))
    except Exception as e:
        logger.error(f"Failed to log project configuration: {e}")


def filter_doc_id(result, prefix, doc_id):
    provenances = result.get("predict", {}).get("provenance", {}).get(prefix, [])
    page_idxs = []
    for provenance in provenances:
        pre_doc_id = provenance.get("doc_id", "")
        if doc_id in pre_doc_id:  # NOTE text, image 的 doc_id 是完整路径, shortcut 的 doc_id 是文件名
            page_idxs.append(provenance.get("page_idx+1"))
    return set(page_idxs)


async def cal_recall(result: dict, models: RAGModels):
    gt_provenance_set = set(result.get("qa", {}).get("provenance", []))
    total_relevant = len(gt_provenance_set)
    if total_relevant == 0:
        return {"text_recall": 1.0, "image_recall": 1.0, "shortcut_recall": 1.0}

    doc_id = result.get("qa", {}).get("id", "")

    pred_text_set = filter_doc_id(result, "text", doc_id)
    pred_image_set = filter_doc_id(result, "image", doc_id)
    pred_shortcut_set = filter_doc_id(result, "shortcut", doc_id)

    before_rerank_provenances = result.get("predict", {}).get("provenance", {}).get("before_rerank", [])
    before_rerank_list = []
    for before_rerank_provenance in before_rerank_provenances:
        pre_doc_id = before_rerank_provenance.get("doc_id")
        if doc_id in pre_doc_id:
            before_rerank_list.append(before_rerank_provenance.get("page_idx+1"))
    ranking_positions = [index for index, page_idx in enumerate(before_rerank_list) if page_idx in gt_provenance_set]
    tp_before_rerank = len(set(before_rerank_list).intersection(gt_provenance_set))

    tp_text = len(gt_provenance_set.intersection(pred_text_set))
    tp_image = len(gt_provenance_set.intersection(pred_image_set))
    tp_shortcut = len(gt_provenance_set.intersection(pred_shortcut_set))

    # 整体召回率
    pred_combined_set = pred_text_set.union(pred_image_set).union(pred_shortcut_set)
    total_tp_overall = len(gt_provenance_set.intersection(pred_combined_set))

    recall_scores = {
        "text_recall": tp_text / total_relevant,
        "image_recall": tp_image / total_relevant,
        "shortcut_recall": tp_shortcut / total_relevant,
        "overall_recall": total_tp_overall / total_relevant,
        "before_rerank": tp_before_rerank / total_relevant,
        "ranking_positions": ranking_positions,
    }
    result["recall"] = recall_scores
    logger.info(f"recall: {recall_scores}")
    if project_config.BAYES and project_config.MODEL_PA_BY_KG:
        await cal_single_kg_recall(
            result=result, models=models, vec_overall_uuid_set=pred_combined_set
        )  # inside function, result["kg_recall"] = scores
        logger.info(f"kg recall: {result['kg_recall']}")


def extract_pages_from_docs(docs, target_doc_id):
    pages = set()
    if not docs:
        return pages
    for doc in docs:
        if doc is None:
            continue
        source_path = doc.metadata.get("source", "")
        if Path(source_path).name == target_doc_id:
            try:
                page_idx = int(doc.metadata.get("page_idx", -1)) + 1
                pages.add(page_idx)
            except (ValueError, TypeError):
                continue
    return pages


def calculate_recall(retrieved_set, gt_set):
    if not gt_set:
        return 0.0
    hit_count = len(retrieved_set.intersection(gt_set))
    return hit_count / len(gt_set)


async def cal_single_kg_recall(result: dict, models: RAGModels, vec_overall_uuid_set: set = set()):
    gt_provenance_set = set(result.get("qa", {}).get("provenance", []))
    doc_id = result.get("qa", {}).get("id", "")
    connectivity_map = result.get("middle_results", {}).get("connectivity_map", {})
    if not connectivity_map:
        return 0.0, 0.0, 0.0, 0.0
    uuid_list = list(connectivity_map.keys())
    if not uuid_list:
        return 0.0, 0.0, 0.0, 0.0
    tasks = [
        models.text_embedding.vectorstore_embd.aget_by_ids(uuid_list),
        models.image_embedding.vectorstore_embd.aget_by_ids(uuid_list),
        models.shortcut_embedding.vectorstore_embd.aget_by_ids(uuid_list),
    ]
    text_kg_docs, image_kg_docs, shortcut_kg_docs = await asyncio.gather(*tasks)
    text_pages = extract_pages_from_docs(text_kg_docs, doc_id)
    image_pages = extract_pages_from_docs(image_kg_docs, doc_id)
    shortcut_pages = extract_pages_from_docs(shortcut_kg_docs, doc_id)
    combined_pages = text_pages.union(image_pages).union(shortcut_pages)

    vec_kg_pages = combined_pages.union(vec_overall_uuid_set)

    scores = {
        "text_recall": calculate_recall(text_pages, gt_provenance_set),
        "image_recall": calculate_recall(image_pages, gt_provenance_set),
        "shortcut_recall": calculate_recall(shortcut_pages, gt_provenance_set),
        "overall_recall": calculate_recall(combined_pages, gt_provenance_set),
        "vec_kg_overall_recall": calculate_recall(vec_kg_pages, gt_provenance_set),
    }
    result["kg_recall"] = scores
    return (
        scores["text_recall"],
        scores["image_recall"],
        scores["shortcut_recall"],
        scores["overall_recall"],
    )


async def _analysis_kg_retriever_results(results: list, models: RAGModels):
    total_text_recall = 0.0
    total_image_recall = 0.0
    total_shortcut_recall = 0.0
    total_combined_recall = 0.0
    num_results = len(results) if results else 0

    all_scores = await asyncio.gather(*(cal_single_kg_recall(r, models) for r in results))

    num_results = len(results)
    sums = [sum(metric) for metric in zip(*all_scores)]

    overall_scores = {
        "text_recall": sums[0] / num_results,
        "image_recall": sums[1] / num_results,
        "shortcut_recall": sums[2] / num_results,
        "overall_recall": sums[3] / num_results,
    }

    logger.info(f"KG provenance recall: {overall_scores}")


async def _retriever(models: RAGModels, question, result):

    # question = f"Question about {result['qa']['id']}: {result['qa']['question']}"
    start_time = time.time()
    if "time_record" not in result:
        result["time_record"] = {}

    retriever_results = await models.retriever(
        question, if_shortcut=project_config.SHORT_CUT, start_time=start_time, result=result
    )

    retrieval_end_time = time.time()

    result["middle_results"]["retriever_results"] = models.retriever_results_to_uuid_dict(retriever_results)

    rerank_start_time = time.time()

    if project_config.BAYES:
        merged_results = models.bayes_model.merge_retriever_results(retriever_results)  # cannot delete
        # all_combinations = models.bayes_model.generate_all_combinations(merged_results)
        all_combinations = await models.bayes_model.generate_all_combinations_from_retri(
            retriever_results, kg=models.kg, query=question, result=result
        )
        reranked_retriever_results = models.bayes_model.get_top_k_ab_scores(all_combinations)
    else:
        reranked_retriever_results = await models.model_rerank(question, retriever_results, result)

    rerank_end_time = time.time()

    result["time_record"]["start_time"] = start_time
    result["time_record"]["retriever_cost"] = round(retrieval_end_time - start_time, 4)
    result["time_record"]["rerank_cost"] = round(rerank_end_time - rerank_start_time, 4)

    return reranked_retriever_results


async def _ask(models: RAGModels, messages) -> Tuple[str, dict]:
    predict_answer, token_usage = await models.ask(messages=messages)
    return predict_answer, token_usage
