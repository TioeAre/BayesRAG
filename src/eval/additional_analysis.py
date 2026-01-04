import os, sys, copy, pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config
from src.langchain.models.models import RAGModels
from src.eval.eval import cal_recall
from typing import Union, Tuple, Optional, List, Dict
from loguru import logger
from pathlib import Path


async def addi_get_recall_middle_result(path, models: RAGModels, top_k=20):
    result = pickle.loads(Path(path).read_bytes())
    # copied_retriever_results = copy.deepcopy(result["middle_results"]["retriever_results"])
    copied_retriever_results = await models.uuid_dict_to_retriever_results(
        result["middle_results"]["retriever_results"]
    )
    result["middle_results"] = {
        "retriever_results": {
            "text_results": copied_retriever_results["text_results"][:top_k],
            "image_results": copied_retriever_results["image_results"][:top_k],
            "shortcut_results": copied_retriever_results["shortcut_results"][:top_k],
            "bm25_results": copied_retriever_results["bm25_results"][:top_k],
        },
    }
    return result


def addi_filter_doc_id(provenances: List, doc_id):
    # provenances = [{"doc_id": "xxx.pdf", "page_idx+1": 132}]
    page_idxs = []
    for provenance in provenances:
        pre_doc_id = provenance.get("doc_id", "")
        if doc_id in pre_doc_id:  # NOTE text, image 的 doc_id 是完整路径, shortcut 的 doc_id 是文件名
            page_idxs.append(provenance.get("page_idx+1"))
    return set(page_idxs)


async def addi_cal_recall(
    ori_result: Dict,
    models: RAGModels,
):
    # retriever_results = copy.deepcopy(ori_result["middle_results"]["retriever_results"])
    retriever_results = await models.uuid_dict_to_retriever_results(ori_result["middle_results"]["retriever_results"])
    gt_provenance_set = set(ori_result.get("qa", {}).get("provenance", []))
    total_relevant = len(gt_provenance_set)
    if total_relevant == 0:
        return {
            "text_recall": 1.0,
            "image_recall": 1.0,
            "shortcut_recall": 1.0,
            "overall_recall": 1.0,
        }, False

    text_results = retriever_results["text_results"]
    image_results = retriever_results["image_results"]
    shortcut_results = retriever_results["shortcut_results"]

    def get_provenances(retriever_results: List):
        provenances = []
        for retriever_result in retriever_results:
            provenance = {
                "doc_id": retriever_result["result"].metadata["source"],
                "page_idx+1": retriever_result["result"].metadata["page_idx"] + 1,
            }
            provenances.append(provenance)
        return provenances

    text_provenances = get_provenances(text_results)
    image_provenances = get_provenances(image_results)
    shortcut_provenances = get_provenances(shortcut_results)

    doc_id = ori_result.get("qa", {}).get("id", "")

    pred_text_set = addi_filter_doc_id(text_provenances, doc_id)
    pred_image_set = addi_filter_doc_id(image_provenances, doc_id)
    pred_shortcut_set = addi_filter_doc_id(shortcut_provenances, doc_id)

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
    }
    # logger.info(recall_scores)
    return recall_scores, True


async def analysis_recall(results: List[Dict], models: RAGModels):
    # provenance recall
    total_text_recall = 0
    total_image_recall = 0
    total_shortcut_recall = 0
    total_overall_recall = 0

    total_num = 0
    for result in results:

        recalls, answerable = await addi_cal_recall(result, models)
        if not answerable:
            continue
        total_num += 1
        total_text_recall += recalls["text_recall"]
        total_image_recall += recalls["image_recall"]
        total_shortcut_recall += recalls["shortcut_recall"]
        total_overall_recall += recalls["overall_recall"]

    if total_num == 0:
        logger.info("Provenance recall: {}".format({"text_recall": 1.0, "image_recall": 1.0, "shortcut_recall": 1.0}))
    overall_scores = {
        "text_recall": total_text_recall / total_num,
        "image_recall": total_image_recall / total_num,
        "shortcut_recall": total_shortcut_recall / total_num,
        "overall_recall": total_overall_recall / total_num,
    }

    logger.info("Provenance recall: {}".format(overall_scores))


async def addi_get_provenance_middle_result(path, models: RAGModels, top_k=20):
    result = pickle.loads(Path(path).read_bytes())
    question = result["qa"]["question"]
    # copied_middle_result = result["middle_results"]["retriever_results"]
    copied_middle_result = await models.uuid_dict_to_retriever_results(result["middle_results"]["retriever_results"])

    merged_results = models.bayes_model.merge_retriever_results(copied_middle_result)  # cannot delete
    # all_combinations = models.bayes_model.generate_all_combinations(merged_results)
    all_combinations = await models.bayes_model.generate_all_combinations_from_retri(
        copied_middle_result,
        kg=models.kg,
        query=question,
        result=result,
        connectivity_map=result["middle_results"]["connectivity_map"],
    )
    reranked_retriever_results = models.bayes_model.get_top_k_ab_scores(all_combinations)

    result["predict"]["provenance"] = {
        "text": [],
        "image": [],
        "shortcut": [],
    }

    def _make_provenance(
        reranked_retriever_results,
        result=result,
        if_shortcut=project_config.SHORT_CUT,
    ):
        ### text knowledge base
        for doc_id in reranked_retriever_results.keys():
            for page_idx in reranked_retriever_results[doc_id].keys():
                for text_result in reranked_retriever_results[doc_id][page_idx]["text_results"]:
                    result["predict"]["provenance"]["text"].append(
                        {
                            "doc_id": Path(text_result["result"].metadata["source"]).name,
                            "page_idx+1": int(text_result["result"].metadata["page_idx"]) + 1,
                        }
                    )
                ### image knowledge base
                for image_result in reranked_retriever_results[doc_id][page_idx]["image_results"]:
                    encoded_image = image_result["result"]
                    result["predict"]["provenance"]["image"].append(
                        {
                            "doc_id": Path(encoded_image.metadata["source"]).name,
                            "page_idx+1": int(encoded_image.metadata["page_idx"]) + 1,
                        }
                    )
                if if_shortcut:
                    for shortcut_result in reranked_retriever_results[doc_id][page_idx]["shortcut_results"]:
                        encoded_image = shortcut_result["result"]
                        result["predict"]["provenance"]["shortcut"].append(
                            {
                                "doc_id": Path(encoded_image.metadata["source"]).name,
                                "page_idx+1": int(encoded_image.metadata["page_idx"]) + 1,
                            }
                        )

    _make_provenance(reranked_retriever_results, result)

    # result["middle_results"] = {"connectivity_map": copied_middle_result["connectivity_map"]}
    result["middle_results"] = {}

    await cal_recall(result, models)

    return result


def analysis_provenance(results: List[Dict]):
    # provenance recall
    total_text_recall = 0
    total_image_recall = 0
    total_shortcut_recall = 0
    total_before_rerank = 0
    total_overall_recall = 0

    total_kg_text_recall = 0
    total_kg_image_recall = 0
    total_kg_shortcut_recall = 0
    total_kg_overall_recall = 0
    total_vec_kg_overall_recall = 0

    total_num = 0
    for result in results:
        gt_provenance_set = set(result.get("qa", {}).get("provenance", []))
        if "recall" not in result.keys() or len(gt_provenance_set) == 0:
            continue

        total_num += 1
        total_text_recall += result["recall"]["text_recall"]
        total_image_recall += result["recall"]["image_recall"]
        total_shortcut_recall += result["recall"]["shortcut_recall"]
        total_before_rerank += result["recall"]["before_rerank"]
        total_overall_recall += result["recall"]["overall_recall"]

        if project_config.BAYES and project_config.MODEL_PA_BY_KG:
            if "kg_recall" not in result.keys():
                continue
            total_kg_text_recall += result["kg_recall"]["text_recall"]
            total_kg_image_recall += result["kg_recall"]["image_recall"]
            total_kg_shortcut_recall += result["kg_recall"]["shortcut_recall"]
            total_kg_overall_recall += result["kg_recall"]["overall_recall"]
            total_vec_kg_overall_recall += result["kg_recall"]["vec_kg_overall_recall"]

        if "before_rerank" in result["predict"]["provenance"]:
            del result["predict"]["provenance"]["before_rerank"]
    if total_num == 0:
        return {"text_recall": 1.0, "image_recall": 1.0, "shortcut_recall": 1.0}
    overall_scores = {
        "text_recall": total_text_recall / total_num,
        "image_recall": total_image_recall / total_num,
        "shortcut_recall": total_shortcut_recall / total_num,
        "overall_recall": total_overall_recall / total_num,
        "before_rerank": total_before_rerank / total_num,
    }
    if project_config.BAYES and project_config.MODEL_PA_BY_KG:
        kg_overall_scores = {
            "text_recall": total_kg_text_recall / total_num,
            "image_recall": total_kg_image_recall / total_num,
            "shortcut_recall": total_kg_shortcut_recall / total_num,
            "overall_recall": total_kg_overall_recall / total_num,
            "vec_kg_overall_recall": total_vec_kg_overall_recall / total_num,
        }
    logger.info("Provenance recall: {}".format(overall_scores))
    if project_config.BAYES and project_config.MODEL_PA_BY_KG:
        logger.info("KG Provenance recall: {}".format(kg_overall_scores))
