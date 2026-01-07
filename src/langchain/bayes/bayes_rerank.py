import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config

# isort:skip
import copy, math, json
import numpy as np
from pathlib import Path
from langchain_core.documents import Document
from src.langchain.utils.image import is_base64
from src.utils.gpt import GPT
from src.utils.utils import _run_sync
from src.langchain.utils.uuid import generate_stable_uuid_for_text
from src.langchain.utils.rerank import add_to_combination_results, add_to_final_results
from src.RAGAnything.kg import KG

from loguru import logger
import asyncio
import nest_asyncio

nest_asyncio.apply()


class BayesReranker(object):
    def __init__(self):
        self.max_text_score = 0.0
        self.max_image_score = 0.0
        self.max_shortcut_score = 0.0

        self.min_text_score = 100.0
        self.min_image_score = 100.0
        self.min_shortcut_score = 100.0

    def _piecewise_score_mapping(self, val: float, global_max: float) -> float:
        FOCUS_WINDOW = 0.15
        HIGH_SCORE_BASE = 0.3
        threshold = max(0.5, global_max - FOCUS_WINDOW)
        if val < threshold:
            if threshold <= 0:
                return 0.0
            return (val / threshold) * HIGH_SCORE_BASE
        else:
            local_norm = (val - threshold) / (global_max - threshold)
            local_norm = pow(local_norm, 3.0)
            return HIGH_SCORE_BASE + local_norm * (1.0 - HIGH_SCORE_BASE)

    def _normalize_and_sharpen(self, score, max_score, min_score, gamma=1.0, min_bound=0.7):
        if max_score - min_score < (max_score / 10):
            return 1
        raw_norm = score / max_score
        # norm_score = min_bound + (1.0 - min_bound) * raw_norm
        norm_score = (score) / (max_score)
        sharpened_score = math.pow(norm_score, gamma)
        return sharpened_score

    def cal_bbox_dis(self, doc1: Document, doc2: Document) -> tuple[float, float]:
        # center = (x, y)
        para_bbox1 = json.loads(doc1.metadata["para_bbox"])
        center1 = [
            (float(para_bbox1[0]) + float(para_bbox1[2])) / 2,
            (float(para_bbox1[1]) + float(para_bbox1[3])) / 2,
        ]
        para_bbox2 = json.loads(doc2.metadata["para_bbox"])
        center2 = [
            (float(para_bbox2[0]) + float(para_bbox2[2])) / 2,
            (float(para_bbox2[1]) + float(para_bbox2[3])) / 2,
        ]
        page_num1 = float(doc1.metadata["page_idx"])
        page_num2 = float(doc2.metadata["page_idx"])
        page_diff = page_num2 - page_num1
        page_size = json.loads(doc1.metadata["page_size"])
        page_height = float(page_size[1])
        if page_diff == 0:
            pass
        else:
            # center2 在 center1 的上一页, center1 的 y + page.y*page_diff
            if page_diff < 0:
                center1[1] += abs(page_diff) * page_height
            # center2 在 center1 的下一页, center2 的 y + page.y*page_diff
            else:
                center2[1] += abs(page_diff) * page_height
        distance = math.sqrt(math.pow((center1[0] - center2[0]), 2) + math.pow((center1[1] - center2[1]), 2))
        diagonal_length = math.sqrt(math.pow(float(page_size[0]), 2) + math.pow(float(page_size[1]), 2))
        return distance, diagonal_length

    def cal_BA_score_by_linear_weight(self, text_result: dict, image_result: dict, shortcut_result: dict):
        # return (
        #     project_config.WEIGHT_TEXT_VECTOR * float(text_result["score"])
        #     + project_config.WEIGHT_IMAGE * float(image_result["score"])
        #     + project_config.WEIGHT_SHOERCUT * float(shortcut_result["score"])
        # )
        # return (
        #     project_config.WEIGHT_TEXT_VECTOR
        #     * ((float(text_result["score"]) - self.min_text_score) / (self.max_text_score - self.min_text_score))
        #     + project_config.WEIGHT_IMAGE
        #     * ((float(image_result["score"]) - self.min_image_score) / (self.max_image_score - self.min_image_score))
        #     + project_config.WEIGHT_SHOERCUT
        #     * (
        #         (float(shortcut_result["score"]) - self.min_shortcut_score)
        #         / (self.max_shortcut_score - self.min_shortcut_score)
        #     )
        # )
        return (
            project_config.WEIGHT_TEXT_VECTOR
            * self._normalize_and_sharpen(
                score=text_result["score"], max_score=self.max_text_score, min_score=self.min_text_score
            )
            + project_config.WEIGHT_IMAGE
            * self._normalize_and_sharpen(
                score=image_result["score"], max_score=self.max_image_score, min_score=self.min_image_score
            )
            + project_config.WEIGHT_SHOERCUT
            * self._normalize_and_sharpen(
                score=shortcut_result["score"], max_score=self.max_shortcut_score, min_score=self.min_shortcut_score
            )
        )
        # norm_text = self._piecewise_score_mapping(float(text_result["score"]), self.max_text_score)
        # norm_image = self._piecewise_score_mapping(float(image_result["score"]), self.max_image_score)
        # norm_shortcut = self._piecewise_score_mapping(float(shortcut_result["score"]), self.max_shortcut_score)
        # return (
        #     project_config.WEIGHT_TEXT_VECTOR * norm_text
        #     + project_config.WEIGHT_IMAGE * norm_image
        #     + project_config.WEIGHT_SHOERCUT * norm_shortcut
        # )

    def cal_BA_score_by_Dempster_Shafer(
        self,
        text_result: dict,
        image_result: dict,
        shortcut_result: dict,
        alpha=project_config.D_S_ALPHA,
        beta=project_config.D_S_BETA,
    ):
        scores = [text_result["score"], image_result["score"], shortcut_result["score"]]
        processed_scores = [
            self._normalize_and_sharpen(text_result["score"], self.max_text_score, self.min_text_score),
            # text_result["score"],
            self._normalize_and_sharpen(image_result["score"], self.max_image_score, self.min_image_score),
            self._normalize_and_sharpen(shortcut_result["score"], self.max_shortcut_score, self.min_shortcut_score),
            # shortcut_result["score"],
        ]

        current_m = {"Y": 0.0, "N": 0.0, "Theta": 1.0}
        for score in processed_scores:
            m_y = alpha * score
            m_n = beta * (1 - score)
            m_theta = 1 - m_y - m_n
            m_theta = max(0.0, m_theta)

            new_source = {"Y": m_y, "N": m_n, "Theta": m_theta}
            k = (current_m["Y"] * new_source["N"]) + (current_m["N"] * new_source["Y"])
            if k >= 0.999:
                # extreme conflict, return 0
                return 0.0

            denominator = 1 - k
            # m(Y) = (m1(Y)m2(Y) + m1(Y)m2(T) + m1(T)m2(Y)) / (1-K)
            new_m_y = (
                current_m["Y"] * new_source["Y"]
                + current_m["Y"] * new_source["Theta"]
                + current_m["Theta"] * new_source["Y"]
            ) / denominator
            # m(N) = (m1(N)m2(N) + m1(N)m2(T) + m1(T)m2(N)) / (1-K)
            new_m_n = (
                current_m["N"] * new_source["N"]
                + current_m["N"] * new_source["Theta"]
                + current_m["Theta"] * new_source["N"]
            ) / denominator
            # m(Theta) = (m1(T)m2(T)) / (1-K)
            new_m_theta = (current_m["Theta"] * new_source["Theta"]) / denominator
            current_m = {"Y": new_m_y, "N": new_m_n, "Theta": new_m_theta}
        # BetP(Y) = m(Y) + m(Theta) / 2
        final_score = current_m["Y"] + (current_m["Theta"] / 2.0)
        return final_score

    def cal_A_score_by_bbox(self, text_result: dict, image_result: dict, shortcut_result: dict):
        shortcut_doc_id = shortcut_result["result"].metadata["source"]
        shortcut_page_idx = shortcut_result["result"].metadata["page_idx"]
        text_doc_id = text_result["result"].metadata["source"]
        text_page_idx = text_result["result"].metadata["page_idx"]
        image_doc_id = image_result["result"].metadata["source"]
        image_page_idx = image_result["result"].metadata["page_idx"]
        A_score = project_config.EPSILON
        if shortcut_doc_id == text_doc_id and shortcut_doc_id == image_doc_id:
            bbox_dis, diagonal_length = self.cal_bbox_dis(text_result["result"], image_result["result"])
            # 计算shortcut与text, image之间的页数距离
            text_page_diff = abs(int(text_page_idx) - int(shortcut_page_idx))
            image_page_diff = abs(int(image_page_idx) - int(shortcut_page_idx))
            if (bbox_dis < (project_config.DISTANCE_THRESHOLD * diagonal_length)) and (
                max(text_page_diff, image_page_diff) < project_config.DISTANCE_THRESHOLD
            ):
                A_score = 1.0
        return max(A_score, project_config.EPSILON)

    def cal_A_score_by_kg(
        self, text_result: dict, image_result: dict, shortcut_result: dict, connectivity_map: dict, kg: KG | None = None
    ):
        A_score = 0
        if kg is not None:
            retriever_ids = [
                text_result["result"].metadata["uuid"],
                image_result["result"].metadata["uuid"],
                shortcut_result["result"].metadata["uuid"],
            ]
            A_score = kg.calculate_triplet_coherence(retriever_ids, connectivity_map, project_config.BONUS_WEIGHT)
        # return 1.0
        return max(A_score, project_config.EPSILON)

    def merge_retriever_results(self, retri_results):
        self.max_text_score = 0.0
        self.max_image_score = 0.0
        self.max_shortcut_score = 0.0
        self.min_text_score = 100.0
        self.min_image_score = 100.0
        self.min_shortcut_score = 100.0
        retriever_results = copy.deepcopy(retri_results)
        merged_results = {
            # "document_id": {
            #     "page_idx": {
            #         "text_results": [],
            #         "bm25_results": [],
            #         "image_results": [],
            #         "shortcut_results": [],
            #         "shortcut_score": 0.0,
            #     }
            # }
        }

        def add_to_merged_results(result, prefix):
            document_id = Path(result["result"].metadata["source"]).name
            page_idx = result["result"].metadata["page_idx"]
            if document_id not in merged_results.keys():
                merged_results[document_id] = {}
            if page_idx not in merged_results[document_id].keys():
                merged_results[document_id][page_idx] = {
                    "text_results": [],
                    "bm25_results": [],
                    "image_results": [],
                    "shortcut_results": [],  # 只会有一个或没有
                    "shortcut_score": 0.0,
                }
            merged_results[document_id][page_idx][prefix].append(result)
            if len(merged_results[document_id][page_idx]["shortcut_results"]) != 0:
                merged_results[document_id][page_idx]["shortcut_score"] = merged_results[document_id][page_idx][
                    "shortcut_results"
                ][0]["score"]

        for result in retriever_results["shortcut_results"]:
            add_to_merged_results(result, "shortcut_results")
            if float(result["score"]) > self.max_shortcut_score:
                self.max_shortcut_score = float(result["score"])
            if float(result["score"]) < self.min_shortcut_score:
                self.min_shortcut_score = float(result["score"])
        for result in retriever_results["text_results"]:
            add_to_merged_results(result, "text_results")
            if float(result["score"]) > self.max_text_score:
                self.max_text_score = float(result["score"])
            if float(result["score"]) < self.min_text_score:
                self.min_text_score = float(result["score"])
        for result in retriever_results["image_results"]:
            add_to_merged_results(result, "image_results")
            if float(result["score"]) > self.max_image_score:
                self.max_image_score = float(result["score"])
            if float(result["score"]) < self.min_image_score:
                self.min_image_score = float(result["score"])
        for result in retriever_results["bm25_results"]:
            add_to_merged_results(result, "bm25_results")
        return merged_results

    # NOTE: deprecated
    def generate_all_combinations(self, merged_results: dict):
        # 计算所有组合, 然后按照组合分数排序, 最后合并为一组 merged_results 的形式
        all_combinations = [
            # {
            #     "text_results": ,
            #     "image_results": ,
            #     "shortcut_results": [],
            #     "A_score": 0.0,
            #     "BA_score": 0.0,
            #     "AB_score": 0.0,
            # }
        ]
        all_combination_prefixs = set()

        for document_id in merged_results.keys():
            for page_idx in merged_results[document_id].keys():
                # add shortcut_results
                # if len(merged_results[document_id][page_idx]["shortcut_results"]) != 0:  # BUG
                shortcut_results = merged_results[document_id][page_idx]["shortcut_results"]
                if shortcut_results:
                    shortcut_prefix = f"{document_id}_{page_idx}_shortcut_results"
                else:
                    shortcut_prefix = "None"
                for text_page_idx in merged_results[document_id].keys():
                    for text_result in merged_results[document_id][text_page_idx]["text_results"]:
                        text_prefix = f"{document_id}_{text_page_idx}_{text_result['result'].metadata['uuid']}"
                        for image_page_idx in merged_results[document_id].keys():
                            for image_result in merged_results[document_id][image_page_idx]["image_results"]:
                                image_prefix = (
                                    f"{document_id}_{image_page_idx}_{image_result['result'].metadata['uuid']}"
                                )
                                # 判断这个组合是否已经存在
                                all_combination_prefix = f"{shortcut_prefix}_{text_prefix}_{image_prefix}"
                                if all_combination_prefix not in all_combination_prefixs:
                                    all_combination_prefixs.add(all_combination_prefix)
                                else:
                                    continue
                                bbox_dis, diagonal_length = self.cal_bbox_dis(
                                    text_result["result"], image_result["result"]
                                )
                                # 计算shortcut与text, image之间的页数距离
                                text_page_diff = abs(int(text_page_idx) - int(page_idx))
                                image_page_diff = abs(int(image_page_idx) - int(page_idx))
                                if (bbox_dis < (project_config.DISTANCE_THRESHOLD * diagonal_length)) and (
                                    max(text_page_diff, image_page_diff) < project_config.DISTANCE_THRESHOLD
                                ):
                                    A_score = 1.0
                                else:
                                    A_score = project_config.EPSILON

                                # text_shortcut_score = merged_results[document_id][text_page_idx]["shortcut_score"]
                                # image_shortcut_score = merged_results[document_id][image_page_idx]["shortcut_score"]
                                # shortcut_score = max(text_shortcut_score, image_shortcut_score)
                                # BA_score = self.cal_BA_score(text_result, image_result, shortcut_score)
                                shortcut_score = float(shortcut_results[0]["score"]) if shortcut_results else 0.0
                                BA_score = self.cal_BA_score_by_linear_weight(
                                    text_result, image_result, shortcut_results[0]
                                )
                                AB_score = A_score * BA_score

                                # if (
                                #     text_result["result"].metadata["page_idx"]
                                #     != image_result["result"].metadata["page_idx"]
                                # ):
                                #     text_shortcut_results = merged_results[document_id][text_page_idx][
                                #         "shortcut_results"
                                #     ]
                                #     image_shortcut_results = merged_results[document_id][image_page_idx][
                                #         "shortcut_results"
                                #     ]
                                #     shortcuts = text_shortcut_results[0:1] + image_shortcut_results[0:1]
                                # else:
                                #     shortcuts = merged_results[document_id][page_idx]["shortcut_results"]
                                # 所有可能的 (text, image, shortcut) 组合
                                add_to_combination_results(
                                    text_result,
                                    image_result,
                                    shortcut_results,
                                    A_score,
                                    BA_score,
                                    AB_score,
                                    all_combinations,
                                )
        return all_combinations

    async def generate_all_combinations_from_retri(
        self,
        retriever_results: dict,
        query: str,
        result: dict,
        kg: KG | None = None,
        connectivity_map: dict | None = None,
    ):
        # 计算所有组合, 然后按照组合分数排序, 最后合并为一组 merged_results 的形式
        all_combinations = [
            # {
            #     "text_results": ,
            #     "image_results": ,
            #     "shortcut_results": [],
            #     "A_score": 0.0,
            #     "BA_score": 0.0,
            #     "AB_score": 0.0,
            # }
        ]
        all_combination_prefixs = set()

        if project_config.MODEL_PA_BY_KG:
            if (kg is not None) and (not connectivity_map):
                # logger.info(f"start kg query, {query}")
                kg_result = await kg.query(query)
                # result["middle_results"]["kg_result"] = kg_result
                # logger.info(f"finish kg query")
                connectivity_map = await kg.cal_prob_from_relations(
                    search_result=kg_result, scaling_factor=project_config.SCALING_FACTOR
                )
                result["middle_results"]["connectivity_map"] = connectivity_map

        def is_significant(res_item, index, nums, shortcut_doc_id=None, o_k=5):
            if index < project_config.COMB_TOP_K_THRESHOLD:
                return True
            if project_config.MODEL_PA_BY_KG and nums < (project_config.COMB_TOP_K_THRESHOLD + o_k):
                res_uuid = res_item["result"].metadata.get("uuid")
                if res_uuid in connectivity_map:
                    return True
            if not project_config.MODEL_PA_BY_KG:
                doc_id = res_item["result"].metadata.get("source")
                if shortcut_doc_id:
                    if shortcut_doc_id not in doc_id:
                        return False
                if nums < project_config.COMB_TOP_K_THRESHOLD + o_k:
                    return True
            return False

        nums = {"text_result": 0, "image_result": 0, "shortcut_result": 0}

        for s_idx, shortcut_result in enumerate(retriever_results["shortcut_results"]):
            shortcut_doc_id = None
            if project_config.MODEL_PA_BY_KG:
                shortcut_o_k = 20
            else:
                shortcut_o_k = 50
            if not is_significant(shortcut_result, s_idx, nums["shortcut_result"], shortcut_doc_id, shortcut_o_k):
                continue
            shortcut_doc_id = shortcut_result["result"].metadata.get("source")
            nums["text_result"] = 0
            nums["image_result"] = 0
            nums["shortcut_result"] += 1
            shortcut_doc_id = shortcut_result["result"].metadata["source"]
            shortcut_page_idx = shortcut_result["result"].metadata["page_idx"]
            shortcut_prefix = f"{shortcut_doc_id}_{shortcut_page_idx}_shortcut_results"
            for t_idx, text_result in enumerate(retriever_results["text_results"]):
                if not is_significant(text_result, t_idx, nums["text_result"], shortcut_doc_id, 5):
                    continue
                nums["image_result"] = 0
                nums["text_result"] += 1
                text_doc_id = text_result["result"].metadata["source"]
                text_page_idx = text_result["result"].metadata["page_idx"]
                text_prefix = f'{text_doc_id}_{text_page_idx}_{text_result["result"].metadata["uuid"]}'
                for i_idx, image_result in enumerate(retriever_results["image_results"]):
                    if not is_significant(image_result, i_idx, nums["image_result"], shortcut_doc_id, 10):
                        continue
                    nums["image_result"] += 1
                    image_doc_id = image_result["result"].metadata["source"]
                    image_page_idx = image_result["result"].metadata["page_idx"]
                    image_prefix = f'{image_doc_id}_{image_page_idx}_{image_result["result"].metadata["uuid"]}'
                    # 判断这个组合是否已经存在
                    all_combination_prefix = f"{shortcut_prefix}_{text_prefix}_{image_prefix}"
                    if all_combination_prefix not in all_combination_prefixs:
                        all_combination_prefixs.add(all_combination_prefix)
                    else:
                        continue

                    if project_config.MODEL_PA_BY_KG:
                        A_score = self.cal_A_score_by_kg(
                            text_result=text_result,
                            image_result=image_result,
                            shortcut_result=shortcut_result,
                            connectivity_map=connectivity_map,  # type: ignore
                            kg=kg,
                        )
                    else:
                        A_score = self.cal_A_score_by_bbox(
                            text_result=text_result, image_result=image_result, shortcut_result=shortcut_result
                        )

                    if project_config.DShafer:
                        BA_score = self.cal_BA_score_by_Dempster_Shafer(
                            text_result=text_result, image_result=image_result, shortcut_result=shortcut_result
                        )
                    else:
                        BA_score = self.cal_BA_score_by_linear_weight(
                            text_result=text_result, image_result=image_result, shortcut_result=shortcut_result
                        )
                    AB_score = A_score * BA_score
                    # 所有可能的 (text, image, shortcut) 组合
                    add_to_combination_results(
                        text_result,
                        image_result,
                        shortcut_result,
                        A_score,
                        BA_score,
                        AB_score,
                        all_combinations,
                    )
        return all_combinations

    def get_top_k_ab_scores(self, all_combinations):
        final_results = {
            # "document_id": {
            #     "page_idx": {
            #         "text_results": [],
            #         "image_results": [],
            #         "shortcut_results": [],
            #     }
            # }
        }
        text_count = 0
        image_count = 0
        shortcut_count = 0
        seen_text_ids = set()
        seen_image_ids = set()
        seen_shortcut_ids = set()

        sorted_results = sorted(
            all_combinations,
            key=lambda result_dict: result_dict.get("AB_score", -1.0),
            reverse=True,
        )
        # top_k_results = sorted_results[: project_config.TUPLE_RERANK_TOP_K]
        for result in sorted_results:
            text_result = result["text_results"]
            image_result = result["image_results"]
            shortcuts_result = result["shortcut_results"]

            t_id = text_result["result"].metadata.get("uuid", text_result["result"].page_content)
            i_id = image_result["result"].metadata.get("uuid", image_result["result"].page_content)
            s_id = shortcuts_result["result"].metadata.get("uuid", shortcuts_result["result"].page_content)

            text_document_id = Path(text_result["result"].metadata["source"]).name
            image_document_id = Path(image_result["result"].metadata["source"]).name
            shortcut_document_id = Path(shortcuts_result["result"].metadata["source"]).name

            text_page_idx = text_result["result"].metadata["page_idx"]
            image_page_idx = image_result["result"].metadata["page_idx"]
            shortcut_page_idx = shortcuts_result["result"].metadata["page_idx"]

            if text_count < project_config.TUPLE_RERANK_TOP_K_TEXT:
                if t_id not in seen_text_ids:
                    add_to_final_results(text_result, "text_results", text_document_id, text_page_idx, final_results)
                    seen_text_ids.add(t_id)
                    text_count += 1
            if image_count < project_config.TUPLE_RERANK_TOP_K_IMAGE:
                if i_id not in seen_image_ids:
                    add_to_final_results(
                        image_result, "image_results", image_document_id, image_page_idx, final_results
                    )
                    seen_image_ids.add(i_id)
                    image_count += 1
            if shortcut_count < project_config.TUPLE_RERANK_TOP_K_SHORTCUT:
                if s_id not in seen_shortcut_ids:
                    add_to_final_results(
                        shortcuts_result, "shortcut_results", shortcut_document_id, shortcut_page_idx, final_results
                    )
                    seen_shortcut_ids.add(s_id)
                    shortcut_count += 1

            if (
                text_count >= project_config.TUPLE_RERANK_TOP_K_TEXT
                and image_count >= project_config.TUPLE_RERANK_TOP_K_IMAGE
                and shortcut_count >= project_config.TUPLE_RERANK_TOP_K_SHORTCUT
            ):
                break
        return final_results
