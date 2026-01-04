import json, os, sys, pickle
from collections import defaultdict
from loguru import logger
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config


def calculate_metrics(data_path, num=0):

    with open(data_path, "r") as f:
        datas = json.load(f)
        data = datas
        if num != 0:
            data = datas[:num]
        total_count = len(data)
        total_score = 0
        total_gpt_score = 0

        total_tokens = {"prompt": 0, "completion": 0, "total": 0}

        time_keys = [
            "bm25_cost",
            "text_cost",
            "image_cost",
            "shortcut_cost",
            "text_rerank_cost",
            "retriever_cost",
            "rerank_cost",
            "qa_cost",
        ]
        total_time_costs = {key: 0.0 for key in time_keys}

        doc_type_stats = defaultdict(lambda: {"scores": [], "gpt_scores": []})

        na_scores = []
        na_gpt_scores = []

        if total_count == 0:
            logger.warning("数据为空，无法计算。")
            return

        for item in data:
            score = item.get("score", 0.0)
            gpt_score = item.get("gpt_score", 0.0)

            total_score += score
            total_gpt_score += gpt_score

            usage = item.get("token_usage", {})
            total_tokens["prompt"] += usage.get("prompt_tokens", 0)
            total_tokens["completion"] += usage.get("completion_tokens", 0)
            total_tokens["total"] += usage.get("total_tokens", 0)

            time_cost_data = item.get("time_record", {})
            for key in time_keys:
                total_time_costs[key] += time_cost_data.get(key, 0)

            doc_type = item.get("qa", {}).get("doc_type", "Unknown")
            doc_type_stats[doc_type]["scores"].append(score)
            doc_type_stats[doc_type]["gpt_scores"].append(gpt_score)

            if item.get("answer") == "Not answerable":
                na_scores.append(score)
                na_gpt_scores.append(gpt_score)

        logger.info(data_path)

        logger.info(f"====== 总体统计 (Total Samples: {total_count}) ======")
        logger.info(f"Average Score:     {total_score / total_count:.4f}")
        logger.info(f"Average GPT Score: {total_gpt_score / total_count:.4f}")
        logger.info("-" * 40)

        logger.info(f"====== 按 Doc Type 分类统计 ======")
        for dtype, stats in doc_type_stats.items():
            count = len(stats["scores"])
            avg_s = sum(stats["scores"]) / count
            avg_gs = sum(stats["gpt_scores"]) / count
            logger.info(f"Type: {dtype} (Count: {count})")
            logger.info(f"  - Avg Score:     {avg_s:.4f}")
            logger.info(f"  - Avg GPT Score: {avg_gs:.4f}")
        logger.info("-" * 40)

        logger.info(f"====== Answer 为 'Not answerable' 的统计 ======")
        na_count = len(na_scores)
        if na_count > 0:
            logger.info(f"Count: {na_count}")
            logger.info(f"Avg Score:     {sum(na_scores) / na_count:.4f}")
            logger.info(f"Avg GPT Score: {sum(na_gpt_scores) / na_count:.4f}")
        else:
            logger.info("未找到 'answer': 'Not answerable' 的样本。")

        logger.info(f"====== 平均 Token Usage ======")
        logger.info(f"Avg Prompt Tokens:     {total_tokens['prompt'] / total_count:.2f}")
        logger.info(f"Avg Completion Tokens: {total_tokens['completion'] / total_count:.2f}")
        logger.info(f"Avg Total Tokens:      {total_tokens['total'] / total_count:.2f}")
        logger.info("-" * 40)

        logger.info(f"====== 平均耗时统计 (Time Cost) ======")
        for key, total_val in total_time_costs.items():
            avg_val = total_val / total_count
            logger.info(f"Avg {key:15}: {avg_val:.4f}s")
        logger.info("-" * 40)


def calculate_metrics_from_pkl(data_path):

    pkl_dir = Path(data_path)
    total_count = 0
    total_score = 0
    total_gpt_score = 0
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    time_keys = [
        "bm25_cost",
        "text_cost",
        "image_cost",
        "shortcut_cost",
        "text_rerank_cost",
        "retriever_cost",
        "rerank_cost",
        "qa_cost",
    ]
    total_time_costs = {key: 0.0 for key in time_keys}
    doc_type_stats = defaultdict(lambda: {"scores": [], "gpt_scores": []})
    na_scores = []
    na_gpt_scores = []
    i = 0

    for pkl_file in pkl_dir.glob("*.pkl"):
        i = i + 1
        # if i > 410:
        #     break
        print(pkl_file)
        data = pickle.loads(Path(pkl_file).read_bytes())
        total_count += 1

        score = data.get("score", 0.0)
        gpt_score = data.get("gpt_score", 0.0)

        total_score += score
        total_gpt_score += gpt_score

        usage = data.get("token_usage", {})
        total_tokens["prompt"] += usage.get("prompt_tokens", 0)
        total_tokens["completion"] += usage.get("completion_tokens", 0)
        total_tokens["total"] += usage.get("total_tokens", 0)

        time_cost_data = data.get("time_record", {})
        for key in time_keys:
            total_time_costs[key] += time_cost_data.get(key, 0)

        doc_type = data.get("qa", {}).get("doc_type", "Unknown")
        doc_type_stats[doc_type]["scores"].append(score)
        doc_type_stats[doc_type]["gpt_scores"].append(gpt_score)

        if data.get("answer") == "Not answerable":
            na_scores.append(score)
            na_gpt_scores.append(gpt_score)

    logger.info(data_path)

    logger.info(f"====== 总体统计 (Total Samples: {total_count}) ======")
    logger.info(f"Average Score:     {total_score / total_count:.4f}")
    logger.info(f"Average GPT Score: {total_gpt_score / total_count:.4f}")
    logger.info("-" * 40)

    logger.info(f"====== 按 Doc Type 分类统计 ======")
    for dtype, stats in doc_type_stats.items():
        count = len(stats["scores"])
        avg_s = sum(stats["scores"]) / count
        avg_gs = sum(stats["gpt_scores"]) / count
        logger.info(f"Type: {dtype} (Count: {count})")
        logger.info(f"  - Avg Score:     {avg_s:.4f}")
        logger.info(f"  - Avg GPT Score: {avg_gs:.4f}")
    logger.info("-" * 40)

    logger.info(f"====== Answer 为 'Not answerable' 的统计 ======")
    na_count = len(na_scores)
    if na_count > 0:
        logger.info(f"Count: {na_count}")
        logger.info(f"Avg Score:     {sum(na_scores) / na_count:.4f}")
        logger.info(f"Avg GPT Score: {sum(na_gpt_scores) / na_count:.4f}")
    else:
        logger.info("未找到 'answer': 'Not answerable' 的样本。")

    logger.info(f"====== 平均 Token Usage ======")
    logger.info(f"Avg Prompt Tokens:     {total_tokens['prompt'] / total_count:.2f}")
    logger.info(f"Avg Completion Tokens: {total_tokens['completion'] / total_count:.2f}")
    logger.info(f"Avg Total Tokens:      {total_tokens['total'] / total_count:.2f}")
    logger.info("-" * 40)

    logger.info(f"====== 平均耗时统计 (Time Cost) ======")
    for key, total_val in total_time_costs.items():
        avg_val = total_val / total_count
        logger.info(f"Avg {key:15}: {avg_val:.4f}s")
    logger.info("-" * 40)


if __name__ == "__main__":
    # mmlongbench_result_path = f"{project_config.project_root}/eval_other_method/ViDoRAG/MMLongBench_results/gpt_4o_mini_Bench_results/1_simplified_results.json"

    # mmlongbench_result_path = f"{project_config.project_root}/eval_other_method/RAGFlow/MMLongBench_results/gpt_4o_mini_Bench_results/1_simplified_results.json"

    # mmlongbench_result_path = f"{project_config.project_root}/eval_other_method/VisRAG/MMLongBench_results/gpt_4o_mini_Bench_results/1_simplified_results.json"

    mmlongbench_result_path = f"{project_config.project_root}/eval_other_method/RAGAngything/MMLongBench_results/Qwen3_Bench_results/1_simplified_results.json"
    # mmlongbench_result_path = f"{project_config.project_root}/eval_other_method/RAGAngything/MMLongBench_results/gpt_4o_mini_Bench_results/1_simplified_results.json"

    # mmlongbench_result_path = f"{project_config.project_root}/eval_other_method/pure_model/MMLongBench_results/gpt_4o_mini_Bench_results/1_simplified_results.json"

    # mmlongbench_result_path = (
    #     f"{project_config.project_root}/eval/mmLongBench/results/Qwen3_Bench_results_test1/1_simplified_results.json"
    # )

    # mmlongbench_result_path = f"{project_config.project_root}/eval/mmLongBench/results/gpt_4o_mini_Bench_results_reranker/1_simplified_results.json"
    # mmlongbench_result_path = f"{project_config.project_root}/eval/mmLongBench/results/gpt_4o_mini_Bench_results_final/1_simplified_results.json"

    calculate_metrics(mmlongbench_result_path)

    # calculate_metrics_from_pkl(f"{project_config.project_root}/eval/mmLongBench/results/gpt_4o_mini_Bench_results")

    # calculate_metrics_from_pkl(f"{project_config.project_root}/eval/mmLongBench/results/Qwen3_Bench_results_test1")

    # calculate_metrics_from_pkl(
    # f"{project_config.project_root}/eval_other_method/RAGAngything/MMLongBench_results/gpt_4o_mini_Bench_results"
    # )
    # calculate_metrics_from_pkl(
    #     f"{project_config.project_root}/eval_other_method/RAGAngything/MMLongBench_results/Qwen3_Bench_results"
    # )

    # calculate_metrics_from_pkl(
    #     f"{project_config.project_root}/eval/mmLongBench/results/gpt_4o_mini_Bench_results_linear"
    # )
    # calculate_metrics_from_pkl(f"{project_config.project_root}/eval/mmLongBench/results/Qwen3_Bench_results_test1")
    # calculate_metrics_from_pkl(
    #     f"{project_config.project_root}/eval/mmLongBench/results/gpt_4o_mini_Bench_results_final"
    # )
