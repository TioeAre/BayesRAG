import json, os, sys, pickle
from collections import defaultdict
from loguru import logger
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config


types_mapping = {
    "text-only": "text",
    "multimodal-f": "mm",
    "multimodal-t": "mm",
    "multimodal": "mm",
    "meta-data": "meta",
    "unanswerable": "una",
    "una-web": "una",
}

file_ranges = {
    "aca": range(0, 49),
    "fin": range(49, 89),
    "gov": range(89, 133),
    "law": range(133, 179),
    "new": range(179, 229),
}


def calculate_metrics(data_path):

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        if not data:
            logger.warning("数据列表为空，无法计算。")
            return

        total_score = 0
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

        format_scores = defaultdict(list)
        type_scores = defaultdict(list)
        domain_scores = defaultdict(list)

        count = len(data)

        for item in data:
            score = item.get("score", 0.0)
            total_score += score
            qa_info = item.get("qa", {})

            ans_format = qa_info.get("answer_format", "unknown")
            format_scores[ans_format].append(score)

            mapped_type = types_mapping.get(ans_format, "other")
            type_scores[mapped_type].append(score)

            num_id_str = qa_info.get("num_id", "-1")
            try:
                num_id = int(num_id_str)
                found_domain = False
                for domain, rng in file_ranges.items():
                    if num_id in rng:
                        domain_scores[domain].append(score)
                        found_domain = True
                        break
                if not found_domain:
                    domain_scores["out_of_range"].append(score)
            except (ValueError, TypeError):
                domain_scores["unknown_id"].append(score)

            tokens = item.get("token_usage", {})
            total_tokens["prompt"] += tokens.get("prompt_tokens", 0)
            total_tokens["completion"] += tokens.get("completion_tokens", 0)
            total_tokens["total"] += tokens.get("total_tokens", 0)

            time_cost_data = item.get("time_record", {})
            for key in time_keys:
                total_time_costs[key] += time_cost_data.get(key, 0)

        logger.info(f"Processing File: {data_path}")

        logger.info(f"====== 总体统计 (Total Samples: {count}) ======")
        logger.info(f"Total Average Score: {total_score / count:.4f}")
        logger.info("-" * 30)

        logger.info("====== 按 Types (大类) 统计 Score ======")
        for t_type, scores in type_scores.items():
            avg_type_score = sum(scores) / len(scores)
            logger.info(f"Type: {t_type:<10} | Count: {len(scores):<3} | Avg Score: {avg_type_score:.4f}")
        logger.info("-" * 30)

        logger.info("====== 按 Domain (ID Range) 统计 Score ======")
        for domain in file_ranges.keys():
            scores = domain_scores.get(domain, [])
            if scores:
                avg_dom_score = sum(scores) / len(scores)
                logger.info(f"Domain: {domain:<10} | Count: {len(scores):<3} | Avg Score: {avg_dom_score:.4f}")
            else:
                logger.info(f"Domain: {domain:<10} | Count: 0   | Avg Score: N/A")
        for domain, scores in domain_scores.items():
            if domain not in file_ranges:
                avg = sum(scores) / len(scores)
                logger.info(f"Domain: {domain:<10} | Count: {len(scores):<3} | Avg Score: {avg:.4f}")
        logger.info("-" * 30)

        logger.info("====== 按原始 Answer Format 统计 Score ======")
        for fmt, scores in format_scores.items():
            avg_fmt_score = sum(scores) / len(scores)
            logger.info(f"Format: {fmt:<15} | Count: {len(scores):<3} | Avg Score: {avg_fmt_score:.4f}")
        logger.info("-" * 30)

        logger.info("====== 平均 Token Usage ======")
        logger.info(f"Avg Prompt Tokens:     {total_tokens['prompt'] / count:.2f}")
        logger.info(f"Avg Completion Tokens: {total_tokens['completion'] / count:.2f}")
        logger.info(f"Avg Total Tokens:      {total_tokens['total'] / count:.2f}")

        logger.info(f"====== 平均耗时统计 (Time Cost) ======")
        for key, total_val in total_time_costs.items():
            avg_val = total_val / count
            logger.info(f"Avg {key:15}: {avg_val:.4f}s")
        logger.info("-" * 40)


def calculate_metrics_from_pkl(data_path):

    pkl_dir = Path(data_path)
    total_count = 0
    total_score = 0
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

    format_scores = defaultdict(list)
    type_scores = defaultdict(list)
    domain_scores = defaultdict(list)
    i = 0

    for pkl_file in pkl_dir.glob("*.pkl"):
        i = i + 1
        # if i > 410:
        #     break
        # print(pkl_file)
        data = pickle.loads(Path(pkl_file).read_bytes())
        data.pop("middle_results", None)
        total_count += 1

        score = data.get("score", 0.0)
        total_score += score

        qa_info = data.get("qa", {})
        ans_format = qa_info.get("answer_format", "unknown")
        format_scores[ans_format].append(score)
        mapped_type = types_mapping.get(ans_format, "other")
        type_scores[mapped_type].append(score)
        num_id_str = qa_info.get("num_id", "-1")

        try:
            num_id = int(num_id_str)
            found_domain = False
            for domain, rng in file_ranges.items():
                if num_id in rng:
                    domain_scores[domain].append(score)
                    found_domain = True
                    break
            if not found_domain:
                domain_scores["out_of_range"].append(score)
        except (ValueError, TypeError):
            domain_scores["unknown_id"].append(score)

        tokens = data.get("token_usage", {})
        total_tokens["prompt"] += tokens.get("prompt_tokens", 0)
        total_tokens["completion"] += tokens.get("completion_tokens", 0)
        total_tokens["total"] += tokens.get("total_tokens", 0)

        time_cost_data = data.get("time_record", {})
        for key in time_keys:
            total_time_costs[key] += time_cost_data.get(key, 0)

    logger.info(f"Processing File: {data_path}")

    logger.info(f"====== 总体统计 (Total Samples: {total_count}) ======")
    logger.info(f"Total Average Score: {total_score / total_count:.4f}")
    logger.info("-" * 30)

    logger.info("====== 按 Types (大类) 统计 Score ======")
    for t_type, scores in type_scores.items():
        avg_type_score = sum(scores) / len(scores)
        logger.info(f"Type: {t_type:<10} | Count: {len(scores):<3} | Avg Score: {avg_type_score:.4f}")
    logger.info("-" * 30)

    logger.info("====== 按 Domain (ID Range) 统计 Score ======")
    for domain in file_ranges.keys():
        scores = domain_scores.get(domain, [])
        if scores:
            avg_dom_score = sum(scores) / len(scores)
            logger.info(f"Domain: {domain:<10} | Count: {len(scores):<3} | Avg Score: {avg_dom_score:.4f}")
        else:
            logger.info(f"Domain: {domain:<10} | Count: 0   | Avg Score: N/A")
    for domain, scores in domain_scores.items():
        if domain not in file_ranges:
            avg = sum(scores) / len(scores)
            logger.info(f"Domain: {domain:<10} | Count: {len(scores):<3} | Avg Score: {avg:.4f}")
    logger.info("-" * 30)

    logger.info("====== 按原始 Answer Format 统计 Score ======")
    for fmt, scores in format_scores.items():
        avg_fmt_score = sum(scores) / len(scores)
        logger.info(f"Format: {fmt:<15} | Count: {len(scores):<3} | Avg Score: {avg_fmt_score:.4f}")
    logger.info("-" * 30)

    logger.info("====== 平均 Token Usage ======")
    logger.info(f"Avg Prompt Tokens:     {total_tokens['prompt'] / total_count:.2f}")
    logger.info(f"Avg Completion Tokens: {total_tokens['completion'] / total_count:.2f}")
    logger.info(f"Avg Total Tokens:      {total_tokens['total'] / total_count:.2f}")

    logger.info(f"====== 平均耗时统计 (Time Cost) ======")
    for key, total_val in total_time_costs.items():
        avg_val = total_val / total_count
        logger.info(f"Avg {key:15}: {avg_val:.4f}s")
    logger.info("-" * 40)


if __name__ == "__main__":
    # docbench_result_path = f"{project_config.project_root}/eval_other_method/ViDoRAG/DocBench_results/gpt_4o_mini_Bench_results/1_simplified_results.json"

    # docbench_result_path = f"{project_config.project_root}/eval_other_method/RAGFlow/DocBench_results/gpt_4o_mini_Bench_results/1_simplified_results.json"

    # docbench_result_path = f"{project_config.project_root}/eval_other_method/VisRAG/DocBench_results/gpt_4o_mini_Bench_results/1_simplified_results.json"

    # docbench_result_path = f"{project_config.project_root}/eval_other_method/RAGAngything/DocBench_results/Qwen3_Bench_results/1_simplified_results.json"
    docbench_result_path = f"{project_config.project_root}/eval_other_method/RAGAngything/DocBench_results/gpt_4o_mini_Bench_results/1_simplified_results.json"

    # docbench_result_path = (
    #     f"{project_config.project_root}/eval/DocBench/results/Qwen3_Bench_results2/1_simplified_results.json"
    # )
    docbench_result_path = (
        f"{project_config.project_root}/eval/DocBench/results/gpt_4o_mini_Bench_results_final/1_simplified_results.json"
    )
    docbench_result_path = f"{project_config.project_root}/eval/count_result/docbench_union.json"

    calculate_metrics(docbench_result_path)

    # calculate_metrics_from_pkl(
    #     "{project_config.project_root}/eval_other_method/RAGAngything/DocBench_results/Qwen3_Bench_results"
    # )

    # calculate_metrics_from_pkl("{project_config.project_root}/eval/DocBench/results/Qwen3_Bench_results2")
    # calculate_metrics_from_pkl(
    #     "{project_config.project_root}/eval/DocBench/results/gpt_4o_mini_Bench_results_final"
    # )
