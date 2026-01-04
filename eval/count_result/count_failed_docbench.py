import json, os, sys, pickle
from collections import defaultdict
from loguru import logger
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config


def conduct_case_study(data_paths, num=0):
    if not data_paths or not isinstance(data_paths, list):
        logger.error("data_paths not available list")
        return

    all_datasets = []
    for path in data_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                if num != 0:
                    d = d[:num]
                all_datasets.append(d)
        except Exception as e:
            return

    base_data = all_datasets[0]
    total_samples = len(base_data)
    for i, d in enumerate(all_datasets[1:], 1):
        if len(d) != total_samples:
            logger.warning(f"警告：文件 {data_paths[i]} 的样本数量 ({len(d)}) 与基准文件 ({total_samples}) 不一致！")

    target_cases = []

    for idx in range(total_samples):
        base_item = base_data[idx]
        base_gpt_score = base_item.get("score", 0.0)

        if base_gpt_score >= 1.0:
            others_failed = True

            for other_dataset in all_datasets[1:]:
                other_item = other_dataset[idx]
                if other_item.get("score", 0.0) >= 1.0:
                    others_failed = False
                    break

            if others_failed:
                case_info = {
                    "index": idx,
                    "doc_id": base_item.get("qa", {}).get("id", "N/A"),
                    "num_id": base_item.get("qa", {}).get("num_id", "N/A"),
                    "question": base_item.get("qa", {}).get("question", "N/A"),
                    "ground_truth": base_item.get("qa", {}).get("ground_truth", "N/A"),
                    "base_model_answer": base_item.get("pred", ""),
                    "other_models": [
                        {
                            "predict_answer": d[idx].get("pred", ""),
                        }
                        for d in all_datasets[1:]
                    ],
                }
                target_cases.append(case_info)

    logger.info(f"分析完成。对比文件数: {len(data_paths)}")
    logger.info(f"总样本量: {total_samples}")
    logger.info(f"符合条件 (Base胜出, 其他全败) 的 Case 数量: {len(target_cases)}")
    logger.info("-" * 40)

    if target_cases:
        logger.info("部分 Case 详情如下：")
        for i, case in enumerate(target_cases[:5]):
            logger.info(f"Case {i+1} (Index: {case['index']}):")
            logger.info(f"  [Question]: {case['question']}")
            logger.info(f"  [GT]: {case['ground_truth']}")
            logger.info(f"  [Base Answer]: {case['base_model_answer']}")
            logger.info(f"  [Other GPT Scores]: {case['other_models']}")
            logger.info("-" * 20)

    return target_cases


def conduct_failed(data_paths, num=0):
    if not data_paths or not isinstance(data_paths, list):
        logger.error("data_paths not available list")
        return

    all_datasets = []
    for path in data_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                if num != 0:
                    d = d[:num]
                all_datasets.append(d)
        except Exception as e:
            return

    base_data = all_datasets[0]
    total_samples = len(base_data)
    for i, d in enumerate(all_datasets[1:], 1):
        if len(d) != total_samples:
            logger.warning(f"警告：文件 {data_paths[i]} 的样本数量 ({len(d)}) 与基准文件 ({total_samples}) 不一致！")

    target_cases = []

    for idx in range(total_samples):
        base_item = base_data[idx]
        base_gpt_score = base_item.get("score", 0.0)

        if base_gpt_score == 0.0:

            case_info = {
                "index": idx,
                "doc_id": base_item.get("qa", {}).get("id", "N/A"),
                "num_id": base_item.get("qa", {}).get("num_id", "N/A"),
                "question": base_item.get("qa", {}).get("question", "N/A"),
                "ground_truth": base_item.get("qa", {}).get("ground_truth", "N/A"),
                "base_model_answer": base_item.get("pred", ""),
                "score": base_item.get("score", 0.0),
            }
            target_cases.append(case_info)

    if target_cases:
        logger.info("部分 Case 详情如下：")
        for i, case in enumerate(target_cases[:5]):
            logger.info(f"Case {i+1} (Index: {case['index']}):")
            logger.info(f"  [Question]: {case['question']}")
            logger.info(f"  [GT]: {case['ground_truth']}")
            logger.info(f"  [Base Answer]: {case['base_model_answer']}")
            logger.info("-" * 20)

    return target_cases


def conduct_case_study_from_pkl(data_path, num=0):
    pkl_dir = Path(data_path)
    i = 0

    all_datasets = []
    for pkl_file in pkl_dir.glob("*.pkl"):
        try:
            data = pickle.loads(Path(pkl_file).read_bytes())
            data.pop("middle_results", None)
            all_datasets.append(data)
        except Exception as e:
            return

    base_data = all_datasets
    total_samples = len(base_data)

    target_cases = []

    for idx in range(total_samples):
        base_item = base_data[idx]
        base_gpt_score = base_item.get("score", 0.0)

        if base_gpt_score < 1.0:
            case_info = {
                "index": idx,
                "doc_id": base_item.get("qa", {}).get("id", "N/A"),
                "num_id": base_item.get("qa", {}).get("num_id", "N/A"),
                "question": base_item.get("qa", {}).get("question", "N/A"),
                "ground_truth": base_item.get("qa", {}).get("ground_truth", "N/A"),
                "base_model_answer": base_item.get("pred", ""),
            }
            target_cases.append(case_info)

    logger.info(f"总样本量: {total_samples}")
    logger.info(f"符合条件 (Base胜出, 其他全败) 的 Case 数量: {len(target_cases)}")
    logger.info("-" * 40)

    if target_cases:
        logger.info("部分 Case 详情如下：")
        for i, case in enumerate(target_cases[:5]):
            logger.info(f"Case {i+1} (Index: {case['index']}):")
            logger.info(f"  [Question]: {case['question']}")
            logger.info(f"  [GT]: {case['ground_truth']}")
            logger.info(f"  [Base Answer]: {case['base_model_answer']}")
            logger.info("-" * 20)

    return target_cases


def conduct_union(data_paths, num=0):
    if not data_paths or not isinstance(data_paths, list):
        logger.error("data_paths not available list")
        return

    all_datasets = []
    for path in data_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                if num != 0:
                    d = d[:num]
                all_datasets.append(d)
        except Exception as e:
            return

    base_data = all_datasets[0]
    total_samples = len(base_data)
    for i, d in enumerate(all_datasets[1:], 1):
        if len(d) != total_samples:
            logger.warning(f"警告：文件 {data_paths[i]} 的样本数量 ({len(d)}) 与基准文件 ({total_samples}) 不一致！")

    target_cases = []

    for idx in range(total_samples):
        base_item = base_data[idx]
        base_gpt_score = base_item.get("score", 0.0)

        if base_gpt_score >= 1.0:
            target_cases.append(base_item)
            continue

        other_failed = True
        for other_dataset in all_datasets[1:]:
            other_item = other_dataset[idx]
            if other_item.get("score", 0.0) >= 1.0:
                target_cases.append(other_item)
                other_failed = False
                break
        if other_failed:
            target_cases.append(base_item)

    return target_cases


if __name__ == "__main__":
    # paths = [
    #     f"{project_config.project_root}/eval_other_method/RAGAngything/DocBench_results/Qwen3_Bench_results/1_simplified_results.json",
    #     f"{project_config.project_root}/eval/DocBench/results/Qwen3_Bench_results2/1_simplified_results.json",
    # ]
    paths = [
        f"{project_config.project_root}/eval/DocBench/results/gpt_4o_mini_Bench_results_final/1_simplified_results.json",
        f"{project_config.project_root}/eval_other_method/RAGAngything/DocBench_results/Qwen3_Bench_results/1_simplified_results.json",
    ]

    # target_cases = conduct_failed(paths)

    # target_cases = conduct_case_study_from_pkl(
    # f"{project_config.project_root}/eval/DocBench/results/gpt_4o_mini_Bench_results_final"
    # )

    # with open("{project_config.project_root}/eval/count_result/docbench_failed_case_study.json", "w") as f:
    # json.dump(target_cases, f, indent=2, ensure_ascii=False)
    # print(json.dumps(target_cases))

    paths = [
        f"{project_config.project_root}/eval/DocBench/results/gpt_4o_mini_Bench_results_final/1_simplified_results.json",
        f"{project_config.project_root}/eval/DocBench/results/gpt_4o_mini_Bench_results_final/1_simplified_results.backup.json",
    ]

    target_cases = conduct_union(paths)

    with open("{project_config.project_root}/eval/count_result/docbench_union.json", "w") as f:
        json.dump(target_cases, f, indent=2, ensure_ascii=False)
