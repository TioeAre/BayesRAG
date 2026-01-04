import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.project_config import project_config

from eval.third_party.MMLongBench.eval.eval_score import eval_acc_and_f1
from eval.mmLongBench.test_mmLongBench import read_dataset, _evaluate
from src.eval.eval import print_config
from src.utils.utils import save_pkl_result
from src.langchain.utils.uuid import generate_stable_uuid_for_text
from loguru import logger
from pathlib import Path
import asyncio, traceback, pickle
import datetime
import copy
import json
from typing import List, Dict


def analysis(results: list):
    acc, f1 = eval_acc_and_f1(results)
    gpt_acc = 0.0
    for result in results:
        gpt_acc += result.get("gpt_score", 0.0)
    logger.info("Avg acc: {}".format(acc))
    logger.info("Avg gpt acc: {}".format(gpt_acc / len(results)))
    logger.info("Avg f1: {}".format(f1))


def parse_visrag_results(
    data_path: str = os.getenv(
        "VISRAG_RAW_RESULTS_PATH",
        "/projects/rag_method/UltraRAG/output/memory_test_mmLongBench_eval_visrag_mmLongBench_20251217_143634.json",
    )
) -> List[Dict]:
    raw_data = json.load(open(data_path, "r"))
    entries = []
    for item in raw_data:
        step = item.get("step")
        memory = item.get("memory", {})
        if step == "benchmark.get_data":
            questions = memory["memory_q_ls"]
            ground_truths = memory["memory_gt_ls"]
        elif step == "generation.multimodal_generate":
            predict_answers = memory["memory_ans_ls"]
        elif step == "retriever.retriever_search":
            retriever = memory["memory_ret_psg"]

    for i in range(len(questions)):
        entry = {
            "qa": {"question": questions[i], "ground_truth": ground_truths[i][0], "id": i},
            "predict": predict_answers[i],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "middle_results": {"references": retriever},
        }
        entries.append(entry)

    return entries


async def process_single_sample(
    semaphore: asyncio.Semaphore,
    num: int,
    sample_data: dict,
    middle_output_path: str,
    visrah_results: List[Dict],
):
    async with semaphore:
        result = copy.deepcopy(sample_data)
        result["predict"] = {}
        result["predict"]["provenance"] = {"text": [], "image": [], "shortcut": [], "before_rerank": []}
        result["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        result["middle_results"] = {}
        question = sample_data["qa"]["question"]
        ground_truth = sample_data["qa"]["ground_truth"]

        try:
            loop = asyncio.get_running_loop()
            predict_answer, references, token_usage = (
                visrah_results[num]["predict"],
                visrah_results[num]["middle_results"]["references"],
                visrah_results[num]["token_usage"],
            )
            result["middle_results"]["references"] = references
            await _evaluate(
                question=question,
                predict_answer=predict_answer,
                ground_truth=ground_truth,
                result=result,
                sample_data=sample_data,
                token_usage=token_usage,
            )

            save_pkl_result(result, middle_output_path)

            return result
        except Exception as e:
            logger.error(f"Error processing sample {sample_data.get('qa', {}).get('id')}: {e}")
            logger.debug(traceback.format_exc())
            result["error"] = str(e)
            return result


async def main():
    print_config()
    dataset_info = read_dataset(full=project_config.TEST_BENCH_FULL)
    timestamp = datetime.datetime.now().isoformat()

    if project_config.RESULT_DIR_NAME == "timestamp":
        base_output_dir = (
            f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval_other_method/VisRAG/MMLongBench_results/{timestamp}"
        )
    else:
        base_output_dir = f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval_other_method/VisRAG/MMLongBench_results/{project_config.RESULT_DIR_NAME}"

    semaphore = asyncio.Semaphore(project_config.EVAL_CONCURRENCY_LIMIT)
    tasks = []
    i = 0
    visrah_results = parse_visrag_results()

    for sample_data in dataset_info:
        i += 1
        qa_id = sample_data["qa"]["id"]
        middle_output_path = (
            f"{base_output_dir}/2_{qa_id}_{generate_stable_uuid_for_text(sample_data['qa']['question'])}.pkl"
        )
        if not os.path.exists(middle_output_path):
            task = asyncio.create_task(
                process_single_sample(
                    semaphore=semaphore,
                    sample_data=sample_data,
                    num=i,
                    middle_output_path=middle_output_path,
                    visrah_results=visrah_results,
                )
            )
        else:
            task = asyncio.create_task(
                asyncio.to_thread(lambda path=middle_output_path: pickle.loads(Path(path).read_bytes()))
            )
        tasks.append(task)

    logger.info(f"Starting concurrent evaluation of {len(tasks)} samples...")
    results = await asyncio.gather(*tasks)
    analysis(results)

    if project_config.WRITE_RESULTS:
        os.makedirs(base_output_dir, exist_ok=True)

        simplified_results = []
        for result in results:
            simple_result = copy.deepcopy(result)
            simple_result.pop("middle_results", None)
            simplified_results.append(simple_result)
        simplified_output_path = f"{base_output_dir}/1_simplified_results.json"
        with open(simplified_output_path, "w") as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to directory: {base_output_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
