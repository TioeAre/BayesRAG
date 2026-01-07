import os, sys
from typing import Union, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.project_config import project_config

from eval.third_party.MMLongBench.eval.eval_score import eval_acc_and_f1
from eval.mmLongBench.test_mmLongBench import read_dataset, _evaluate
from src.eval.eval import print_config
from src.utils.gpt import extract_content_outside_think
from src.utils.utils import save_pkl_result
from src.langchain.utils.uuid import generate_stable_uuid_for_text
from src.RAGAnything.kg import KG
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
from raganything import RAGAnything
import asyncio, traceback, pickle
import base64
import datetime, time
import re
import copy
import json
import uuid


def analysis(results: list):
    acc, f1 = eval_acc_and_f1(results)
    gpt_acc = 0.0
    for result in results:
        gpt_acc += result.get("gpt_score", 0.0)
    logger.info("Avg acc: {}".format(acc))
    logger.info("Avg gpt acc: {}".format(gpt_acc / len(results)))
    logger.info("Avg f1: {}".format(f1))


async def raganything_chat_request(question: str, rag: RAGAnything) -> str:
    try:
        answer = await rag.aquery(query=question, mode="hybrid", vlm_enhanced=True)
        reasoning_content, result = extract_content_outside_think(answer)
        return result
    except Exception as e:
        logger.error(f"RAGAnything Request Error: {e}")
        logger.debug(traceback.format_exc())
        raise e


async def process_single_sample(
    semaphore: asyncio.Semaphore,
    rag: RAGAnything,
    sample_data: dict,
    middle_output_path: str,
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
            predict_answer = await raganything_chat_request(question=question, rag=rag)
            await _evaluate(
                question=question,
                predict_answer=predict_answer,
                ground_truth=ground_truth,
                result=result,
                sample_data=sample_data,
                token_usage=result["token_usage"],
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

    kg = KG()
    await kg.rag._ensure_lightrag_initialized()

    if project_config.RESULT_DIR_NAME == "timestamp":
        base_output_dir = f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval_other_method/RAGAngything/MMLongBench_results/{timestamp}"
    else:
        base_output_dir = f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval_other_method/RAGAngything/MMLongBench_results/{project_config.RESULT_DIR_NAME}"

    semaphore = asyncio.Semaphore(project_config.EVAL_CONCURRENCY_LIMIT)
    tasks = []

    for sample_data in dataset_info:
        qa_id = sample_data["qa"]["id"]
        middle_output_path = (
            f"{base_output_dir}/2_{qa_id}_{generate_stable_uuid_for_text(sample_data['qa']['question'])}.pkl"
        )
        if not os.path.exists(middle_output_path):
            task = asyncio.create_task(
                process_single_sample(
                    semaphore=semaphore,
                    rag=kg.rag,
                    sample_data=sample_data,
                    middle_output_path=middle_output_path,
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
