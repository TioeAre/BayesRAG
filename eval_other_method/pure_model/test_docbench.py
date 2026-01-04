import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config

import json
import asyncio
import traceback
import pickle
import datetime
import re
import copy
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor


from src.eval.eval import print_config
from src.utils.unigpt import GPT
from src.utils.utils import save_pkl_result
from src.dataset_loader.docbench import dataset as Dataset
from src.langchain.utils.uuid import generate_stable_uuid_for_text
from eval.DocBench.test_DocBench import load_eval_prompt, read_dataset, analysis, _evaluate
from eval_other_method.pure_model.test_mmlongbench import openai_chat_request

from openai import OpenAI
import fitz  # PyMuPDF
import random
import io
from PIL import Image


async def process_single_sample(
    semaphore: asyncio.Semaphore,
    pdf_path: str,
    sample_data: dict,
    DOCBENCH_EVAL_PROMPT_TEMPLATE: str,
    middle_output_path: str,
):
    async with semaphore:
        result = copy.deepcopy(sample_data)
        result["predict"] = {}
        result["predict"]["provenance"] = {
            "text": [],
            "image": [],
            "shortcut": [],
            "before_rerank": [],
        }
        result["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        result["middle_results"] = {}
        question = sample_data["qa"]["question"]
        ground_truth = sample_data["qa"]["ground_truth"]

        try:
            loop = asyncio.get_running_loop()
            predict_answer, references, token_usage = await loop.run_in_executor(
                None, openai_chat_request, question, pdf_path
            )
            result["middle_results"]["references"] = references
            await _evaluate(
                question=question,
                predict_answer=predict_answer,
                ground_truth=ground_truth,
                result=result,
                sample_data=sample_data,
                DOCBENCH_EVAL_PROMPT_TEMPLATE=DOCBENCH_EVAL_PROMPT_TEMPLATE,
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    DOCBENCH_EVAL_PROMPT_TEMPLATE = load_eval_prompt()

    if project_config.RESULT_DIR_NAME == "timestamp":
        base_output_dir = (
            f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval_other_method/pure_model/DocBench_results/{timestamp}"
        )
    else:
        base_output_dir = f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval_other_method/pure_model/DocBench_results/{project_config.RESULT_DIR_NAME}"

    semaphore = asyncio.Semaphore(project_config.EVAL_CONCURRENCY_LIMIT)
    tasks = []

    for sample_data in dataset_info:
        qa_id = sample_data["qa"]["id"]
        middle_output_path = (
            f"{base_output_dir}/2_{qa_id}_{generate_stable_uuid_for_text(sample_data['qa']['question'])}.pkl"
        )
        pdf_path = os.path.join(project_config.project_root, "dataset", "DocBench", sample_data["qa"]["num_id"], qa_id)
        if not os.path.exists(middle_output_path):
            task = asyncio.create_task(
                process_single_sample(
                    semaphore,
                    pdf_path,
                    sample_data,
                    DOCBENCH_EVAL_PROMPT_TEMPLATE,
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
        with open(simplified_output_path, "w", encoding="utf-8") as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Split results saved to directory: {base_output_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
