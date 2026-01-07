import os, sys
from typing import Union, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.project_config import project_config

from eval.third_party.MMLongBench.eval.eval_score import eval_acc_and_f1
from eval.mmLongBench.test_mmLongBench import read_dataset, _evaluate
from src.eval.eval import print_config
from src.utils.gpt import GPT, extract_content_outside_think
from src.utils.utils import save_pkl_result
from src.langchain.utils.uuid import generate_stable_uuid_for_text
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
import asyncio, traceback, pickle
import base64
import datetime, time
import re
import copy
import json
import uuid
import fitz  # PyMuPDF
import random
import io
from PIL import Image


def analysis(results: list):
    acc, f1 = eval_acc_and_f1(results)
    gpt_acc = 0.0
    for result in results:
        gpt_acc += result.get("gpt_score", 0.0)
    logger.info("Avg acc: {}".format(acc))
    logger.info("Avg gpt acc: {}".format(gpt_acc / len(results)))
    logger.info("Avg f1: {}".format(f1))


def openai_chat_request(question: str, pdf_path: str) -> Tuple[str, Optional[list], dict]:
    model = project_config.GENERATE_MODEL_NAME
    stream = False
    images_b64 = []
    qa_prompt = "Please answer the following question based on the provided images.\nThink step by step, use evidence from the images whenever possible, and give a clear, factual answer.\nIf the images are unclear or insufficient, say so briefly.\n\nQuestion: "
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if total_pages <= 50:
            page_indices = list(range(total_pages))
        else:
            page_indices = sorted(random.sample(range(total_pages), 50))
        logger.info(f"Processing PDF: {pdf_path}, Total pages: {total_pages}, Selected: {len(page_indices)}")
        for idx in page_indices:
            page = doc[idx]
            zoom = 144 / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_data = pix.tobytes("jpg")
            base64_str = base64.b64encode(img_data).decode("utf-8")
            images_b64.append(base64_str)
        doc.close()
    except Exception as e:
        logger.error(f"PDF Processing Error: {e}")
        raise e
    qa_prompt += question
    content = [{"type": "text", "text": qa_prompt}]
    for b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})  # type: ignore
    messages = [{"role": "user", "content": content}]

    chat = GPT()

    try:
        result, _, reasoning_content, _, token_usage = chat.send_chat_request(messages)
        return result, None, token_usage

    except Exception as e:
        logger.error(f"OpenAI Request Error: {e}")
        logger.debug(traceback.format_exc())
        raise e


async def process_single_sample(
    semaphore: asyncio.Semaphore,
    pdf_path: str,
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
            f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval_other_method/pure_model/MMLongBench_results/{timestamp}"
        )
    else:
        base_output_dir = f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval_other_method/pure_model/MMLongBench_results/{project_config.RESULT_DIR_NAME}"

    semaphore = asyncio.Semaphore(project_config.EVAL_CONCURRENCY_LIMIT)
    tasks = []

    for sample_data in dataset_info:
        qa_id = sample_data["qa"]["id"]
        middle_output_path = (
            f"{base_output_dir}/2_{qa_id}_{generate_stable_uuid_for_text(sample_data['qa']['question'])}.pkl"
        )
        pdf_path = os.path.join(project_config.project_root, "dataset", "MMLongBench-Doc", "documents", qa_id)
        if not os.path.exists(middle_output_path):
            task = asyncio.create_task(
                process_single_sample(
                    semaphore=semaphore,
                    pdf_path=pdf_path,
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
