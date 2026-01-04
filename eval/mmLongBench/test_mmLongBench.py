import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config

from eval.third_party.MMLongBench.eval.eval_score import eval_score, eval_acc_and_f1
from src.eval.eval import print_config, filter_doc_id, cal_recall, _analysis_kg_retriever_results, _retriever, _ask
from src.eval.additional_analysis import (
    analysis_recall,
    addi_get_recall_middle_result,
    analysis_provenance,
    addi_get_provenance_middle_result,
)
from src.utils.unigpt import GPT, extract_content_outside_think
from src.utils.utils import save_pkl_result
from src.dataset_loader.mmlongbench import dataset as Dataset
from src.langchain.models.models import RAGModels
from src.langchain.utils.uuid import generate_stable_uuid_for_text
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict
from openai import OpenAI
import asyncio, traceback, pickle
import base64
import datetime, time
import re
import copy
import json
import uuid

# from langfuse import Langfuse
# langfuse = Langfuse(
#     secret_key=project_config.LANGFUSE _SECRET_KEY,
#     public_key=project_config.LANGFUSE_PUBLIC_KEY,
#     host=project_config.TRACELOOP_BASE_URL,
# )

# from traceloop.sdk import Traceloop
# from traceloop.sdk.decorators import workflow, task

# Traceloop.init(disable_batch=True)


def read_dataset(
    full=True, sample_path=f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval/mmLongBench/test_dataset.json"
):
    if full:
        dataset = Dataset()
        dataset_info = dataset.read_dataset()
    else:
        with open(sample_path, "r") as f:
            dataset_info = json.load(f)
            # dataset_info = dataset_info[:4]  # HACK for quick test
    return dataset_info


# @task(name="extract_answer")
async def extract_answer(question, output) -> str:
    # answer_agent = GPT(model="qwen3_32b", vendor="unisound", stream=False, temperature=0.2)
    # answer_agent = GPT(model="azure-gpt-4o", vendor="azure", stream=False, temperature=0.2)
    answer_agent = OpenAI(
        base_url=project_config.QWEN3_VL_BASE_URL,
        api_key=project_config.API_KEY,
        timeout=3600,
    )
    with open(
        f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval/mmLongBench/prompt_for_answer_extraction.md",
        "r",
    ) as f:
        prompt = f.read()
        prompt += "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
        # {"role": "assistant", "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)},
    ]
    # response, _, _, _, token_usage = answer_agent.send_chat_request(messages=messages)
    completion = answer_agent.chat.completions.create(
        model=project_config.LLM_MODEL_NAME,  # type: ignore
        messages=messages,  # type: ignore
        max_tokens=2048,
    )
    _, response = extract_content_outside_think(str(completion.choices[0].message.content))
    return response


def analysis(results: List[Dict]):
    # mmLongBench acc and f1
    acc, f1 = eval_acc_and_f1(results)
    gpt_acc = 0.0
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
        gpt_acc += result.get("gpt_score", 0.0)
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
    logger.info("Avg acc: {}".format(acc))
    logger.info("Avg gpt acc: {}".format(gpt_acc / len(results)))
    logger.info("Avg f1: {}".format(f1))
    logger.info("Provenance recall: {}".format(overall_scores))
    if project_config.BAYES and project_config.MODEL_PA_BY_KG:
        logger.info("KG Provenance recall: {}".format(kg_overall_scores))


async def gpt_acc(question, predict_answer, ground_truth, answer_type) -> Tuple[float, str]:
    prompt = """
You are required to determine if a predicted answer is correct or can reasonably answer the question compared to the ground truth. The question will be placed within <question></question> tags, answer type will be placed within <type></type> tags, predicted answer will be placed within <predict></predict> tags, and the ground truth answer will be placed within <gt></gt> tags.

You must output the final score as a floating-point number (0.0 to 1.0) enclosed within `<answer>` tags.

Output Format: Output **only** the final numerical score inside the tags. Do not provide reasoning.

Example:
<answer>1.0</answer>

Question: <question>{{question}}</question>

Predict Answer: <predict>{{sys_ans}}</predict>

Ground Truth: <gt>{{ref_ans}}</gt>
"""
    # judge_agent = GPT(model="qwen3_32b", vendor="unisound", stream=False, temperature=0.2)
    judge_agent = OpenAI(
        base_url=project_config.QWEN3_VL_BASE_URL,
        api_key=project_config.API_KEY,
        timeout=3600,
    )
    cur_prompt = (
        prompt.replace("{{question}}", str(question))
        .replace("{{sys_ans}}", str(predict_answer))
        .replace("{{ref_ans}}", str(ground_truth))
    )
    messages = [
        {"role": "system", "content": "You are a helpful and objective evaluator."},
        {"role": "user", "content": cur_prompt},
    ]
    try:
        # response_content, _, _, _, token_usage = judge_agent.send_chat_request(messages=messages)
        completion = judge_agent.chat.completions.create(
            model=project_config.LLM_MODEL_NAME,  # type: ignore
            messages=messages,  # type: ignore
            max_tokens=2048,
        )
        _, response_content = extract_content_outside_think(str(completion.choices[0].message.content))
        match = re.search(r"<answer>\s*([\d\.]+)\s*</answer>", response_content)

        if match:
            try:
                score = float(match.group(1))
                if score > 1.0:
                    score = 1.0
                elif score < 0.0:
                    score = 0.0
            except ValueError:
                logger.warning(f"Failed to parse float from extracted content: {match.group(1)}")
                score = 0.0
        else:
            logger.warning(f"No <answer> tags found in Judge response: {response_content}")
            fallback_match = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", response_content)
            score = float(fallback_match.group(0)) if fallback_match else 0.0
    except Exception as e:
        logger.error(f"Error during LLM Judge evaluation: {e}")
        score = 0.0

    return score, response_content


async def _evaluate(
    question: str, predict_answer: str, ground_truth: str, result: dict, sample_data: dict, token_usage: dict
):
    extracted_res = await extract_answer(question=question, output=predict_answer)

    ### mmLongBench eval
    pred_ans = ""
    # pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
    pattern = re.compile(r"Extracted answer:(.*?)(?:Answer format:|$)", re.DOTALL)
    match = pattern.search(extracted_res)
    if match:
        pred_ans = match.group(1).strip()
    score = eval_score(gt=ground_truth, pred=pred_ans, answer_type=sample_data["qa"]["answer_format"])

    ### gpt score
    gpt_score, gpt_acc_response_content = await gpt_acc(
        question=question,
        predict_answer=pred_ans,
        ground_truth=ground_truth,
        answer_type=sample_data["qa"]["answer_format"],
    )
    if score > gpt_score:
        gpt_score = score

    ### log
    result["predict"]["predict_answer"] = predict_answer
    result["predict"]["extracted_res"] = extracted_res
    result["score"] = score
    result["gpt_score"] = gpt_score
    result["middle_results"]["judge_res"] = gpt_acc_response_content
    result["token_usage"] = token_usage
    result["pred"] = pred_ans
    result["answer"] = ground_truth
    logger.info(f"pred_ans: {pred_ans}")
    logger.info(f"ground_truth: {ground_truth}")
    logger.info(f"score: {score}")
    logger.info(f"gpt_score: {gpt_score}")


async def process_single_sample(
    semaphore: asyncio.Semaphore,
    models: RAGModels,
    sample_data: dict,
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
        result["predict"]["uuid"] = {"text": [], "image": [], "shortcut": []}
        result["middle_results"] = {}

        question = sample_data["qa"]["question"]
        ground_truth = sample_data["qa"]["ground_truth"]

        try:
            if project_config.TEST_RAG:
                reranked_retriever_results = await _retriever(models=models, question=question, result=result)
                messages = models.prepare_message(
                    question,
                    reranked_retriever_results,
                    result,
                    if_shortcut=project_config.SHORT_CUT,
                )

                if project_config.IF_ASK:
                    ask_start_time = time.time()
                    predict_answer, token_usage = await _ask(models=models, messages=messages)
                    ask_end_time = time.time()
                    result["time_record"]["qa_cost"] = round(ask_end_time - ask_start_time, 4)
                    await _evaluate(
                        question=question,
                        predict_answer=predict_answer,
                        ground_truth=ground_truth,
                        result=result,
                        sample_data=sample_data,
                        token_usage=token_usage,
                    )
            await cal_recall(result=result, models=models)

            # result.pop("middle_results", None)
            save_pkl_result(result, middle_output_path)

            return result
        except Exception as e:
            logger.error(f"Error processing sample {sample_data.get('qa', {}).get('id')}: {e}")
            logger.debug(traceback.format_exc())
            result["error"] = str(e)
            return result


async def read_pkl(middle_output_path):
    result = pickle.loads(Path(middle_output_path).read_bytes())
    result.pop("middle_results", None)
    return result


async def main():
    print_config()
    models = RAGModels()
    dataset_info = read_dataset(full=project_config.TEST_BENCH_FULL)
    timestamp = datetime.datetime.now().isoformat()
    if models.kg is not None:
        await models.kg.rag._ensure_lightrag_initialized()

    if project_config.ADD_VECTOR:
        pdf_names = []
        seen = set()
        for item in dataset_info:
            if "qa" in item and "id" in item["qa"]:
                pdf_id = item["qa"]["id"]
                if pdf_id not in seen:
                    pdf_names.append(pdf_id)
                    seen.add(pdf_id)

        for pdf_name in pdf_names:
            pdf_path = os.path.join(
                f"{project_config.DATA_ROOT}/projects/MRAG3.0/dataset/MMLongBench-Doc/documents",
                pdf_name,
            )
            retry_count = 0
            while True:
                try:
                    logger.info(f"Processing PDF: {pdf_name}")
                    await models.aadd_vector_store_kg(pdf_path=pdf_path)
                    logger.info(f"Successfully finished: {pdf_name}")
                    break
                except KeyboardInterrupt:
                    logger.error("User interrupted the process.")
                    return
                except Exception as e:
                    logger.error(f"Error processing {pdf_name}: {e}")
                    retry_count += 1
                    if retry_count > 5:
                        logger.error(f"Skipping {pdf_name} after too many retries.")
                        break
                    logger.warning("Sleeping for 1 seconds before retrying...")
                    await asyncio.sleep(1)
                    continue
        return

    if project_config.RESULT_DIR_NAME == "timestamp":
        base_output_dir = f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval/mmLongBench/results/{timestamp}"
    else:
        base_output_dir = (
            f"{project_config.DATA_ROOT}/projects/MRAG3.0/eval/mmLongBench/results/{project_config.RESULT_DIR_NAME}"
        )

    faild_idx = []
    faild_path = "{project_config.project_root}/eval/count_result/mmlongbench_failed_case_study.json"
    with open(faild_path, "r") as f:
        faild_cases = json.load(f)
        for faild_case in faild_cases:
            faild_idx.append(faild_case["index"])
    idx = 0

    semaphore = asyncio.Semaphore(project_config.EVAL_CONCURRENCY_LIMIT)
    tasks = []
    for sample_data in dataset_info:
        qa_id = sample_data["qa"]["id"]
        middle_output_path = (
            f"{base_output_dir}/2_{qa_id}_{generate_stable_uuid_for_text(sample_data['qa']['question'])}.pkl"
        )
        if not os.path.exists(middle_output_path) or idx in faild_idx:
            task = asyncio.create_task(
                process_single_sample(
                    semaphore=semaphore,
                    models=models,
                    sample_data=sample_data,
                    middle_output_path=middle_output_path,
                )
            )
        elif project_config.ADDITIONAL_ANALYSIS:
            if project_config.ADDITIONAL_ANALYSIS_TYPE.lower() == "recall":
                task = asyncio.create_task(
                    addi_get_recall_middle_result(
                        middle_output_path, models, top_k=int(os.getenv("ADDITIONAL_ANALYSIS_RECALL_TOPK", 20))
                    )
                )
            elif project_config.ADDITIONAL_ANALYSIS_TYPE.lower() == "provenance":
                task = asyncio.create_task(
                    addi_get_provenance_middle_result(
                        middle_output_path, models=models, top_k=int(os.getenv("ADDITIONAL_ANALYSIS_RECALL_TOPK", 20))
                    )
                )
        else:
            # task = asyncio.create_task(
            #     asyncio.to_thread(lambda path=middle_output_path: pickle.loads(Path(path).read_bytes()))
            # )
            task = asyncio.create_task(read_pkl(middle_output_path))
            logger.debug(f"Loading existing result for {qa_id} from {middle_output_path}")
        idx += 1
        tasks.append(task)

    logger.info(f"Starting concurrent evaluation of {len(tasks)} samplescon mmLongBench...")
    results = await asyncio.gather(*tasks)
    logger.info(f"Fininshed valuation of {len(tasks)} samplescon mmLongBench")
    if project_config.ADDITIONAL_ANALYSIS:
        if project_config.ADDITIONAL_ANALYSIS_TYPE.lower() == "recall":
            await analysis_recall(results=results, models=models)
        elif project_config.ADDITIONAL_ANALYSIS_TYPE.lower() == "provenance":
            analysis_provenance(results=results)
    else:
        analysis(results)
    logger.info(f"Fininshed analysis")

    if (project_config.TEST_RAG and project_config.IF_ASK) or project_config.WRITE_RESULTS:
        os.makedirs(base_output_dir, exist_ok=True)
        for result in results:
            result.pop("middle_results", None)
        simplified_output_path = f"{base_output_dir}/1_simplified_results.json"
        with open(simplified_output_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Split results saved to directory: {base_output_dir}")


# 16 + 37.83 G
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
