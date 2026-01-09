import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config

import json
import asyncio
import traceback
import pickle
import datetime, time
import re
import copy
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


from src.eval.eval import print_config, _retriever, _ask
from src.utils.gpt import GPT, extract_content_outside_think
from src.utils.utils import save_pkl_result
from src.langchain.models.models import RAGModels
from src.dataset_loader.docbench import dataset as Dataset
from src.langchain.utils.uuid import generate_stable_uuid_for_text


def load_eval_prompt(
    EVAL_PROMPT_PATH: str = f"{project_config.project_root}/eval/third_party/DocBench/evaluation_prompt.txt",
):
    try:
        # with open(EVAL_PROMPT_PATH, "r", encoding="utf-8") as f:
        #     content = f.read()
        # return content
        return """
Task Overview:
You are tasked with evaluating user answers based on a given question, reference answer, and additional reference text. Your goal is to assess the correctness of the user answer using a specific metric.

Evaluation Criteria:
1. Yes/No Questions: Verify if the user's answer aligns with the reference answer in terms of a "yes" or "no" response.
2. Short Answers/Directives: Ensure key details such as numbers, specific nouns/verbs, and dates match those in the reference answer.
3. Abstractive/Long Answers: The user's answer can differ in wording but must convey the same meaning and contain the same key information as the reference answer to be considered correct.

Evaluation Process:
1. Identify the type of question presented.
2. Apply the relevant criteria from the Evaluation Criteria.
3. Compare the user's answer against the reference answer accordingly.
4. Consult the reference text for clarification when needed.
5. Score the answer with a binary label 0 or 1, where 0 denotes wrong and 1 denotes correct.
NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {{question}}
User Answer: {{sys_ans}}
Reference Answer: {{ref_ans}}
Reference Text: {{ref_text}}

Evaluation Form (score ONLY):
- Correctness:
"""
    except Exception as e:
        logger.error(f"Error reading evaluation prompt: {e}")
        return """
        Question: {{question}}
        System Answer: {{sys_ans}}
        Reference Answer: {{ref_ans}}
        Reference Evidence: {{ref_text}}
        Please evaluate the correctness.
        """


def read_dataset(
    full: bool = True, sample_path: str = f"{project_config.project_root}/eval/DocBench/test_dataset.json"
):
    if full:
        dataset = Dataset()
        dataset_info = dataset.read_dataset()
    else:
        with open(sample_path, "r") as f:
            dataset_info = json.load(f)
        # dataset_info = dataset_info[:4]  # for quick test
    return dataset_info


def analysis(results: list):
    total_score = 0
    valid_count = 0
    for res in results:
        if "score" in res:
            total_score += res["score"]
            valid_count += 1

    avg_score = total_score / valid_count if valid_count > 0 else 0
    logger.info(f"Average Score: {avg_score}")


async def _evaluate(
    question: str,
    predict_answer: str,
    ground_truth: str,
    result: dict,
    sample_data: dict,
    DOCBENCH_EVAL_PROMPT_TEMPLATE: str,
    token_usage: dict,
):
    ref_text = sample_data["qa"].get("evidence", "")
    if not ref_text and "provenance" in sample_data["qa"]:
        ref_text = "No explicit text evidence provided."
    cur_prompt = (
        DOCBENCH_EVAL_PROMPT_TEMPLATE.replace("{{question}}", str(question))
        .replace("{{sys_ans}}", str(predict_answer))
        .replace("{{ref_ans}}", str(ground_truth))
        .replace("{{ref_text}}", str(ref_text))
    )

    judge_agent = OpenAI(
        base_url=project_config.JUDGE_AGENT_BASE_URL,
        api_key=project_config.JUDGE_AGENT_API_KEY,
        timeout=3600,
    )
    messages = [
        {"role": "system", "content": "You are a helpful and objective evaluator."},
        {"role": "user", "content": cur_prompt},
    ]

    try:
        completion = judge_agent.chat.completions.create(
            model=project_config.JUDGE_AGENT_MODEL_NAME,  # type: ignore
            messages=messages,  # type: ignore
            max_tokens=2048,
        )
        _, response_content = extract_content_outside_think(str(completion.choices[0].message.content))

        try:
            score = float(response_content)
        except Exception as e:
            logger.warning(f"Failed to parse float from Judge response: {response_content}")
            match = re.search(r"\b(100|[1-9]?[0-9])\b", response_content)
            score = int(match.group(1)) if match else 0

    except Exception as e:
        logger.error(f"Error during LLM Judge evaluation: {e}")
        score = 0

    result["middle_results"]["judge_res"] = response_content
    result["token_usage"] = token_usage

    result["predict"]["predict_answer"] = predict_answer
    result["score"] = score
    result["pred"] = predict_answer
    result["answer"] = ground_truth
    logger.info(f"pred_ans: {predict_answer}")
    logger.info(f"ground_truth: {ground_truth}")
    logger.info(f"Judge Score: {score}")


async def process_single_sample(
    semaphore: asyncio.Semaphore,
    models: RAGModels,
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
        result["predict"]["uuid"] = {"text": [], "image": [], "shortcut": []}
        result["middle_results"] = {}

        # question = f"Question about {sample_data['qa']['id']}: {sample_data['qa']['question']}"
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
                        DOCBENCH_EVAL_PROMPT_TEMPLATE=DOCBENCH_EVAL_PROMPT_TEMPLATE,
                        token_usage=token_usage,
                    )

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if models.kg is not None:
        await models.kg.rag._ensure_lightrag_initialized()

    if project_config.ADD_VECTOR:
        logger.info("Starting Vector Store Ingestion for DocBench...")
        processed_ids = set()
        for item in dataset_info:
            doc_id = item["qa"]["id"]
            folder_id = item["qa"]["num_id"]
            if doc_id in processed_ids:
                continue
            pdf_path = os.path.join(
                f"{project_config.project_root}/dataset/DocBench/{folder_id}",
                doc_id,
            )
            if not os.path.exists(pdf_path):
                logger.error(f"PDF not found: {pdf_path}")
                continue
            retry_count = 0
            while True:
                try:
                    logger.info(f"Processing PDF: {doc_id} -> {pdf_path}")
                    await models.aadd_vector_store_kg(pdf_path=pdf_path)
                    logger.info(f"Successfully finished: {doc_id}")
                    processed_ids.add(doc_id)
                    break
                except KeyboardInterrupt:
                    return
                except Exception as e:
                    logger.error(f"Error processing {doc_id}: {e}")
                    retry_count += 1
                    if retry_count > 5:
                        logger.error(f"Skipping {doc_id} after too many retries.")
                        break
                    await asyncio.sleep(1)
        return

    if project_config.RESULT_DIR_NAME == "timestamp":
        base_output_dir = f"{project_config.project_root}/eval/DocBench/results/{timestamp}"
    else:
        base_output_dir = f"{project_config.project_root}/eval/DocBench/results/{project_config.RESULT_DIR_NAME}"

    faild_idx = []
    faild_path = f"{project_config.project_root}/eval/count_result/docbench_failed_case_study.json"
    if Path(faild_path).exists() and project_config.TEST_FAILED_CASE:
        with open(faild_path, "r") as f:
            faild_cases = json.load(f)
            for faild_case in faild_cases:
                faild_idx.append(faild_case["index"])
    idx = 0

    DOCBENCH_EVAL_PROMPT_TEMPLATE = load_eval_prompt()
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
                    DOCBENCH_EVAL_PROMPT_TEMPLATE=DOCBENCH_EVAL_PROMPT_TEMPLATE,
                    middle_output_path=middle_output_path,
                )
            )
        else:
            task = asyncio.create_task(read_pkl(middle_output_path))
            logger.debug(f"Loading existing result for {qa_id} from {middle_output_path}")
        idx += 1
        tasks.append(task)

    logger.info(f"Starting concurrent evaluation of {len(tasks)} samples on DocBench...")
    results = await asyncio.gather(*tasks)
    analysis(results)

    if (project_config.TEST_RAG and project_config.IF_ASK) or project_config.WRITE_RESULTS:
        os.makedirs(base_output_dir, exist_ok=True)
        simplified_results = []
        for result in results:
            simple_result = copy.deepcopy(result)
            simple_result.pop("middle_results", None)
            simplified_results.append(simple_result)
        simplified_output_path = f"{base_output_dir}/1_simplified_results.json"
        with open(simplified_output_path, "w", encoding="utf-8") as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        logger.info(f"split results saved to directory: {base_output_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
