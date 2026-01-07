import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config
import asyncio, nest_asyncio, math

# isort:skip
import traceback, time
from pathlib import Path
from src.langchain.embedding.huggingface import QwenEmbedding, VllmEmbedding
from src.langchain.embedding.openclip import PEEmbedding
from src.langchain.embedding.nomic import NomicEmbedding
from src.langchain.reranker.qwen3 import Qwen3CrossEncoder
from src.langchain.reranker.bge_reranker_v2_m3 import BgeCrossEncoder
from src.langchain.reranker.vllm_reranker import VLLMRemoteCrossEncoder
from src.langchain.bayes.bayes_rerank import BayesReranker
from src.langchain.retriever.bm25 import chroma2bm25
from src.langchain.utils.image import is_base64
from src.utils.gpt import GPT, extract_content_outside_think
from src.langchain.utils.rerank import add_to_final_results
from src.RAGAnything.kg import KG

from loguru import logger
from openai import OpenAI
from typing import Union, Tuple, Optional, List, Dict
from langchain_core.documents import Document


class RAGModels(object):
    def __init__(self):
        if project_config.EMBEDDING_MODEL_NAME.lower().startswith("vllm"):
            self.text_embedding = VllmEmbedding(database=project_config.TEXT_DATABASE)
        elif project_config.EMBEDDING_MODEL_NAME.lower().startswith("pe"):
            self.text_embedding = PEEmbedding(database=project_config.TEXT_DATABASE, collection_name="text")
        else:
            self.text_embedding = QwenEmbedding(database=project_config.TEXT_DATABASE)

        self.bm25_retriever = chroma2bm25(chroma=self.text_embedding.vectorstore_embd, k=project_config.BM25_K)

        if not project_config.ONLY_ADD_KG:
            if project_config.EMBEDDING_MODEL_NAME.lower().startswith("pe"):
                self.image_embedding = PEEmbedding(
                    database=project_config.IMAGE_DATABASE, embedding_model=self.text_embedding.embedding_model
                )
            else:
                self.image_embedding = PEEmbedding(database=project_config.IMAGE_DATABASE)

            if project_config.SHORT_CUT:
                self.shortcut_embedding = NomicEmbedding(database=project_config.SHORTCUT_DATABASE)

            if project_config.BAYES:
                self.bayes_model = BayesReranker()

            if project_config.RERANK:
                if project_config.RERANK_MODEL_NAME.startswith("Qwen3"):
                    self.reranker_model = Qwen3CrossEncoder()
                elif project_config.RERANK_MODEL_NAME.startswith("bge"):
                    self.reranker_model = BgeCrossEncoder()
                elif project_config.RERANK_MODEL_NAME.startswith("vllm"):
                    self.reranker_model = VLLMRemoteCrossEncoder()

        if project_config.BAYES and project_config.MODEL_PA_BY_KG:
            self.kg = KG()
        else:
            self.kg = None

        if project_config.GENERATE_MODEL_NAME.startswith("azure"):
            self.qa_agent = GPT(model=project_config.GENERATE_MODEL_NAME)
        elif project_config.GENERATE_MODEL_NAME.startswith("Qwen"):
            self.qa_agent = OpenAI(api_key=project_config.LLM_MODEL_API_KEY, base_url=project_config.QWEN3_VL_BASE_URL)
        else:
            self.qa_agent = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

    if project_config.ADD_VECTOR:

        async def aadd_vector_store_kg(self, pdf_path):
            from src.langchain.document_parse.mineru import MinerULoader

            if not os.path.exists(pdf_path):
                logger.error(f"File not found: {pdf_path}")
                return
            nest_asyncio.apply()

            if self.kg is not None:
                await self.kg.rag._ensure_lightrag_initialized()
            text_semaphore = asyncio.Semaphore(1)
            image_semaphore = asyncio.Semaphore(1)
            text_vec_semaphore = asyncio.Semaphore(1)
            image_vec_semaphore = asyncio.Semaphore(1)
            logger.debug(f"Starting processing for PDF: {pdf_path}")
            all_running_tasks = []

            async def _process_batch_vector(docs, batch_type, semaphore):
                """Add to embedding vector store"""
                if not docs or project_config.ONLY_ADD_KG:
                    return
                async with semaphore:
                    try:
                        logger.debug(f"[{batch_type}] Vector Store processing started. Size: {len(docs)}")
                        if batch_type.startswith("IMAGE"):
                            self.image_embedding.add_image_to_vectorstore(image_documents=docs)
                        elif batch_type.startswith("TEXT"):
                            self.text_embedding.add_text_to_vectorstore(documents=docs)
                        logger.debug(f"[{batch_type}] Vector Store processing finished.")
                    except Exception as e:
                        logger.error(f"Error in {batch_type} Vector Store task: {e}")

            async def _process_batch_kg(docs, batch_type, semaphore):
                """Add to Knowledge Graph"""
                if self.kg is None or not docs:
                    return
                async with semaphore:
                    try:
                        logger.debug(f"[{batch_type}] KG Batch processing started. Size: {len(docs)}")
                        await asyncio.gather(*[self.kg.add_documents_kg(doc) for doc in docs])
                        logger.debug(f"[{batch_type}] KG Batch processing finished.")
                    except Exception as e:
                        logger.error(f"Error in {batch_type} KG task: {e}")

            async def _process_shortcut(path):
                """Add PDF level shortcut embedding"""
                if not project_config.ONLY_ADD_KG:
                    if project_config.SHORT_CUT:
                        try:
                            logger.debug(f"[SHORTCUT] Adding PDF to shortcut vector store: {path}")
                            # await asyncio.to_thread(self.shortcut_embedding.add_pdf_to_vectorstore, path)
                            self.shortcut_embedding.add_pdf_to_vectorstore(path)
                            logger.debug(f"[SHORTCUT] Finished adding PDF to shortcut vector store.")
                        except Exception as e:
                            logger.error(f"Error in Shortcut Vector Store task: {e}")

            if self.kg is not None:
                pdf_kg_task = asyncio.create_task(self.kg.add_pdf_kg(pdf_path))
                all_running_tasks.append(pdf_kg_task)

            document_loader = MinerULoader(
                file_path=pdf_path,
                table_enable=False,
                backend=project_config.MINERU_BACKEND,
                server_url=project_config.MINERU_SERVER_URL,
                parse_result_dir=project_config.PARSE_RESULT_DIR,
            )
            doc_iterator = document_loader.lazy_load()
            text_batch = []
            image_batch = []
            total_docs_processed = 0
            previous_text_doucment = None

            while True:
                try:
                    document = next(doc_iterator)
                    total_docs_processed += 1

                    if is_base64(document.page_content):
                        image_batch.append(document)
                        if len(image_batch) >= project_config.ADD_VECTOR_IMAGE_BATCH_SIZE:
                            logger.debug(f"Triggering batch for {len(image_batch)} image documents...")
                            task_vec = asyncio.create_task(
                                _process_batch_vector(image_batch, "IMAGE", image_vec_semaphore)
                            )
                            all_running_tasks.append(task_vec)
                            task_kg = asyncio.create_task(_process_batch_kg(image_batch, "IMAGE", image_semaphore))
                            all_running_tasks.append(task_kg)
                            image_batch = []
                    else:
                        current_text_doucment = document
                        if previous_text_doucment is None:
                            previous_text_doucment = current_text_doucment
                            continue
                        prev_word_counts = document_loader.count_words(previous_text_doucment.page_content)
                        if prev_word_counts < project_config.MERGE_DOCUMENT_THRESHOLD:
                            merged_document = document_loader.merge_docs(
                                base_doc=previous_text_doucment, new_doc=current_text_doucment
                            )
                            if len(merged_document) == 1:
                                previous_text_doucment = merged_document[0]
                            else:
                                text_batch.append(previous_text_doucment)
                                previous_text_doucment = current_text_doucment
                        else:
                            text_batch.append(previous_text_doucment)
                            previous_text_doucment = current_text_doucment

                        if len(text_batch) >= project_config.ADD_VECTOR_TEXT_BATCH_SIZE:
                            logger.debug(f"Triggering batch for {len(text_batch)} text documents...")
                            task_vec = asyncio.create_task(
                                _process_batch_vector(text_batch, "TEXT", text_vec_semaphore)
                            )
                            all_running_tasks.append(task_vec)
                            task_kg = asyncio.create_task(_process_batch_kg(text_batch, "TEXT", text_semaphore))
                            all_running_tasks.append(task_kg)
                            text_batch = []

                except StopIteration:
                    logger.debug("Document stream finished. Processing final batches...")
                    if previous_text_doucment:
                        text_batch.append(previous_text_doucment)
                    if len(text_batch) > 0:
                        logger.debug(f"Adding final batch of {len(text_batch)} text documents...")
                        all_running_tasks.append(
                            asyncio.create_task(_process_batch_vector(text_batch, "TEXT_FINAL", text_vec_semaphore))
                        )
                        all_running_tasks.append(
                            asyncio.create_task(_process_batch_kg(text_batch, "TEXT_FINAL", text_semaphore))
                        )
                    if len(image_batch) > 0:
                        logger.debug(f"Adding final batch of {len(image_batch)} image documents...")
                        all_running_tasks.append(
                            asyncio.create_task(_process_batch_vector(image_batch, "IMAGE_FINAL", image_vec_semaphore))
                        )
                        all_running_tasks.append(
                            asyncio.create_task(_process_batch_kg(image_batch, "IMAGE_FINAL", image_semaphore))
                        )
                    break
                except Exception as e:
                    logger.error(f"Error while processing documents from {pdf_path}: {e}")
                    logger.debug(traceback.format_exc())
                    text_batch = []
                    image_batch = []
                    previous_text_doucment = None
                    continue

            logger.debug(f"Processed {total_docs_processed} documents from {pdf_path}")
            if total_docs_processed > 0:
                shortcut_task = asyncio.create_task(_process_shortcut(pdf_path))
                all_running_tasks.append(shortcut_task)
            else:
                logger.warning(f"No documents processed from {pdf_path}. Skipping shortcut vector store add.")
            if all_running_tasks:
                logger.info(f"Waiting for {len(all_running_tasks)} background (Vector & KG) tasks to complete...")
                await asyncio.gather(*all_running_tasks)
                logger.info("All tasks completed.")

    def custom_relevance_score_fn(self, distance: float) -> float:
        # relevance = 1.0 - distance / 2.0
        relevance = 1.0 / (1.0 + math.exp(distance - 1.0))
        return float(max(0.0, min(1.0, relevance)))

    # @workflow(name="retriever")
    async def retriever(self, query: str, start_time: float, result: Dict, if_shortcut=True):
        logger.debug(f"Retriever results")
        bm25_invoke_results = self.bm25_retriever.invoke(query)
        bm25_results = [{"result": doc, "score": None} for doc in bm25_invoke_results]
        bm25_end_time = time.time()
        result["time_record"]["bm25_cost"] = round(bm25_end_time - start_time, 4)
        logger.debug(f"BM25 retriever got results")

        text_results_with_smaller_scores = await self.text_embedding.vectorstore_embd.asimilarity_search_with_score(
            query, k=project_config.TEXT_K
        )
        text_results_with_score = []
        for doc, raw_score in text_results_with_smaller_scores:
            relevance_score = self.custom_relevance_score_fn(raw_score)
            text_results_with_score.append({"result": doc, "score": relevance_score})
        text_results_with_score.sort(key=lambda x: x["score"], reverse=True)
        text_end_time = time.time()
        result["time_record"]["text_cost"] = round(text_end_time - start_time, 4)
        logger.debug(f"Text retriever got {len(text_results_with_score)} results")
        if project_config.DEBUG:
            if text_results_with_score:
                logger.debug(f"Top Text score: {text_results_with_score[0]['score']}")
                logger.debug(f"results_with_smaller_scores Top Text score: {text_results_with_smaller_scores[0][1]}")
                logger.debug(f"results_with_smaller_scores Low Text score: {text_results_with_smaller_scores[-1][1]}")
        if project_config.BAYES and project_config.RERANK:
            text_results = text_results_with_score
            text_results.extend(bm25_results)
            text_results_for_rerank = []
            for text_result in text_results_with_score:
                text_results_for_rerank.append(text_result["result"])
            sorted_results = self.reranker_model.get_sorted_results_with_score(query, text_results_for_rerank)
            text_reranker_top_score = sorted_results[0][1]
            if project_config.BAYES and text_reranker_top_score < project_config.RERANK_THRESHOLD:
                sorted_results = sorted_results[: max(int(project_config.RERANK_TOP_K_FOR_BAYES / 10), 1)]
            else:
                sorted_results = sorted_results[: project_config.RERANK_TOP_K_FOR_BAYES]
            text_results = [{"result": doc, "score": score} for doc, score in sorted_results]
            text_rerank_end_time = time.time()
            result["time_record"]["text_rerank_cost"] = round(text_rerank_end_time - start_time, 4)
            logger.debug(f"Text reranker got {len(text_results)} results")
        else:
            text_results = text_results_with_score

        if project_config.BAYES and text_reranker_top_score < project_config.RERANK_THRESHOLD:
            image_results_with_smaller_scores = (
                await self.image_embedding.vectorstore_embd.asimilarity_search_with_score(
                    query, k=(project_config.IMAGE_K * 2)
                )
            )
        else:
            image_results_with_smaller_scores = (
                await self.image_embedding.vectorstore_embd.asimilarity_search_with_score(
                    query, k=project_config.IMAGE_K
                )
            )
        image_results = []
        for doc, raw_score in image_results_with_smaller_scores:
            relevance_score = self.custom_relevance_score_fn(raw_score)
            image_results.append({"result": doc, "score": relevance_score})
        image_results.sort(key=lambda x: x["score"], reverse=True)
        image_end_time = time.time()
        result["time_record"]["image_cost"] = round(image_end_time - start_time, 4)
        logger.debug(f"Image retriever got {len(image_results)} results")
        if project_config.DEBUG:
            if image_results:
                logger.debug(f"Top Image score: {image_results[0]['score']}")
                logger.debug(f"results_with_smaller_scores Top Image score: {image_results_with_smaller_scores[0][1]}")
                logger.debug(f"results_with_smaller_scores Low Image score: {image_results_with_smaller_scores[-1][1]}")

        shortcut_results = []
        if if_shortcut:
            if project_config.BAYES and text_reranker_top_score < project_config.RERANK_THRESHOLD:
                shortcut_results_with_smaller_scores = (
                    await self.shortcut_embedding.vectorstore_embd.asimilarity_search_with_score(
                        query, k=(project_config.SHORT_CUT_K * 5)
                    )
                )
            else:
                shortcut_results_with_smaller_scores = (
                    await self.shortcut_embedding.vectorstore_embd.asimilarity_search_with_score(
                        query, k=project_config.SHORT_CUT_K
                    )
                )
            for doc, raw_score in shortcut_results_with_smaller_scores:
                relevance_score = self.custom_relevance_score_fn(raw_score)
                shortcut_results.append({"result": doc, "score": relevance_score})
            shortcut_results.sort(key=lambda x: x["score"], reverse=True)
            shortcut_end_time = time.time()
            result["time_record"]["shortcut_cost"] = round(shortcut_end_time - start_time, 4)
            logger.debug(f"Shortcut retriever got {len(shortcut_results)} results")
            if project_config.DEBUG:
                if shortcut_results:
                    logger.debug(f"Top Shortcut score: {shortcut_results[0]['score']}")
                    logger.debug(
                        f"results_with_smaller_scores Top Shortcut score: {shortcut_results_with_smaller_scores[0][1]}"
                    )
                    logger.debug(
                        f"results_with_smaller_scores Low Shortcut score: {shortcut_results_with_smaller_scores[-1][1]}"
                    )

        # for res, score in text_results:
        #     logger.info(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
        # for res, score in image_results:
        #     logger.info(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
        return {
            "text_results": text_results,
            "image_results": image_results,
            "shortcut_results": shortcut_results,
            "bm25_results": bm25_results,
        }

    async def model_rerank(self, question, retriever_results, result):
        final_results = {
            # "document_id": {
            #     "page_idx": {
            #         "text_results": [],
            #         "image_results": [],
            #         "shortcut_results": [],
            #     }
            # }
        }
        ### rerank
        text_results = [text_result["result"] for text_result in retriever_results["text_results"]]
        bm25_results = [text_result["result"] for text_result in retriever_results["bm25_results"]]
        text_results.extend(bm25_results)
        if project_config.RERANK:
            sorted_results = self.reranker_model.get_sorted_results_with_score(question, text_results)
            for text_result, score in sorted_results:
                result["predict"]["provenance"]["before_rerank"].append(
                    {
                        "doc_id": Path(text_result.metadata["source"]).name,
                        "page_idx+1": int(text_result.metadata["page_idx"]) + 1,
                    }
                )
            result["middle_results"]["reranked_sorted_results"] = self.results_with_score_tuple_to_uuid_tuple(
                sorted_results
            )
            reranked_text_results = sorted_results[: project_config.RERANK_TOP_K]
        else:
            reranked_text_results = [text_result for text_result in text_results]
        ### add_to_final_results
        for reranked_text_result, rerank_score in reranked_text_results:
            add_to_final_results(
                {"result": reranked_text_result},
                "text_results",
                Path(reranked_text_result.metadata["source"]).name,
                reranked_text_result.metadata["page_idx"],
                final_results,
            )
        for image_result in retriever_results["image_results"]:
            add_to_final_results(
                image_result,
                "image_results",
                Path(image_result["result"].metadata["source"]).name,
                image_result["result"].metadata["page_idx"],
                final_results,
            )
        for shortcut_result in retriever_results["shortcut_results"]:
            add_to_final_results(
                shortcut_result,
                "shortcut_results",
                Path(shortcut_result["result"].metadata["source"]).name,
                shortcut_result["result"].metadata["page_idx"],
                final_results,
            )
        return final_results

    # @workflow(name="ask")

    def prepare_message(self, question: str, reranked_retriever_results: dict, result: dict, if_shortcut=True) -> list:
        content = list()
        question_prompt = f"""You are an intelligent assistant. Please summarize the content of the knowledge base (both in text, images and screenshots of pdf pages) to answer questions, think step by step.

First, analyze the keywords in the question, and then check whether all texts and images of the given knowledge base contain content related to the keywords in the question. Finally summarize and answer questions

Here is the question:
Question: {question}
The above is the question.

Here is the knowledge base:
    """
        ### text knowledge base
        for doc_id in reranked_retriever_results.keys():
            for page_idx in reranked_retriever_results[doc_id].keys():
                for text_result in reranked_retriever_results[doc_id][page_idx]["text_results"]:
                    question_prompt += f"The text knowledge on page {page_idx+1} of {doc_id} is as follows"
                    question_prompt += f"\n{text_result['result'].page_content}\n"
                    result["predict"]["provenance"]["text"].append(
                        {
                            "doc_id": Path(text_result["result"].metadata["source"]).name,
                            "page_idx+1": int(text_result["result"].metadata["page_idx"]) + 1,
                        }
                    )
                    result["predict"]["uuid"]["text"].append(text_result["result"].metadata["uuid"])
                ### image knowledge base
                for image_result in reranked_retriever_results[doc_id][page_idx]["image_results"]:
                    encoded_image = image_result["result"]
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image.page_content}"},
                        }
                    )
                    result["predict"]["provenance"]["image"].append(
                        {
                            "doc_id": Path(encoded_image.metadata["source"]).name,
                            "page_idx+1": int(encoded_image.metadata["page_idx"]) + 1,
                        }
                    )
                    result["predict"]["uuid"]["image"].append(encoded_image.metadata["uuid"])
                if if_shortcut:
                    for shortcut_result in reranked_retriever_results[doc_id][page_idx]["shortcut_results"]:
                        encoded_image = shortcut_result["result"]
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image.page_content}"},
                            }
                        )
                        result["predict"]["provenance"]["shortcut"].append(
                            {
                                "doc_id": Path(encoded_image.metadata["source"]).name,
                                "page_idx+1": int(encoded_image.metadata["page_idx"]) + 1,
                            }
                        )
                        result["predict"]["uuid"]["shortcut"].append(encoded_image.metadata["uuid"])
        question_prompt += "\nThe above is the knowledge base."
        content.append(
            {
                "type": "text",
                "text": question_prompt,
            }
        )
        messages = [{"role": "user", "content": content}]
        return messages

    async def ask(self, messages: list) -> Tuple[str, dict]:
        response = ""
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if project_config.IF_ASK:
            if project_config.GENERATE_MODEL_NAME.startswith("azure"):
                response, _, _, _, token_usage = self.qa_agent.send_chat_request(messages)  # type: ignore
            else:
                completion = self.qa_agent.chat.completions.create(  # type: ignore
                    model=project_config.GENERATE_MODEL_NAME,
                    messages=messages,
                    stream=False,
                    temperature=0.1,
                    max_completion_tokens=10240,
                    max_tokens=10240,
                )
                reasoning_content, response = extract_content_outside_think(completion.choices[0].message.content)
                if completion.usage:
                    token_usage = {
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "total_tokens": completion.usage.total_tokens,
                    }
        return response, token_usage

    def results_with_score_tuple_to_uuid_tuple(
        self, results_with_score: List[Tuple[Document, float]]
    ) -> List[Tuple[str, float]]:
        uuid_list_with_score = []
        for result, score in results_with_score:
            uuid_list_with_score.append((result.metadata["uuid"], score))
        return uuid_list_with_score

    def retriever_dict_to_uuid_dict(self, retriever_result: Dict) -> Dict:
        # "retriever_result": {"result": doc, "score": score}
        uuid_dict = {"result": retriever_result["result"].metadata["uuid"], "score": retriever_result["score"]}
        return uuid_dict

    def retriever_dict_in_list_to_uuid_list(self, retriever_result_list: List[Dict]) -> List:
        # "retriever_result_list": [{"result": doc, "score": score} for doc, score in sorted_results]
        uuid_list = []
        for retriever_result in retriever_result_list:
            uuid_list.append(self.retriever_dict_to_uuid_dict(retriever_result))
        return uuid_list

    def retriever_results_to_uuid_dict(self, retriever_results: Dict) -> Dict:
        # retriever_results = {
        #     "text_results": [{"result": doc, "score": score} for doc, score in sorted_results],
        #     "image_results": ,
        #     "shortcut_results": ,
        #     "bm25_results": [{"result": doc, "score": None} for doc in bm25_invoke_results],
        # }
        uuid_dict = {}
        for key in retriever_results.keys():
            uuid_dict[key] = self.retriever_dict_in_list_to_uuid_list(retriever_results[key])
        return uuid_dict

    async def uuid_dict_to_retriever_dict(self, uuid_result: Dict, type: str) -> Dict:
        doc = None
        score = uuid_result["score"]
        uuid = [uuid_result["result"]]
        if type == "text_results" or type == "bm25_results":
            doc = await self.text_embedding.vectorstore_embd.aget_by_ids(uuid)
        elif type == "image_results":
            doc = await self.image_embedding.vectorstore_embd.aget_by_ids(uuid)
        elif type == "shortcut_results":
            doc = await self.shortcut_embedding.vectorstore_embd.aget_by_ids(uuid)

        uuid_result = {"result": doc, "score": score}

        return uuid_result

    async def uuid_list_to_retriever_dict_in_list(self, uuid_results: List[Dict], type: str) -> List:
        retriever_result_list = []
        for uuid_result in uuid_results:
            retriever_result_list.append(await self.uuid_dict_to_retriever_dict(uuid_result, type))
        return retriever_result_list

    async def uuid_dict_to_retriever_results(self, uuid_dict: Dict) -> Dict:
        retriever_results = {}

        if "text_results" in uuid_dict.keys():
            if uuid_dict["text_results"]:
                retriever_results["text_results"] = await self.uuid_list_to_retriever_dict_in_list(
                    uuid_dict["text_results"], "text_results"
                )
        if "image_results" in uuid_dict.keys():
            if uuid_dict["image_results"]:
                retriever_results["image_results"] = await self.uuid_list_to_retriever_dict_in_list(
                    uuid_dict["image_results"], "image_results"
                )
        if "shortcut_results" in uuid_dict.keys():
            if uuid_dict["shortcut_results"]:
                retriever_results["shortcut_results"] = await self.uuid_list_to_retriever_dict_in_list(
                    uuid_dict["shortcut_results"], "shortcut_results"
                )
        if "bm25_results" in uuid_dict.keys():
            if uuid_dict["bm25_results"]:
                retriever_results["bm25_results"] = await self.uuid_list_to_retriever_dict_in_list(
                    uuid_dict["bm25_results"], "text_results"
                )

        return retriever_results
