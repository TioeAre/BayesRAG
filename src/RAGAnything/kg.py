import os, sys, math
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config
from src.langchain.utils.image import is_base64
from src.langchain.utils.uuid import generate_stable_uuid_for_text
from src.RAGAnything.model_func import (
    llm_model_func_qwen3,
    vision_model_func_qwen3vl,
    llm_model_func_gpt4o,
    vision_model_func_gpt4o,
    embedding_func,
    rerank_model_func,
)

from loguru import logger
from pathlib import Path
from raganything import RAGAnything, RAGAnythingConfig

import asyncio, nest_asyncio, functools, json
import tempfile
import fitz  # PyMuPDF

from dataclasses import asdict
from functools import partial
from lightrag import QueryParam
from lightrag.utils import logger as lightrag_logger
from lightrag.operate import (
    get_keywords_from_query,
    _perform_kg_search,
)


class KG(object):
    def __init__(self) -> None:
        self.config = RAGAnythingConfig(
            working_dir=os.path.join(project_config.project_root, "database", "raganything_db"),  # type: ignore
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        lightrag_kwargs = {
            "kv_storage": "MongoKVStorage",
            "vector_storage": "MongoVectorDBStorage",
            # "graph_storage": "NetworkXStorage",  # "Neo4JStorage"
            "graph_storage": "Neo4JStorage",
            "doc_status_storage": "MongoDocStatusStorage",
            "llm_model_max_async": project_config.RAGANYTHING_LLM_MODEL_MAX_ASYNC,
            "summary_context_size": project_config.SUMMARY_CONTEXT_SIZE,
        }
        if project_config.RAGANYTHING_RERANK:
            lightrag_kwargs["rerank_model_func"] = rerank_model_func

        if project_config.IF_GPT4o:
            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=llm_model_func_gpt4o,
                vision_model_func=vision_model_func_gpt4o,
                embedding_func=embedding_func,
                lightrag_kwargs=lightrag_kwargs,
            )
        else:
            self.rag = RAGAnything(
                config=self.config,
                llm_model_func=llm_model_func_qwen3,
                vision_model_func=vision_model_func_qwen3vl,
                embedding_func=embedding_func,
                lightrag_kwargs=lightrag_kwargs,
            )

    def _render_and_save_page(self, doc, page_num, image_path):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        pix.save(image_path)
        return image_path

    async def doc_uuid_exist(self, doc_uuid):
        try:
            tasks = [
                self.rag.lightrag.full_entities.get_by_id(doc_uuid),  # type: ignore
                self.rag.lightrag.full_docs.get_by_id(doc_uuid),  # type: ignore
                self.rag.lightrag.full_relations.get_by_id(doc_uuid),  # type: ignore
                self.rag.lightrag.doc_status.get_by_id(doc_uuid),  # type: ignore
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    continue
                if res is not None:
                    return True
            return False
        except Exception as e:
            print(f"Error checking doc_uuid existence: {e}")
            return False

    async def add_documents_kg(self, document):
        if is_base64(document.page_content):
            content = {
                "type": "image",
                "img_path": document.metadata["image_path"],
                "page_idx": document.metadata["page_idx"],
            }
        else:
            content = {
                "type": "text",
                "text": document.page_content,
                "page_idx": document.metadata["page_idx"],
            }
        doc_uuid = document.metadata["uuid"]
        await self.rag._ensure_lightrag_initialized()
        doc_status = await self.doc_uuid_exist(doc_uuid)  # type: ignore
        if not doc_status:
            logger.info(f"{doc_uuid}, {document.metadata['source']}, {content['type']}, to insert")
            await self.rag.insert_content_list(
                content_list=[content],
                file_path=document.metadata["source"],
                split_by_character=None,
                split_by_character_only=False,
                doc_id=doc_uuid,
                display_stats=True,
            )
        else:
            logger.info(f"{doc_uuid}, {document.metadata['source']}, {content['type']}, already inserted")

    async def add_pdf_kg(
        self, pdf_path: str, start_page=None, end_page=None, tmp_dir=f"{project_config.DATA_ROOT}/projects/MRAG3.0/tmp"
    ):
        def chunked(iterable, n):
            for i in range(0, len(iterable), n):
                yield iterable[i : i + n]

        await self.rag._ensure_lightrag_initialized()
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        os.makedirs(tmp_dir, exist_ok=True)
        loop = asyncio.get_running_loop()
        doc = fitz.open(pdf_path)
        try:
            total_pages = len(doc)
            if start_page is None or start_page < 0:
                start_page = 0
            if end_page is None or end_page >= total_pages:
                end_page = total_pages - 1
            pdf_name = Path(pdf_path).name
            page_range = list(range(start_page, end_page + 1))
            with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
                for batch_pages in chunked(page_range, project_config.ADD_KG_PDF_BATCH_SIZE):
                    batch_tasks = []
                    for p_num in batch_pages:
                        uuid = generate_stable_uuid_for_text(f"{pdf_name}_{p_num}")
                        batch_tasks.append((p_num, uuid))
                    status_checks = [self.doc_uuid_exist(uuid) for _, uuid in batch_tasks]  # type: ignore
                    statuses = await asyncio.gather(*status_checks)
                    pages_to_process = []
                    for (p_num, uuid), exists in zip(batch_tasks, statuses):
                        if not exists:
                            pages_to_process.append((p_num, uuid))
                    if not pages_to_process:
                        logger.info(f"{batch_tasks}, {pdf_name}, already inserted")
                        continue
                    render_tasks = []
                    for p_num, uuid in pages_to_process:
                        image_path = os.path.join(temp_dir, f"page_{p_num + 1}.png")
                        task = loop.run_in_executor(
                            None, functools.partial(self._render_and_save_page, doc, p_num, image_path)
                        )
                        render_tasks.append(task)
                    saved_image_paths = await asyncio.gather(*render_tasks)
                    content_list = []

                    for (p_num, uuid), img_path in zip(pages_to_process, saved_image_paths):
                        content_list.append(
                            {
                                "type": "image",
                                "img_path": img_path,
                                "page_idx": p_num,
                                "__uuid__": uuid,  # 临时存一下 ID
                            }
                        )

                    insert_tasks = []
                    for item in content_list:
                        doc_id = item.pop("__uuid__")
                        logger.info(f"{doc_id}, {pdf_name}, to insert")
                        insert_tasks.append(
                            self.rag.insert_content_list(
                                content_list=[item],
                                file_path=pdf_path,
                                split_by_character=None,
                                split_by_character_only=False,
                                doc_id=doc_id,
                                display_stats=True,
                            )
                        )
                    await asyncio.gather(*insert_tasks)
        finally:
            doc.close()

    async def query(self, query: str, mode="hybrid") -> dict:

        await self.rag._ensure_lightrag_initialized()
        self.rag.logger.info(f"Executing VLM enhanced query: {query[:100]}...")
        if hasattr(self.rag, "_current_images_base64"):
            delattr(self.rag, "_current_images_base64")
        # 1. Get original retrieval prompt (without generating final answer)
        query_param = QueryParam(mode=mode, only_need_prompt=True)  # type: ignore
        lightrag_logger.debug(f"[aquery_llm] Query param: {query_param}")
        lightrag_config = asdict(self.rag.lightrag)  # type: ignore

        if query_param.model_func:
            use_model_func = query_param.model_func
        else:
            use_model_func = lightrag_config["llm_model_func"]
            use_model_func = partial(use_model_func, _priority=5)

        # NOTE: 使用 llm 提取 keywords
        hl_keywords, ll_keywords = await get_keywords_from_query(
            query, query_param, lightrag_config, self.rag.lightrag.llm_response_cache  # type: ignore
        )

        lightrag_logger.debug(f"High-level keywords: {hl_keywords}")
        lightrag_logger.debug(f"Low-level  keywords: {ll_keywords}")

        # Handle empty keywords
        if ll_keywords == [] and query_param.mode in ["local", "hybrid", "mix"]:
            lightrag_logger.warning("low_level_keywords is empty")
        if hl_keywords == [] and query_param.mode in ["global", "hybrid", "mix"]:
            lightrag_logger.warning("high_level_keywords is empty")
        if hl_keywords == [] and ll_keywords == []:
            if len(query) < 50:
                lightrag_logger.warning(f"Forced low_level_keywords to origin query: {query}")
                ll_keywords = [query]
            # else:
            #     return None

        ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
        hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

        # Stage 1: Pure search
        # NOTE: knowledge search
        search_result = await _perform_kg_search(
            query,
            ll_keywords_str,
            hl_keywords_str,
            self.rag.lightrag.chunk_entity_relation_graph,  # type: ignore
            self.rag.lightrag.entities_vdb,  # type: ignore
            self.rag.lightrag.relationships_vdb,  # type: ignore
            self.rag.lightrag.text_chunks,  # type: ignore
            query_param,
            self.rag.lightrag.chunks_vdb,  # type: ignore
        )

        return search_result

    async def cal_prob_from_relations(self, search_result: dict, scaling_factor: float = 0.1) -> dict:
        """calculate probability between two entity from search results from knowledge graph

        Parameters
        ----------
        search_result : dict
            search results from knowledge graph
        scaling_factor : float, optional
            hyperparameter, by default 0.5

        Returns
        -------
        dict
            probability dicts
        """
        final_entities = search_result["final_entities"]
        final_relations = search_result["final_relations"]

        # mapping entity to chunk_id
        entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)  # [entity]: set("chunk_id")
        for entity in final_entities:
            entity_name = entity.get("entity_name")
            source_id_str = entity.get("source_id")  # source_id actual as chunk_id
            if entity_name and source_id_str:
                # entity_to_chunks[entity_name].add(source_id)
                ids = [x for x in source_id_str.split("<SEP>") if x]
                entity_to_chunks[entity_name].update(ids)

        pair_stats = defaultdict(lambda: {"score": 0.0, "evidence": []})
        valid_relations_count = 0
        chunk_doc_map: Dict[str, str | None] = {}

        # process relations to score chunks
        for relation in final_relations:
            src_entity, tgt_entity = None, None
            if "src_tgt" in relation and isinstance(relation["src_tgt"], (list, tuple)):
                src_entity, tgt_entity = relation["src_tgt"]
            else:
                src_entity, tgt_entity = relation.get("src_id"), relation.get("tgt_id")
            if not src_entity or not tgt_entity:
                continue
            weight = float(relation.get("weight", 1.0))
            description = relation.get("description", "No description")
            source_chunks = entity_to_chunks.get(src_entity, set())  # get the set() of chunk_id
            target_chunks = entity_to_chunks.get(tgt_entity, set())
            if not source_chunks or not target_chunks:
                continue
            valid_relations_count += 1
            for s_chunk_id in source_chunks:
                for t_chunk_id in target_chunks:
                    if s_chunk_id == t_chunk_id:
                        continue

                    # NOTE: convert chunk_id to doc_id (uuid)
                    if s_chunk_id not in chunk_doc_map:
                        s_chunk = await self.rag.lightrag.chunks_vdb.get_by_id(s_chunk_id)  # type: ignore
                        if s_chunk:
                            chunk_doc_map[s_chunk_id] = s_chunk.get("full_doc_id")
                        else:
                            chunk_doc_map[s_chunk_id] = None
                    if t_chunk_id not in chunk_doc_map:
                        t_chunk = await self.rag.lightrag.chunks_vdb.get_by_id(t_chunk_id)  # type: ignore
                        if t_chunk:
                            chunk_doc_map[t_chunk_id] = t_chunk.get("full_doc_id")
                        else:
                            chunk_doc_map[t_chunk_id] = None
                    s_doc_id = chunk_doc_map.get(s_chunk_id)
                    t_doc_id = chunk_doc_map.get(t_chunk_id)
                    if s_doc_id and t_doc_id:
                        pair_key = tuple(sorted((s_doc_id, t_doc_id)))
                        if project_config.MODEL_PA_BY_KG_WEIGHT:
                            pair_stats[pair_key]["score"] += weight  # type: ignore
                        else:
                            pair_stats[pair_key]["score"] += 1  # type: ignore
                        pair_stats[pair_key]["evidence"].append(  # type: ignore
                            {
                                "src_entity": src_entity,
                                "tgt_entity": tgt_entity,
                                "relation_desc": description,
                                "weight": weight,
                            }
                        )

        logger.debug(
            f"Processed {len(final_relations)} relations, found {valid_relations_count} valid connections between chunks."
        )

        # normalization
        connectivity_map = {}
        for pair, stats in pair_stats.items():
            doc_a, doc_b = pair
            raw_score = stats["score"]
            # probability = 1 - e^(-k * Score)
            probability = 1.0 - math.exp(-raw_score * scaling_factor)  # type: ignore
            connection_details = {
                "probability": round(probability, 4),
                "raw_score": raw_score,
                "evidence_count": len(stats["evidence"]),  # type: ignore
                # "top_evidence": stats["evidence"][: getattr(project_config, "KG_TOP_EVIDENCE", 3)],  # type: ignore
            }
            if doc_a not in connectivity_map.keys():
                connectivity_map[doc_a] = {}
            if doc_b not in connectivity_map.keys():
                connectivity_map[doc_b] = {}
            connectivity_map[doc_a][doc_b] = connection_details
            connectivity_map[doc_b][doc_a] = connection_details

        return connectivity_map

    def calculate_triplet_coherence(
        self, triplet_ids: List[str], connectivity_map: Dict[str, Dict[str, Any]], triangle_bonus_weight: float = 0.2
    ) -> float:
        """calculate probilities of search results with uuids from (text, image, shortcut)

        Parameters
        ----------
        triplet_ids : List[str]
            uuid of (text, image, shortcut)
        connectivity_map : Dict[str, Dict[str, Any]]
            probability in knowledge graph by self.cal_prob_from_relations()
        triangle_bonus_weight : float, optional
            hyperparameter, by default 0.2

        Returns
        -------
        float
            P(A) score

        Raises
        ------
        ValueError
            _description_
        """
        if len(triplet_ids) != 3:
            raise ValueError("Must provide exactly 3 IDs for a triplet.")

        id1, id2, id3 = triplet_ids

        def get_prob(u, v):
            if u in connectivity_map and v in connectivity_map[u]:
                return connectivity_map[u][v].get("probability", 0.0)
            return 0.0

        p12 = get_prob(id1, id2)
        p23 = get_prob(id2, id3)
        p13 = get_prob(id1, id3)

        return min((p12 + p23 + p13) / 3.0, 1.0)
