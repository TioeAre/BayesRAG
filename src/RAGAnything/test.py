import asyncio, uuid, json
from dataclasses import asdict
from functools import partial
from kg import KG
from lightrag import QueryParam
from lightrag.utils import logger as lightrag_logger
from lightrag.operate import (
    get_keywords_from_query,
    _perform_kg_search,
    _apply_token_truncation,
    _merge_all_chunks,
    _build_context_str,
)

content_list = [
    # Introduction text
    {
        "type": "text",
        "text": "Welcome to the RAGAnything System Documentation. This guide covers the advanced multimodal document processing capabilities and features of our comprehensive RAG system.",
        "page_idx": 0,  # Page number where this content appears
    },
    # System architecture image
    {
        "type": "image",
        "img_path": "/projects/rag_method/RAG-Anything/assets/rag_anything_framework.png",  # IMPORTANT: Use absolute path to image file
        "image_caption": ["Figure 1: RAGAnything System Architecture"],
        "image_footnote": [
            "The architecture shows the complete pipeline from document parsing to multimodal query processing"
        ],
        "page_idx": 1,  # Page number where this image appears
    },
    # Performance comparison table
    {
        "type": "table",
        "table_body": """| System | Accuracy | Processing Speed | Memory Usage |
                            |--------|----------|------------------|--------------|
                            | RAGAnything | 95.2% | 120ms | 2.1GB |
                            | Traditional RAG | 87.3% | 180ms | 3.2GB |
                            | Baseline System | 82.1% | 220ms | 4.1GB |
                            | Simple Retrieval | 76.5% | 95ms | 1.8GB |""",
        "table_caption": ["Table 1: Performance Comparison of Different RAG Systems"],
        "table_footnote": ["All tests conducted on the same hardware with identical test datasets"],
        "page_idx": 2,  # Page number where this table appears
    },
    # Mathematical formula
    {
        "type": "equation",
        "latex": "Relevance(d, q) = \\sum_{i=1}^{n} w_i \\cdot sim(t_i^d, t_i^q) \\cdot \\alpha_i",
        "text": "Document relevance scoring formula where w_i are term weights, sim() is similarity function, and α_i are modality importance factors",
        "page_idx": 3,  # Page number where this equation appears
    },
    # Feature description
    {
        "type": "text",
        "text": "The system supports multiple content modalities including text, images, tables, and mathematical equations. Each modality is processed using specialized processors optimized for that content type.",
        "page_idx": 4,  # Page number where this content appears
    },
    # Technical specifications table
    {
        "type": "table",
        "table_body": """| Feature | Specification |
                            |---------|---------------|
                            | Supported Formats | PDF, DOCX, PPTX, XLSX, Images |
                            | Max Document Size | 100MB |
                            | Concurrent Processing | Up to 8 documents |
                            | Query Response Time | <200ms average |
                            | Knowledge Graph Nodes | Up to 1M entities |""",
        "table_caption": ["Table 2: Technical Specifications"],
        "table_footnote": ["Specifications may vary based on hardware configuration"],
        "page_idx": 5,  # Page number where this table appears
    },
    # Conclusion
    {
        "type": "text",
        "text": "RAGAnything represents a significant advancement in multimodal document processing, providing comprehensive solutions for complex knowledge extraction and retrieval tasks.",
        "page_idx": 6,  # Page number where this content appears
    },
]


async def query(rag, query: str, mode):

    await rag._ensure_lightrag_initialized()
    rag.logger.info(f"Executing VLM enhanced query: {query[:100]}...")
    if hasattr(rag, "_current_images_base64"):
        delattr(rag, "_current_images_base64")
    # 1. Get original retrieval prompt (without generating final answer)
    query_param = QueryParam(mode=mode, only_need_prompt=True)
    lightrag_logger.debug(f"[aquery_llm] Query param: {query_param}")
    lightrag_config = asdict(rag.lightrag)

    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = lightrag_config["llm_model_func"]
        use_model_func = partial(use_model_func, _priority=5)

    # NOTE: 使用 llm 提取 keywords
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, lightrag_config, rag.lightrag.llm_response_cache
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
        else:
            return "1"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Stage 1: Pure search
    # NOTE: knowledge search
    search_result = await _perform_kg_search(
        query,
        ll_keywords_str,
        hl_keywords_str,
        rag.lightrag.chunk_entity_relation_graph,
        rag.lightrag.entities_vdb,
        rag.lightrag.relationships_vdb,
        rag.lightrag.text_chunks,
        query_param,
        rag.lightrag.chunks_vdb,
    )

    return search_result

    # if not search_result["final_entities"] and not search_result["final_relations"]:
    #     if query_param.mode != "mix":
    #         return "2"
    #     else:
    #         if not search_result["chunk_tracking"]:
    #             return "3"

    # # Stage 2: Apply token truncation for LLM efficiency
    # truncation_result = await _apply_token_truncation(
    #     search_result,
    #     query_param,
    #     rag.lightrag.text_chunks.global_config,
    # )

    # # Stage 3: Merge chunks using filtered entities/relations
    # # NOTE: merge knowledge graph
    # merged_chunks = await _merge_all_chunks(
    #     filtered_entities=truncation_result["filtered_entities"],
    #     filtered_relations=truncation_result["filtered_relations"],
    #     vector_chunks=search_result["vector_chunks"],
    #     query=query,
    #     knowledge_graph_inst=rag.lightrag.chunk_entity_relation_graph,
    #     text_chunks_db=rag.lightrag.text_chunks,
    #     query_param=query_param,
    #     chunks_vdb=rag.lightrag.chunks_vdb,
    #     chunk_tracking=search_result["chunk_tracking"],
    #     query_embedding=search_result["query_embedding"],
    # )

    # if not merged_chunks and not truncation_result["entities_context"] and not truncation_result["relations_context"]:
    #     return "4"
    # else:
    #     return merged_chunks  # return chunks [{"content": ..., "file_path": ..., "chunk_id": ...}, ...]


async def main(kg: KG):
    await kg.rag._ensure_lightrag_initialized()
    # # NOTE: 每个 chunk 一个一个插入, doc_id 赋值 uuid
    # for content in content_list:
    #     await kg.rag.insert_content_list(
    #         content_list=[content],
    #         file_path="raganything_documentation.pdf",  # Reference file name for citation
    #         split_by_character=None,  # Optional text splitting
    #         split_by_character_only=False,  # Optional text splitting mode
    #         doc_id=str(uuid.uuid4()),  # Custom document ID
    #         display_stats=True,
    #     )
    #     print("-----------------------insert----------------------------")

    # doc_status = await kg.rag.lightrag.full_docs.get_by_id("074eb424-1e26-56c4-9d9f-18599666334c")
    # print(doc_status)

    text_queries = [
        "What is RAGAnything and what are its main features?",
        "How does RAGAnything compare to traditional RAG systems?",
        "What are the technical specifications of the system?",
    ]
    mode = "hybrid"
    for query_text in text_queries:
        # result = await query(kg.rag, query_text, mode=mode)  # raganything.aquery
        print("======================== kg start ========================")
        # print(result)
        chunk_result = await kg.rag.lightrag.full_entities.get_by_id("09be6f81-ac3e-56bb-915e-ed235c401de5")  # type: ignore
        print(chunk_result)
        print("======================== kg result ========================")
        break


# docker run --name ragmongo --hostname localhost --userns=host --user root -e MONGODB_INITDB_ROOT_USERNAME=admin -e MONGODB_INITDB_ROOT_PASSWORD=admin -v ./database/raganything_db/mongo:/data/db -v ./database/mongo_keyfile:/data/configdb/keyfile -p 27017:27017 -d docker.1ms.run/mongodb/mongodb-atlas-local
if __name__ == "__main__":
    kg = KG()

    asyncio.run(main(kg))
