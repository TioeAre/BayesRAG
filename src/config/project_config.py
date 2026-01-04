import os
import sys
import base64
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger


class project_config:
    # load env
    project_root = Path(__file__).parent.parent.parent
    ENV_FILE_NAME = os.getenv("ENV_FILE_NAME", ".env")
    config_path = os.path.join(project_root.absolute(), ENV_FILE_NAME)
    load_dotenv(dotenv_path=config_path, override=False)

    API_KEY = os.getenv("API_KEY", "API_KEY")
    ENABLE_THINK = os.getenv("ENABLE_THINK", "false").lower() == "true"

    QWEN3_VL_BASE_URL = os.getenv("QWEN3_VL_BASE_URL", "https://example.com/v1")

    # UNI_BASE_URL = os.getenv("UNI_BASE_URL", "http://localhost:61217/v1")
    # LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3_32b")
    UNI_BASE_URL = os.getenv("UNI_BASE_URL", "https://example.com/v1")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen3-VL-32B-Instruct")

    RAGANYTHING_LLM_MODEL_MAX_ASYNC = int(os.getenv("RAGANYTHING_LLM_MODEL_MAX_ASYNC", "20"))

    GENERATE_MODEL_NAME = os.getenv("GENERATE_MODEL_NAME", "Qwen3-VL-32B-Instruct")

    EMBEDDING_BASE_URL_RAGANYTHING = os.getenv("EMBEDDING_BASE_URL_RAGANYTHING", "http://localhost:12501/v1")
    # EMBEDDING_BASE_URL_RAGANYTHING = os.getenv(
    #     "EMBEDDING_BASE_URL_RAGANYTHING", "https://example.com/v1"
    # )
    EMBEDDING_BASE_URL_VEC = os.getenv("EMBEDDING_BASE_URL_VEC", "http://localhost:12501/v1")
    RERANKER_BASE_URL_VEC = os.getenv("RERANKER_BASE_URL_VEC", "http://localhost:12502/rerank")
    # RERANKER_BASE_URL_VEC = os.getenv(
    #     "RERANKER_BASE_URL_VEC", "https://example.com/v2/rerank"
    # )

    EVAL_CONCURRENCY_LIMIT = int(os.getenv("EVAL_CONCURRENCY_LIMIT", "10"))
    MINERU_SERVER_URL = os.getenv("MINERU_SERVER_URL", "https://example.com/")
    MINERU_BACKEND = os.getenv("MINERU_BACKEND", "pipeline")
    PARSE_RESULT_DIR = os.getenv("PARSE_RESULT_DIR", "/projects/MRAG3.0/storge/mineru")

    TEXT_DATABASE = os.getenv("TEXT_DATABASE", "./database")
    IMAGE_DATABASE = os.getenv("IMAGE_DATABASE", "./database")
    SHORTCUT_DATABASE = os.getenv("SHORTCUT_DATABASE", "./database")

    # model config
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2560"))
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen3-Embedding-4B")
    EMBEDDING_CUDA_DEVICE = os.getenv("EMBEDDING_CUDA_DEVICE", "cuda:0")

    IMAGE_EMBEDDING_MODEL_NAME = os.getenv("IMAGE_EMBEDDING_MODEL_NAME", "PE-Core-bigG-14-448")
    IMAGE_EMBEDDING_CUDA_DEVICE = os.getenv("IMAGE_EMBEDDING_CUDA_DEVICE", "cuda:0")

    SHORT_CUT_MODEL_NAME = os.getenv("SHORT_CUT_MODEL_NAME", "nomic-ai/colnomic-embed-multimodal-3b")
    SHORT_CUT_CUDA_DEVICE = os.getenv("SHORT_CUT_CUDA_DEVICE", "cuda:0")

    RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "Qwen3-Reranker-4B")
    RERANK_CUDA_DEVICE = os.getenv("RERANK_CUDA_DEVICE", "cuda:0")

    RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.1"))

    ADDITIONAL_ANALYSIS = os.getenv("ADDITIONAL_ANALYSIS", "false").lower() == "true"
    ADDITIONAL_ANALYSIS_TYPE = os.getenv("ADDITIONAL_ANALYSIS_TYPE", "recall")

    # python config
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    DATA_ROOT = os.getenv("DATA_ROOT", "")
    ONLY_ADD_KG = os.getenv("ONLY_ADD_KG", "false").lower() == "true"
    SUMMARY_CONTEXT_SIZE = int(os.getenv("SUMMARY_CONTEXT_SIZE", "2048"))

    # rag config
    IF_ASK = os.getenv("IF_ASK", "false").lower() == "true"
    ADD_VECTOR = os.getenv("ADD_VECTOR", "false").lower() == "true"
    WRITE_RESULTS = os.getenv("WRITE_RESULTS", "false").lower() == "true"
    MERGE_DOCUMENT_THRESHOLD = int(os.getenv("MERGE_DOCUMENT_THRESHOLD", "0"))
    ADD_VECTOR_TEXT_BATCH_SIZE = int(os.getenv("ADD_VECTOR_TEXT_BATCH_SIZE", "100"))
    ADD_VECTOR_IMAGE_BATCH_SIZE = int(os.getenv("ADD_VECTOR_IMAGE_BATCH_SIZE", "50"))
    ADD_KG_PDF_BATCH_SIZE = int(os.getenv("ADD_KG_PDF_BATCH_SIZE", "10"))
    SHORT_CUT = os.getenv("SHORT_CUT", "false").lower() == "true"
    RERANK = os.getenv("RERANK", "false").lower() == "true"
    BAYES = os.getenv("BAYES", "false").lower() == "true"
    DShafer = os.getenv("DShafer", "false").lower() == "true"
    RAGANYTHING_RERANK = os.getenv("RAGANYTHING_RERANK", "false").lower() == "true"

    BM25_K = int(os.getenv("BM25_K", "20"))
    TEXT_K = int(os.getenv("TEXT_K", "20"))
    IMAGE_K = int(os.getenv("IMAGE_K", "5"))
    SHORT_CUT_K = int(os.getenv("SHORT_CUT_K", "5"))
    COMB_TOP_K_THRESHOLD = int(os.getenv("COMB_TOP_K_THRESHOLD", "20"))
    RERANK_TOP_K_FOR_BAYES = int(os.getenv("RERANK_TOP_K_FOR_BAYES", "40"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "8"))
    RERANK_BATCHSIZE = int(os.getenv("RERANK_BATCHSIZE", "10"))
    TUPLE_RERANK_TOP_K = int(os.getenv("TUPLE_RERANK_TOP_K", "5"))
    TUPLE_RERANK_TOP_K_TEXT = int(os.getenv("TUPLE_RERANK_TOP_K_TEXT", "5"))
    TUPLE_RERANK_TOP_K_IMAGE = int(os.getenv("TUPLE_RERANK_TOP_K_IMAGE", "5"))
    TUPLE_RERANK_TOP_K_SHORTCUT = int(os.getenv("TUPLE_RERANK_TOP_K_SHORTCUT", "5"))

    D_S_ALPHA = float(os.getenv("D_S_ALPHA", "0.7"))
    D_S_BETA = float(os.getenv("D_S_BETA", "0.9"))

    WEIGHT_TEXT_VECTOR = float(os.getenv("WEIGHT_TEXT_VECTOR", "0.45"))
    WEIGHT_TEXT_BM25 = float(os.getenv("WEIGHT_TEXT_BM25", "0.05"))
    WEIGHT_IMAGE = float(os.getenv("WEIGHT_IMAGE", "0.3"))
    WEIGHT_SHOERCUT = float(os.getenv("WEIGHT_SHOERCUT", "0.2"))

    DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", 2))
    EPSILON = float(os.getenv("EPSILON", "0.1"))
    SCALING_FACTOR = float(os.getenv("SCALING_FACTOR", "0.1"))
    BONUS_WEIGHT = float(os.getenv("BONUS_WEIGHT", "0.2"))

    MODEL_PA_BY_KG_WEIGHT = os.getenv("MODEL_PA_BY_KG_WEIGHT", "false").lower() == "true"
    MODEL_PA_BY_KG = os.getenv("MODEL_PA_BY_KG", "false").lower() == "true"
    KG_TOP_EVIDENCE = int(os.getenv("KG_TOP_EVIDENCE", "3"))

    # test config
    TEST_BENCH_FULL = os.getenv("TEST_BENCH_FULL", "false").lower() == "true"
    TEST_RAG = os.getenv("TEST_RAG", "false").lower() == "true"

    # traceloop / langfuse config
    TRACELOOP_BASE_URL = os.getenv("TRACELOOP_BASE_URL", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
    os.environ["TRACELOOP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    RESULT_DIR_NAME = os.getenv("RESULT_DIR_NAME", "timestamp")

    # RAGAnything config
    IF_GPT4o = os.getenv("IF_GPT4o", "false").lower() == "true"
    GPT4o_MODEL_NAME = os.getenv("GPT4o_MODEL_NAME", "azure-gpt-4o")

    MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:admin@localhost:27017/?authSource=admin")
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")

    # logger
    PRINT_TERMINAL = os.getenv("PRINT_TERMINAL", "true").lower() == "true"
    PRINT_TERMINAL_LEVEL = os.getenv("PRINT_TERMINAL_LEVEL", "DEBUG")
    PRINT_FILE = os.getenv("PRINT_FILE", "true").lower() == "true"
    logger.remove()  # remove handel
    # print to terminal
    if PRINT_TERMINAL:
        logger.add(sys.stdout, level=PRINT_TERMINAL_LEVEL)  # stderr
    # print to file
    if PRINT_FILE:
        if DEBUG:
            logger.add(
                os.path.join(project_root, "logs", "debug.log"), level="DEBUG", rotation="1 MB", retention="7 days"
            )
        logger.add(
            os.path.join(project_root, "logs", "info_warning.log"),
            level="INFO",
            filter=lambda record: record["level"].no < logger.level("ERROR").no,
            rotation="1 MB",
        )
        logger.add(os.path.join(project_root, "logs", "error.log"), level="ERROR", rotation="1 MB")
