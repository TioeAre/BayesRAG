import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config

# isort:skip
from langchain_chroma.vectorstores import Chroma
from pathlib import Path
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import uuid
from loguru import logger


class QwenEmbedding:

    def __init__(
        self, database=f"{project_config.DATA_ROOT}/projects/MRAG3.0/database/th_text_db", collection_name="text"
    ):  # text_db
        model_kwargs = {"device": f"{project_config.EMBEDDING_CUDA_DEVICE}"}
        encode_kwargs = {"normalize_embeddings": False}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=f"{project_config.DATA_ROOT}/models/{project_config.EMBEDDING_MODEL_NAME}",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        self.vectorstore_embd = Chroma(
            collection_name=collection_name, embedding_function=self.embedding_model, persist_directory=database
        )
        self.retriever_embd = self.vectorstore_embd.as_retriever()

    def add_text_to_vectorstore(self, documents):
        if not documents:
            logger.debug("No documents provided to add.")
            return
        incoming_ids = [str(doc.metadata["uuid"]) for doc in documents]
        existing_docs = self.vectorstore_embd.get_by_ids(incoming_ids)
        existing_ids = {str(doc.metadata["uuid"]) for doc in existing_docs}
        new_documents = [doc for doc in documents if str(doc.metadata["uuid"]) not in existing_ids]
        if new_documents:
            new_ids_to_add = [str(doc.metadata["uuid"]) for doc in new_documents]
            new_documents_to_add = []
            for ori_doc in new_documents:
                new_doc = ori_doc
                new_doc.page_content = f"The following is the content in page {ori_doc.metadata['page_idx']+1} of {ori_doc.metadata['source']}: {ori_doc.page_content}"
                new_documents_to_add.append(new_doc)
            self.vectorstore_embd.add_documents(documents=new_documents_to_add, ids=new_ids_to_add)
            logger.debug("Successfully added new documents.")

        # ids = self.vectorstore_embd.add_documents(documents=documents)
        # self.vectorstore_embd.add_documents(
        #     documents=documents, ids=[str(document.metadata["uuid"]) for document in documents]
        # )
        # return ids


class VllmEmbedding:

    def __init__(
        self, database=f"{project_config.DATA_ROOT}/projects/MRAG3.0/database/th_text_db", collection_name="text"
    ):  # text_db
        self.embedding_model = OpenAIEmbeddings(
            base_url=project_config.EMBEDDING_BASE_URL_VEC, model=project_config.EMBEDDING_MODEL_NAME.split("vllm-")[1], api_key="None"  # type: ignore
        )
        self.vectorstore_embd = Chroma(
            collection_name=collection_name, embedding_function=self.embedding_model, persist_directory=database
        )
        self.retriever_embd = self.vectorstore_embd.as_retriever()

    def add_text_to_vectorstore(self, documents):
        if not documents:
            logger.debug("No documents provided to add.")
            return
        incoming_ids = [str(doc.metadata["uuid"]) for doc in documents]
        existing_docs = self.vectorstore_embd.get_by_ids(incoming_ids)
        existing_ids = {str(doc.metadata["uuid"]) for doc in existing_docs}
        new_documents = [doc for doc in documents if str(doc.metadata["uuid"]) not in existing_ids]
        if new_documents:
            new_ids_to_add = [str(doc.metadata["uuid"]) for doc in new_documents]
            new_documents_to_add = []
            for ori_doc in new_documents:
                new_doc = ori_doc
                new_doc.page_content = f"The following is the content in page {ori_doc.metadata['page_idx']+1} of {ori_doc.metadata['source']}: {ori_doc.page_content}"
                new_documents_to_add.append(new_doc)
            self.vectorstore_embd.add_documents(documents=new_documents_to_add, ids=new_ids_to_add)
            logger.debug("Successfully added new documents.")
