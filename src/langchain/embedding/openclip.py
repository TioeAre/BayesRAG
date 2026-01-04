import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config
from PIL import Image as _PILImage
from langchain_chroma.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import numpy as np
import chromadb
import uuid
import tempfile
import base64, math
from langchain_core.documents import Document

# from langchain_experimental.open_clip import OpenCLIPEmbeddings
from pathlib import Path
from loguru import logger
from src.langchain.utils.image import save_base64_image
from typing import Any, Dict, List
from langchain_core.embeddings import Embeddings
import torch


class OpenCLIPEmbedding(Embeddings):
    def __init__(self, model_name, checkpoint, device, **model_kwargs):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=checkpoint,
            device=self.device,
            **model_kwargs,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_features = []
        with torch.no_grad():
            for text in texts:
                tokenized_text = self.tokenizer(text).to(self.device)  # type: ignore
                embeddings_tensor = self.model.encode_text(tokenized_text)  # type: ignore
                norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
                normalized_embeddings_tensor = embeddings_tensor.div(norm)
                embeddings_list = normalized_embeddings_tensor.squeeze(0).cpu().tolist()
                text_features.append(embeddings_list)
            return text_features

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        try:
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError("Please install the PIL library: pip install pillow")
        pil_images = [_PILImage.open(uri) for uri in uris]
        image_features = []
        for pil_image in pil_images:
            preprocessed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)  # type: ignore
            embeddings_tensor = self.model.encode_image(preprocessed_image)  # type: ignore
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)
            embeddings_list = normalized_embeddings_tensor.squeeze(0).cpu().tolist()
            image_features.append(embeddings_list)
        return image_features


class PEEmbedding:

    def __init__(
        self,
        database=f"{project_config.DATA_ROOT}/projects/MRAG3.0/database/image_db",
        collection_name="image",
        embedding_model=None,
    ):
        model_kwargs = {}
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = OpenCLIPEmbedding(
                model_name="PE-Core-bigG-14-448",
                checkpoint=f"{project_config.DATA_ROOT}/models/{project_config.IMAGE_EMBEDDING_MODEL_NAME}/open_clip_pytorch_model.bin",
                device=f"{project_config.IMAGE_EMBEDDING_CUDA_DEVICE}",
                **model_kwargs,
            )

        self.vectorstore_embd = Chroma(
            collection_name=collection_name, embedding_function=self.embedding_model, persist_directory=database
        )
        self.retriever_embd = self.vectorstore_embd.as_retriever()

    def add_image_to_vectorstore(
        self, image_documents: List[Document], tmp_dir: str = f"{project_config.DATA_ROOT}/projects/MRAG3.0/tmp"
    ):
        if not image_documents:
            return
        incoming_ids = [str(doc.metadata["uuid"]) for doc in image_documents]
        existing_docs = self.vectorstore_embd.get_by_ids(incoming_ids)
        existing_ids = {str(doc.metadata["uuid"]) for doc in existing_docs}
        new_image_documents = [doc for doc in image_documents if str(doc.metadata["uuid"]) not in existing_ids]
        if not new_image_documents:
            return

        image_uris = []
        metadatas = []

        with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
            for doc in new_image_documents:
                try:
                    base64_string = doc.page_content
                    if base64_string != "":
                        image_data = base64.b64decode(base64_string)
                        extension = doc.metadata.get("extension", ".png")
                        file_name = f"{doc.metadata.get('uuid', uuid.uuid4())}{extension}"
                        temp_image_path = os.path.join(temp_dir, file_name)
                        with open(temp_image_path, "wb") as f:
                            f.write(image_data)
                        if os.path.exists(temp_image_path):
                            image_uris.append(temp_image_path)
                            metadatas.append(doc.metadata)
                except Exception as e:
                    logger.error(f"add_image_to_vectorstore error: {e}")
                    continue

            if not image_uris:
                return
            ids = [str(meta.get("uuid", uuid.uuid4())) for meta in metadatas]
            self.vectorstore_embd.add_images(uris=image_uris, ids=ids, metadatas=metadatas)
            logger.debug("Successfully added new images.")

    def add_text_to_vectorstore(self, documents: List[Document]):
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


if __name__ == "__main__":
    image_embedding = PEEmbedding()
    # image_embedding.add_image_to_vectorstore(
    # f"{project_config.DATA_ROOT}/projects/MRAG3.0/storge/mineru/3M_2018_10K/auto/images"
    # )
