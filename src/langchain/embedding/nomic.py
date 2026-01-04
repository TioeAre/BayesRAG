import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config

# isort:skip
import uuid, math
from loguru import logger
import chromadb
import numpy as np
from langchain_chroma.vectorstores import Chroma
from PIL import Image as _PILImage
from pathlib import Path
from typing import List, Union, Sequence
from langchain_core.embeddings import Embeddings
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

import tempfile
import fitz  # PyMuPDF
from typing import List, Any
from peft import PeftModel

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.langchain.utils.image import save_base64_image
from src.langchain.utils.uuid import generate_stable_uuid_for_text


class NomicEmbeddings(Embeddings):

    def __init__(self, model_name: str):
        self.model = ColQwen2_5.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=f"{project_config.SHORT_CUT_CUDA_DEVICE}",
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name, use_fast=True)
        self.batch_size = 10

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        batch_queries = self.processor.process_queries(texts).to(self.model.device)  # type: ignore
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
            pooled_embeddings = query_embeddings.mean(dim=1)
            normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
            # logger.debug(f"query_embeddings: {query_embeddings.shape}")  # NOTE
            return normalized_embeddings.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    def embed_image(self, uris: Sequence[Union[str, _PILImage.Image]]) -> List[List[float]]:
        try:
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError("Please install the PIL library: pip install pillow")
        pil_images = []
        for image_input in uris:
            if isinstance(image_input, str):
                if os.path.exists(image_input):
                    img = _PILImage.open(image_input).convert("RGB")
                    pil_images.append(img)
                else:
                    raise FileNotFoundError(f"Invalid image file path: {image_input}")
            elif isinstance(image_input, _PILImage.Image):
                pil_images.append(image_input.convert("RGB"))
            else:
                raise TypeError(f"Input must be a file path (str) or a PIL.Image.Image, but got {type(image_input)}")
        all_embeddings = []
        for i in range(0, len(pil_images), self.batch_size):
            image_batch = pil_images[i : i + self.batch_size]
            processed_batch = self.processor.process_images(image_batch).to(self.model.device)  # type: ignore
            with torch.no_grad():
                batch_embeddings = self.model(**processed_batch)
                pooled_embeddings = batch_embeddings.mean(dim=1)
                normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
                all_embeddings.extend(normalized_embeddings.cpu().tolist())
        return all_embeddings


class NomicEmbedding:

    def __init__(
        self, database=f"{project_config.DATA_ROOT}/projects/MRAG3.0/database/shortcut_db", collection_name="shortcut"
    ):
        self.embedding_model = NomicEmbeddings(
            model_name=f"{project_config.SHORT_CUT_MODEL_NAME}",
        )
        self.vectorstore_embd = Chroma(
            collection_name=collection_name, embedding_function=self.embedding_model, persist_directory=database
        )
        self.retriever_embd = self.vectorstore_embd.as_retriever()

    def add_pdf_to_vectorstore(
        self,
        pdf_path,
        start_page=None,
        end_page=None,
        tmp_dir=f"{project_config.DATA_ROOT}/projects/MRAG3.0/tmp",
    ):
        if not os.path.exists(pdf_path) or not os.path.isfile(pdf_path):
            logger.warning(f"exists return []")
            return []
        if start_page != None and end_page != None:
            if start_page < 0 or start_page > end_page:
                logger.warning(f"start_page return []")
                return []

        image_uris = []
        with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
            doc = fitz.open(pdf_path)
            if start_page == None or start_page >= len(doc):
                start_page = 0
            if end_page == None or end_page >= len(doc):
                end_page = len(doc) - 1
            pdf_name = Path(pdf_path).name
            potential_metadatas = [
                {
                    "source": pdf_name,
                    "page_idx": page_num,
                    "uuid": generate_stable_uuid_for_text(f"{pdf_name}_{page_num}"),
                }
                for page_num in range(start_page, end_page + 1)
            ]
            if not potential_metadatas:
                doc.close()
                return []
            potential_ids = [str(meta["uuid"]) for meta in potential_metadatas]
            existing_docs = self.vectorstore_embd.get_by_ids(potential_ids)
            existing_ids = {str(doc.metadata["uuid"]) for doc in existing_docs}
            new_metadatas = [meta for meta in potential_metadatas if str(meta["uuid"]) not in existing_ids]
            if not new_metadatas:
                doc.close()
                return []
            image_uris = []
            for meta in new_metadatas:
                page_num = meta["page_idx"]
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150)  # type: ignore
                image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                pix.save(image_path)
                image_uris.append(image_path)
            doc.close()
            if not image_uris:
                return []
            new_ids = [str(meta["uuid"]) for meta in new_metadatas]
            self.vectorstore_embd.add_images(uris=image_uris, ids=new_ids, metadatas=new_metadatas)
            logger.debug("Successfully added new pdf screenshots.")


if __name__ == "__main__":
    test_file = f"{project_config.DATA_ROOT}/projects/MRAG3.0/dataset/MMLongBench-Doc/documents/3M_2018_10K.pdf"
    emb = NomicEmbedding()
    # emb.add_pdf_to_vectorstore(test_file)

    results = emb.vectorstore_embd.similarity_search_with_score(
        "Pursuant to Part IV, Item 16, a summary of Form 10-K content follows, including hyperlinked cross-references (in the EDGAR filing). This allows users to easily locate the corresponding items in Form 10-K, where the disclosure is fully presented. The summary does not include certain Part III information that will be incorporated by reference from the proxy statement, which will be filed after this Form 10-K filing.",
        k=1,
    )
    # logger.info(results)
    base64_str = results[0][0].page_content
    save_base64_image(base64_str, f"{project_config.DATA_ROOT}/projects/MRAG3.0/output1.jpg")

    # results = emb.vectorstore_embd.get()
    # logger.info("get")
    # logger.info(results)
