import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config
import uuid

import chromadb
import numpy as np
from langchain_chroma.vectorstores import Chroma
from PIL import Image as _PILImage
from pathlib import Path
from typing import List
from langchain_core.embeddings import Embeddings
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

import tempfile
import fitz  # PyMuPDF
from typing import List, Any
from peft import PeftModel

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


class ColpaliEmbeddings(Embeddings):

    def __init__(self, model_name: str):
        self.model = ColQwen2_5.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",  # or "mps" if on Apple Silicon
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        # self.model = PeftModel.from_pretrained(base_model, adapter_path).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        batch_queries = self.processor.process_queries(texts).to(self.model.device)  # type: ignore
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
            # print(f"query_embeddings: {query_embeddings.shape}")  # NOTE
            return query_embeddings.cpu().squeeze(0).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    def embed_image(self, uris: List[str]) -> List[List[float]]:
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

        batch_images = self.processor.process_images(pil_images).to(self.model.device)  # type: ignore
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
            print(f"image_embeddings: {image_embeddings.shape}")  # NOTE
            return image_embeddings.cpu().tolist()


class ColpaliEmbedding:

    def __init__(self, database=f"{project_config.project_root}/database/shortcut_db"):
        self.embedding_model = ColpaliEmbeddings(model_name="vidore/colqwen2.5-v0.2")
        self.vectorstore_embd = Chroma(
            collection_name="shortcut", embedding_function=self.embedding_model, persist_directory=database
        )
        self.retriever_embd = self.vectorstore_embd.as_retriever()

    def add_pdf_to_vectorstore(self, pdf_path, start_page=None, end_page=None):
        if not os.path.exists(pdf_path) or not os.path.isfile(pdf_path):
            print(f"exists return []")
            return []
        if start_page != None and end_page != None:
            if start_page < 0 or start_page > end_page:
                print(f"start_page return []")
                return []

        image_uris = []
        with tempfile.TemporaryDirectory() as temp_dir:
            doc = fitz.open(pdf_path)
            if start_page == None or start_page >= len(doc):
                start_page = 0
            if end_page == None or end_page >= len(doc):
                end_page = len(doc) - 1

            for page_num in range(start_page, end_page + 1):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150)  # Use higher DPI for better quality if needed
                image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                pix.save(image_path)
                image_uris.append(image_path)
            doc.close()
            if not image_uris:
                print(f"image_uris return []")
                return []
            uuids = [str(uuid.uuid4()) for _ in image_uris]
            self.vectorstore_embd.add_images(uris=image_uris, ids=uuids)
            return uuids


if __name__ == "__main__":
    test_file = f"{project_config.project_root}/dataset/MMLongBench-Doc/documents/3M_2018_10K.pdf"
    emb = ColpaliEmbedding()
    emb.add_pdf_to_vectorstore(test_file)
    # results = emb.vectorstore_embd.similarity_search_with_score("table", k=1)
    results = emb.vectorstore_embd.get()
    print(results)
