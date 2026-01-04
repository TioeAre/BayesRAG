import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.config.project_config import project_config

import fitz  # PyMuPDF
from pathlib import Path
import json
from huggingface_hub import hf_hub_download
import tarfile


def build_rag_database(domain_name, output_dir="rag_database"):
    """Build RAG database from domain PDFs"""
    # Process PDFs for RAG
    rag_data = []
    pdf_dir = Path(temp_dir)

    for pdf_file in pdf_dir.glob("**/*.pdf"):
        doc = fitz.open(pdf_file)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            images = page.get_images()
            entry = {
                "source": str(pdf_file),
                "page": page_num,
                "text": text,
                "images": len(images),
                "domain": domain_name,
            }
            rag_data.append(entry)

        doc.close()

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{domain_name}_rag.json", "w") as f:
        json.dump(rag_data, f, indent=2)

    print(f"Built RAG database for {domain_name}: {len(rag_data)} entries")
    return rag_data


# Example: Build RAG database for healthcare
healthcare_rag = build_rag_database("healthcare")
