import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config
import copy, glob
import json, random
import pandas as pd
from loguru import logger
from pathlib import Path
from pypdf import PdfReader


class dataset:
    def __init__(self) -> None:
        self.root_dir = f"{project_config.project_root}/dataset/DocBench"

    def read_dataset(self) -> list:
        temp_data = []
        dataset_info = []
        folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]

        for folder in folders:
            folder_path = os.path.join(self.root_dir, folder)

            pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
            if not pdf_files:
                continue
            pdf_path = pdf_files[0]

            # get number of pages
            try:
                reader = PdfReader(pdf_path)
                num_pages = len(reader.pages)
            except Exception as e:
                logger.error(f"Error reading PDF {pdf_path}: {e}")
                num_pages = float("inf")

            jsonl_path = os.path.join(folder_path, f"{folder}_qa.jsonl")
            if not os.path.exists(jsonl_path):
                continue

            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    original_data = json.loads(line)

                    sample = {
                        "qa": {
                            "id": Path(pdf_path).name,
                            "num_id": folder,
                            "question": original_data.get("question", ""),
                            "ground_truth": original_data.get("answer", ""),
                            "evidence": original_data.get("evidence", ""),
                            "answer_format": original_data.get("type", ""),
                            "provenance": [],
                        }
                    }
                    temp_data.append((num_pages, sample))
                    # dataset_info.append(sample)
        temp_data.sort(key=lambda x: x[0])
        dataset_info = [item[1] for item in temp_data]

        logger.info(f"Loaded {len(dataset_info)} samples from DocBench dataset.")
        return dataset_info


if __name__ == "__main__":
    data = dataset()
    datas = data.read_dataset()
    print(datas[0])
    pdf_names = []
    seen = set()
    for item in datas:
        if "qa" in item and "id" in item["qa"]:
            pdf_id = item["qa"]["id"]
            if pdf_id not in seen:
                pdf_names.append(pdf_id)
                seen.add(pdf_id)
    for pdf_name in pdf_names:
        print(pdf_name)
    # logger.info(datas[-1])
    # logger.info(len(datas))  # 1091 qas, 136 docs
    # new_list = random.sample(datas, 30)
    # with open(f"{project_config.project_root}/eval/DocBench/test_dataset.json", "w") as f:
    #     json.dump(new_list, f, ensure_ascii=False, indent=2)
    # with open(f"{project_config.project_root}/eval/DocBench/full_dataset.json", "w") as f:
    #     json.dump(datas, f, ensure_ascii=False, indent=2)
