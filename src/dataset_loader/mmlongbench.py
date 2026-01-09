import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config
import copy
import json, random
import pandas as pd
from loguru import logger


class dataset:
    def __init__(self) -> None:
        self.root_dir = f"{project_config.project_root}/dataset/MMLongBench-Doc"

        self.data_path = os.path.join(self.root_dir, "data", "train-00000-of-00001.parquet")
        self.doc_dir = os.path.join(self.root_dir, "documents")

    def read_dataset(self) -> list:
        dataset = []
        data_struct = {
            "qa": {
                "id": "",  # pdf name: 3M_2018_10K.pdf
                "question": "",
                "ground_truth": "",
                "provenance": [],  # evidence_pages: [5]
                "evidence_sources": [],  # ['Chart']
                "answer_format": "",  # 'Float'
                "doc_type": "",  # 'Financial report'
            },
        }
        df = pd.read_parquet(self.data_path, engine="pyarrow")
        for index, row in df.iterrows():
            data = copy.deepcopy(data_struct)
            data["qa"]["id"] = row["doc_id"]
            data["qa"]["question"] = row["question"]
            data["qa"]["ground_truth"] = row["answer"]
            data["qa"]["provenance"] = json.loads(row["evidence_pages"])
            data["qa"]["evidence_sources"] = row["evidence_sources"]
            data["qa"]["answer_format"] = row["answer_format"]
            data["qa"]["doc_type"] = row["doc_type"]
            dataset.append(data)
        return dataset


if __name__ == "__main__":
    data = dataset()
    datas = data.read_dataset()
    pdf_names = []
    seen = set()
    for item in datas:
        if "qa" in item and "id" in item["qa"]:
            pdf_id = item["qa"]["id"]
            if pdf_id not in seen:
                pdf_names.append(pdf_id)
                seen.add(pdf_id)
    # for pdf_name in pdf_names:
    #     print(pdf_name)
    logger.info(datas[-1])
    logger.info(len(datas))   # 1091 qas, 136 docs
    # new_list = random.sample(datas, 30)
    with open(f"{project_config.project_root}/eval/mmLongBench/full_dataset.json", "w") as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)
