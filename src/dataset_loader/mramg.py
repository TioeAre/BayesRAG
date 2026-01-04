import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config
import copy
import json


class dataset():
    def __init__(self) -> None:
        self.root_dir = f"{project_config.DATA_ROOT}/projects/MRAG3.0/dataset/MRAMG-Bench"

        self.image_root_dir = os.path.join(self.root_dir, "IMAGE")
        self.image_info_dir = os.path.join(self.image_root_dir, "images_info")
        self.image_dir = os.path.join(self.image_root_dir, "images")

        self.data_collections = ["wit", "wiki", "web", "arxiv", "recipe", "manual"]

    def get_data_file_path(self, data_collection: str):
        data_source = os.path.join(self.root_dir, f"doc_{data_collection}.jsonl")
        qa_pair = os.path.join(self.root_dir, f"{data_collection}_mqa.jsonl")
        image_info = os.path.join(self.image_info_dir, f"{data_collection}_imgs_collection.json")
        image_path = os.path.join(self.image_dir, data_collection.upper())

        return data_source, qa_pair, image_info, image_path

    def read_dataset(self, data_collection="wiki") -> list:
        dataset = []
        data_struct = {
            "qa": {
                "id": "",
                "question": "",
                "ground_truth": "",
                "provenance": [],
                "images_list": []
            },
            "provenance": {
                # "#id": {
                #     "text": "",
                #     "images_list": []
                # }
            },
            "images": {
                # "#id": {
                #     "image_url": "https://upload.wikimedia.org/wikipedia/commons/8/85/Plymouth-indiana-courthouse.jpg",
                #     "image_path": ".../w_s15.jpg",
                #     "image_caption": "Marshall County courthouse in Plymouth, Indiana, , Marshall County courthouse in Plymouth, Indiana"
                # }
            }
        }
        data_source, qa_pair, image_info, image_path = self.get_data_file_path(data_collection=data_collection)

        data_docs = {}  # {"#id": {"text": "", images_list: []}}
        with open(data_source, "r", encoding="utf-8") as f:
            for line in f:
                data_doc = json.loads(line)
                key = data_doc[0]
                if key not in data_docs.keys():
                    data_docs[key] = {"text": data_doc[1], "images_list": data_doc[2]}
        image_infos = {}
        with open(image_info, "r", encoding="utf-8") as f:
            image_infos = json.load(f)
        with open(qa_pair, "r", encoding="utf-8") as f:
            for line in f:
                qa = json.loads(line)
                json_data = copy.deepcopy(data_struct)
                json_data["qa"] = qa
                for provenance in json_data["qa"]["provenance"]:
                    json_data["provenance"][provenance] = {
                        "text": data_docs[provenance]["text"],
                        "images_list": data_docs[provenance]["images_list"]}

                for image in json_data["qa"]["images_list"]:
                    json_data["images"][image] = {
                        "image_url": image_infos[image]["image_url"],
                        "image_caption": image_infos[image]["image_caption"],
                        "image_path": os.path.join(image_path, image_infos[image]["image_path"])
                    }
                dataset.append(json_data)

        return dataset


if __name__ == "__main__":
    data = dataset()
    print(data.read_dataset()[-1])
