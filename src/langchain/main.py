import os, sys, uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.project_config import project_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_loader.mmlongbench import dataset as Dataset
from document_parse.mineru import MinerULoader
from embedding.huggingface import QwenEmbedding
from embedding.openclip import PEEmbedding
import asyncio
from loguru import logger


# async def main():
#     dataset = Dataset()
#     text_embedding = QwenEmbedding()
#     image_embedding = PEEmbedding()

#     dataset_info = dataset.read_dataset()
#     test_data = dataset_info[-1]

#     pdf_name = test_data["qa"]["id"]
#     question = test_data["qa"]["question"]
#     ground_truth = test_data["qa"]["ground_truth"]
#     pdf_path = os.path.join(f"{config.DATA_ROOT}/projects/MRAG3.0/dataset/MMLongBench-Doc/documents", pdf_name)
#     image_path = os.path.join(f"{config.DATA_ROOT}/projects/MRAG3.0/storge/mineru", pdf_name, "auto", "images")

#     document_lodaer = MinerULoader(file_path=pdf_path, table_enable=False)
#     test_documents = document_lodaer.lazy_load()

#     # test_documents_ids = text_embedding.add_text_to_vectorstore(documents=test_documents)
#     # image_ids = image_embedding.add_image_to_vectorstore(image_path=image_path)
#     text_embedding.add_text_to_vectorstore(documents=test_documents)
#     # image_embedding.add_image_to_vectorstore(image_path=image_path)

#     # results = text_embedding.vectorstore_embd.similarity_search_with_score(question, k=1)
#     # for res, score in results:
#     #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")


# if __name__ == "__main__":
#     asyncio.run(main())

if __name__ == "__main__":
    dataset = Dataset()
    text_embedding = QwenEmbedding()
    image_embedding = PEEmbedding()

    dataset_info = dataset.read_dataset()
    test_data = dataset_info[-1]

    pdf_name = test_data["qa"]["id"]
    question = test_data["qa"]["question"]
    ground_truth = test_data["qa"]["ground_truth"]
    pdf_path = os.path.join(f"{project_config.project_root}/dataset/MMLongBench-Doc/documents", pdf_name)
    image_path = os.path.join(
        f"{project_config.project_root}/storge/mineru", pdf_name.split(".pdf")[0], "auto", "images"
    )

    document_lodaer = MinerULoader(file_path=pdf_path, table_enable=False)
    test_documents = [document for document in document_lodaer.lazy_load()]

    logger.info("document_lodaer.lazy_load() finished")

    # test_documents_ids = text_embedding.add_text_to_vectorstore(documents=test_documents)
    # image_ids = image_embedding.add_image_to_vectorstore(image_path=image_path)

    # uuids = [str(uuid.uuid4()) for _ in test_documents]
    # logger.info(test_documents)

    text_embedding.add_text_to_vectorstore(documents=test_documents)
    # image_embedding.add_image_to_vectorstore(image_path=image_path)

    results = text_embedding.vectorstore_embd.similarity_search_with_score(question, k=1)
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    # results = image_embedding.vectorstore_embd.similarity_search_with_score(
    #     question, k=1
    # )
    # for res, score in results:
    #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
