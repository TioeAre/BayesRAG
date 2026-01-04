from langchain_chroma.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def bm25_search_from_chroma(chroma, query: str, k: int = 5) -> list[Document]:
    # Get all raw documents from the ChromaDB
    raw_docs = chroma.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=doc, metadata=meta) for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(documents=documents, k=k)
    # similarity_search_retriever = chroma.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[similarity_search_retriever, bm25_retriever], weights=[0.5, 0.5]
    # ) # hybrid_search
    return bm25_retriever.invoke(query)


def chroma2bm25(chroma, k: int = 5) -> BM25Retriever:
    raw_docs = chroma.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=doc, metadata=meta) for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
    ]
    if len(documents) != 0:
        bm25_retriever = BM25Retriever.from_documents(documents=documents, k=k)
    else:
        bm25_retriever = BM25Retriever.from_texts(texts=["test", "test"], metadatas=[{"test":"test"}, {"test":"test"}], k=k)
    return bm25_retriever
