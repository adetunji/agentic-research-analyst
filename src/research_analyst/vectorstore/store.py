from langchain_chroma import Chroma
from research_analyst.config import settings
from research_analyst.vectorstore.embeddings import get_embedding_model


def get_vectorstore() -> Chroma:
    """
    Returns a persistent ChromaDB vectorstore.
    Data is saved to disk at chroma_persist_dir and survives between runs.
    """
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever(k: int | None = None):
    """
    Returns a LangChain retriever backed by ChromaDB.
    k controls how many documents are returned per query.
    """
    k = k or settings.rag_top_k
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
