from langchain_huggingface import HuggingFaceEmbeddings
from research_analyst.config import settings


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a local sentence-transformers embedding model.
    all-MiniLM-L6-v2: 384-dim, ~80MB, CPU-friendly, strong on semantic similarity.
    No API calls — fully local and free.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
