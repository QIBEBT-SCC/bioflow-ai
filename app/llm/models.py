from langchain_qwen3 import Qwen3Embeddings

from app.utils.cache import cache_model


@cache_model()
def qwen3_embedding() -> Qwen3Embeddings:
    embedding = Qwen3Embeddings(use_modelscope=True)
    return embedding
