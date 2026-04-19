import asyncio

from fastembed import TextEmbedding
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize.custom import CustomVectorizer


def build_semantic_cache(redis_url: str, distance_threshold: float = 0.1) -> SemanticCache:
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed(text: str) -> list[float]:
        return next(iter(model.embed([text]))).tolist()

    def embed_many(texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in model.embed(texts)]

    async def aembed(text: str) -> list[float]:
        return await asyncio.to_thread(embed, text)

    async def aembed_many(texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(embed_many, texts)

    vectorizer = CustomVectorizer(
        embed=embed,
        embed_many=embed_many,
        aembed=aembed,
        aembed_many=aembed_many,
    )

    return SemanticCache(
        name="llmcache",
        redis_url=redis_url,
        distance_threshold=distance_threshold,
        vectorizer=vectorizer,
        filterable_fields=[{"name": "model", "type": "tag"}],
    )
