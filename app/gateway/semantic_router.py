import asyncio
from typing import List

from semantic_router import SemanticRouter as _SemanticRouter
from semantic_router.route import Route
from semantic_router.encoders import FastEmbedEncoder

from app.gateway.errors import UnknownModelError
from app.schemas.request import Message


class _AsyncFastEmbedEncoder(FastEmbedEncoder):
    async def acall(self, docs: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(super().__call__, docs)


_PROVIDER_ROUTES: list[Route] = [
    Route(name="anthropic", utterances=[
        "analyze this complex document",
        "help me reason through this difficult problem",
        "write a nuanced long-form essay",
        "summarize this lengthy research paper",
        "explain the deeper implications of this",
    ]),
    Route(name="gemini", utterances=[
        "give me a quick answer",
        "search for current information on this",
        "fast response needed",
        "structured output please",
        "describe what's in this image",
    ]),
    Route(name="ollama", utterances=[
        "keep this conversation private",
        "don't send this to any external API",
        "run this locally on my machine",
        "this contains sensitive data",
        "I need offline processing",
    ]),
]

_MODEL_ROUTES: dict[str, list[Route]] = {
    "anthropic": [
        Route(name="claude-sonnet-4-6", utterances=[
            "everyday coding task",
            "quick question",
            "straightforward request",
            "cost-effective response",
        ]),
        Route(name="claude-opus-4-6", utterances=[
            "hardest problem you can solve",
            "requires maximum intelligence",
            "very long context window needed",
            "deep multi-step reasoning",
        ]),
    ],
    "gemini": [
        Route(name="gemini-2.0-flash", utterances=[
            "fast and cheap",
            "high volume request",
            "low latency needed",
            "simple quick task",
        ]),
        Route(name="gemini-2.5-pro", utterances=[
            "complex reasoning task",
            "deep analysis required",
            "difficult problem to solve",
            "thorough detailed response",
        ]),
    ],
    "ollama": [
        Route(name="qwen3.5", utterances=[
            "general purpose task",
            "chat with a local model",
            "standard local inference",
        ]),
        Route(name="mistral", utterances=[
            "follow these instructions precisely",
            "structured output from local model",
            "instruction tuned task",
        ]),
    ],
}


def _matched_name(result, label: str) -> str:
    choice = result[0] if isinstance(result, list) else result
    if choice.name is None:
        raise UnknownModelError(f"Semantic routing could not match a {label}")
    return choice.name


class SemanticRouter:
    def __init__(self):
        self._encoder = _AsyncFastEmbedEncoder(name="BAAI/bge-small-en-v1.5")
        self._provider_router = _SemanticRouter(encoder=self._encoder, aggregation="max")
        self._model_routers = {
            provider: _SemanticRouter(encoder=self._encoder, aggregation="max")
            for provider in _MODEL_ROUTES
        }

    async def initialize(self) -> None:
        await self._provider_router.aadd(_PROVIDER_ROUTES)
        for provider, routes in _MODEL_ROUTES.items():
            await self._model_routers[provider].aadd(routes)

    async def resolve(self, messages: list[Message]) -> tuple[str, str]:
        vector = (await self._encoder.acall([messages[-1].content]))[0]
        provider = _matched_name(await self._provider_router.acall(vector=vector), "provider")
        model = _matched_name(await self._model_routers[provider].acall(vector=vector), "model")
        return provider, model
