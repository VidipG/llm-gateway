from semantic_router import SemanticRouter as _SemanticRouter
from semantic_router.route import Route
from semantic_router.encoders import FastEmbedEncoder

from app.gateway.router import UnknownModelError
from app.schemas.request import Message


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
        Route(name="llama3.2", utterances=[
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


class SemanticRouter:
    def __init__(self):
        self._encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")
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
        query = messages[-1].content
        vector = (await self._encoder.acall([query]))[0]

        provider_result = await self._provider_router.acall(vector=vector)
        provider_choice = provider_result[0] if isinstance(provider_result, list) else provider_result
        if provider_choice.name is None:
            raise UnknownModelError("Semantic routing failed to match a provider")

        model_result = await self._model_routers[provider_choice.name].acall(vector=vector)
        model_choice = model_result[0] if isinstance(model_result, list) else model_result
        if model_choice.name is None:
            raise UnknownModelError(f"Semantic routing failed to match a model for provider '{provider_choice.name}'")

        return provider_choice.name, model_choice.name
