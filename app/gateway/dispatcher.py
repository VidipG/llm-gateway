import logging
from typing import AsyncIterator

from app.config import Settings
from app.gateway.router import ModelRouter
from app.providers.base import Provider, ProviderError
from app.schemas.request import CompletionRequest
from app.schemas.response import ErrorEvent, StreamChunk

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, providers: dict[str, Provider], settings: Settings):
        self.router = ModelRouter(settings=settings, providers=providers)

    async def stream(self, request: CompletionRequest, request_id: str) -> AsyncIterator[str]:
        provider, model = self.router.resolve(request.model)

        try:
            async for chunk in provider.stream(request, model, request_id):
                yield _format_chunk(chunk)
        except ProviderError as e:
            logger.error("Provider error during stream [%s]: %s", e.provider_name, e.message)
            yield _format_error(e)
            return

        yield "data: [DONE]\n\n"


def _format_chunk(chunk: StreamChunk) -> str:
    return f"data: {chunk.model_dump_json()}\n\n"


def _format_error(error: ProviderError) -> str:
    event = ErrorEvent(error=error.message, code=error.status_code or 502)
    return f"event: error\ndata: {event.model_dump_json()}\n\n"
