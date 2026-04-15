import logging
from functools import singledispatch
from typing import AsyncIterator

from app.config import Settings
from app.gateway.router import ModelRouter
from app.providers.base import Provider, ProviderError
from app.schemas.request import CompletionRequest
from app.schemas.response import ErrorEvent, StreamChunk, UsageEvent

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, providers: dict[str, Provider], settings: Settings):
        self.router = ModelRouter(settings=settings, providers=providers)

    async def stream(self, request: CompletionRequest, request_id: str) -> AsyncIterator[str]:
        provider, model = self.router.resolve(request.model)

        try:
            async for event in provider.stream(request, model, request_id):
                if sse := _format_event(event, request_id):
                    yield sse
        except ProviderError as e:
            logger.error("Provider error during stream [%s]: %s", e.provider_name, e.message)
            yield _format_error(e)
            return

        yield "data: [DONE]\n\n"


@singledispatch
def _format_event(event, request_id: str) -> str | None:
    raise TypeError(f"Unhandled event type: {type(event)}")


@_format_event.register
def _(event: StreamChunk, request_id: str) -> str:
    return f"data: {event.model_dump_json()}\n\n"


@_format_event.register
def _(event: UsageEvent, request_id: str) -> None:
    logger.info(
        "request complete request_id=%s model=%s provider=%s input_tokens=%s output_tokens=%s",
        request_id, event.model, event.provider, event.input_tokens, event.output_tokens,
    )


def _format_error(error: ProviderError) -> str:
    event = ErrorEvent(error=error.message, code=error.status_code or 502)
    return f"event: error\ndata: {event.model_dump_json()}\n\n"
