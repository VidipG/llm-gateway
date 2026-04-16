import logging
from functools import singledispatch
from typing import AsyncIterator

from app.gateway.errors import ConfigurationError, UnknownModelError
from app.gateway.router import Router
from app.providers.base import ProviderError
from app.schemas.request import CompletionRequest
from app.schemas.response import ErrorEvent, StreamChunk, UsageEvent

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, router: Router):
        self.router = router

    async def stream(self, request: CompletionRequest, request_id: str) -> AsyncIterator[str]:
        try:
            provider, model = await self.router.resolve(request.model, request.messages)
        except UnknownModelError as e:
            yield _format_error(str(e), 404)
            return
        except ConfigurationError as e:
            yield _format_error(str(e), 500)
            return

        try:
            async for event in provider.stream(request, model, request_id):
                if sse := _format_event(event, request_id):
                    yield sse
        except ProviderError as e:
            logger.error("Provider error during stream [%s]: %s", e.provider_name, e.message)
            yield _format_error(e.message, e.status_code or 502)
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


def _format_error(message: str, code: int) -> str:
    return f"event: error\ndata: {ErrorEvent(error=message, code=code).model_dump_json()}\n\n"
