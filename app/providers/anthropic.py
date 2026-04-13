from typing import AsyncGenerator

from anthropic import (
    AsyncAnthropic,
    DefaultAioHttpClient,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError as AnthropicAuthenticationError,
    PermissionDeniedError,
    RateLimitError as AnthropicRateLimitError,
    BadRequestError,
    UnprocessableEntityError,
    InternalServerError,
)
from app.providers.base import (
    Provider,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ProviderUnavailableError,
    ProviderTimeoutError,
    map_provider_exception,
)
from app.schemas.request import CompletionRequest
from app.schemas.response import StreamChunk

# APITimeoutError must precede APIConnectionError — it is a subclass
_EXCEPTION_MAP: dict[type[Exception], type[ProviderError]] = {
    APITimeoutError: ProviderTimeoutError,
    APIConnectionError: ProviderUnavailableError,
    AnthropicAuthenticationError: AuthenticationError,
    PermissionDeniedError: AuthenticationError,
    AnthropicRateLimitError: RateLimitError,
    BadRequestError: InvalidRequestError,
    UnprocessableEntityError: InvalidRequestError,
    InternalServerError: ProviderUnavailableError,
}


class AnthropicProvider(Provider):
    def __init__(self, api_key: str, timeout: float):
        self.client = AsyncAnthropic(
            api_key=api_key, timeout=timeout, http_client=DefaultAioHttpClient()
        )

    async def stream(self, request: CompletionRequest, model: str, request_id: str) -> AsyncGenerator[StreamChunk, None]:
        system_prompt, messages = self.extract_system_prompt(request.messages)
        # Anthropic expects {role, content} dict
        mapped_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            kwargs = {
                "model": model,
                "max_tokens": request.max_tokens or 1024,
                "messages": mapped_messages,
            }
            if system_prompt is not None:
                kwargs["system"] = system_prompt
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature

            async with self.client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if (
                        event.type == "content_block_delta"
                        and event.delta.type == "text_delta"
                    ):
                        yield StreamChunk(
                            id=request_id,
                            model=model,
                            delta=event.delta.text,
                        )
                    elif event.type == "message_delta" and event.delta.stop_reason:
                        yield StreamChunk(
                            id=request_id,
                            model=model,
                            delta="",
                            finish_reason=self.normalize_finish_reason(event.delta.stop_reason),
                        )
        except Exception as e:
            raise map_provider_exception(e, provider_name="anthropic", exception_map=_EXCEPTION_MAP) from e

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
