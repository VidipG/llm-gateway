from typing import AsyncGenerator

from ollama import AsyncClient, RequestError, ResponseError

from app.providers.base import (
    Provider,
    ProviderError,
    ProviderUnavailableError,
    map_provider_exception,
)
from app.schemas.request import CompletionRequest
from app.schemas.response import StreamChunk, UsageEvent

# RequestError (connection-level failure) maps directly to ProviderUnavailableError.
# ResponseError (HTTP error with status_code) falls through to _STATUS_CODE_MAP
# in map_provider_exception, which resolves 401→Auth, 429→RateLimit, 400→Invalid, etc.
_EXCEPTION_MAP: dict[type[Exception], type[ProviderError]] = {
    RequestError: ProviderUnavailableError,
}


class OllamaProvider(Provider):
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url
        self.client = AsyncClient(host=base_url, timeout=timeout)

    async def stream(self, request: CompletionRequest, model: str, request_id: str) -> AsyncGenerator[StreamChunk | UsageEvent, None]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        options = {
            **({"temperature": request.temperature} if request.temperature is not None else {}),
            **({"num_predict": request.max_tokens} if request.max_tokens is not None else {}),
        }
        input_tokens = None
        output_tokens = None

        try:
            async for chunk in await self.client.chat(
                model=model,
                messages=messages,
                stream=True,
                options=options,
            ):
                delta = chunk.message.content or ""
                finish_reason = self.normalize_finish_reason(chunk.done_reason or "") if chunk.done else None
                if delta or finish_reason:
                    yield StreamChunk(
                        id=request_id,
                        model=model,
                        delta=delta,
                        finish_reason=finish_reason,
                    )
                if chunk.done:
                    input_tokens = chunk.prompt_eval_count
                    output_tokens = chunk.eval_count
        except Exception as e:
            raise map_provider_exception(e, provider_name="ollama", exception_map=_EXCEPTION_MAP) from e

        yield UsageEvent(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            provider="ollama",
        )

    async def health_check(self) -> bool:
        try:
            await self.client.list()
            return True
        except Exception:
            return False
