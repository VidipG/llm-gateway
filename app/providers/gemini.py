from typing import AsyncGenerator

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.genai.errors import ClientError, ServerError

from app.providers.base import (
    Provider,
    ProviderError,
    ProviderUnavailableError,
    map_provider_exception,
)
from app.schemas.request import CompletionRequest
from app.schemas.response import StreamChunk

# ServerError (5xx) maps directly to ProviderUnavailableError.
# ClientError (4xx) is not included — it falls through to _STATUS_CODE_MAP
# in map_provider_exception, which resolves 401→Auth, 429→RateLimit, 400→Invalid, etc.
_EXCEPTION_MAP: dict[type[Exception], type[ProviderError]] = {
    ServerError: ProviderUnavailableError,
}


class GeminiProvider(Provider):
    def __init__(self, api_key: str, timeout: float):
        timeout_ms = int(timeout * 1000)
        self.client = genai.Client(
            api_key=api_key,
            http_options=HttpOptions(timeout=timeout_ms),
        )

    async def stream(self, request: CompletionRequest, model: str, request_id: str) -> AsyncGenerator[StreamChunk, None]:
        system_prompt, messages = self.extract_system_prompt(request.messages)
        mapped_messages = [
            types.Content(
                role="model" if msg.role == "assistant" else msg.role,
                parts=[types.Part(text=msg.content)],
            )
            for msg in messages
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )

        try:
            async for chunk in await self.client.aio.models.generate_content_stream(
                model=model,
                contents=mapped_messages,
                config=config,
            ):
                finish_reason = None
                if chunk.candidates and chunk.candidates[0].finish_reason:
                    finish_reason = self.normalize_finish_reason(
                        str(chunk.candidates[0].finish_reason)
                    )
                if chunk.text or finish_reason:
                    yield StreamChunk(
                        id=request_id,
                        model=model,
                        delta=chunk.text or "",
                        finish_reason=finish_reason,
                    )
        except Exception as e:
            raise map_provider_exception(e, provider_name="gemini", exception_map=_EXCEPTION_MAP) from e

    async def health_check(self) -> bool:
        try:
            async for _ in await self.client.aio.models.list():
                return True
            return False
        except Exception:
            return False
