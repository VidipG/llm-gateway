from typing import AsyncIterator

import anthropic

from app.providers.base import Provider, ProviderError
from app.schemas.request import CompletionRequest
from app.schemas.response import StreamChunk


class AnthropicProvider(Provider):
    def __init__(self, api_key: str, timeout: float):
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def stream(self, request: CompletionRequest, model: str, request_id: str) -> AsyncIterator[StreamChunk]:
        system_prompt, messages = self.extract_system_prompt(request.messages)
        mapped_messages = [{"role": m.role, "content": m.content} for m in messages]

        try:
            async with self.client.messages.stream(
                model=model,
                system=system_prompt or anthropic.NOT_GIVEN,
                messages=mapped_messages,
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature if request.temperature is not None else anthropic.NOT_GIVEN,
            ) as stream:
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
        except anthropic.APIError as e:
            raise ProviderError(str(e), provider_name="anthropic", status_code=e.status_code) from e

    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
