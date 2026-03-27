from typing import AsyncIterator

from ollama import AsyncClient, ResponseError

from app.providers.base import Provider, ProviderError
from app.schemas.request import CompletionRequest
from app.schemas.response import StreamChunk


class OllamaProvider(Provider):
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url
        self.client = AsyncClient(host=base_url, timeout=timeout)

    async def stream(self, request: CompletionRequest, model: str, request_id: str) -> AsyncIterator[StreamChunk]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        options = {
            **({"temperature": request.temperature} if request.temperature is not None else {}),
            **({"num_predict": request.max_tokens} if request.max_tokens is not None else {}),
        }

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
        except ResponseError as e:
            raise ProviderError(str(e), provider_name="ollama", status_code=e.status_code) from e
        except Exception as e:
            raise ProviderError(f"Ollama unreachable at {self.base_url}", provider_name="ollama") from e

    async def health_check(self) -> bool:
        try:
            await self.client.list()
            return True
        except Exception:
            return False
