from abc import ABC, abstractmethod
from typing import AsyncGenerator, Literal

from app.schemas.request import Message, CompletionRequest
from app.schemas.response import StreamChunk, UsageEvent


class ProviderError(Exception):
    def __init__(self, message: str, provider_name: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.provider_name = provider_name
        self.status_code = status_code


class AuthenticationError(ProviderError): pass
class RateLimitError(ProviderError): pass
class InvalidRequestError(ProviderError): pass
class ProviderUnavailableError(ProviderError): pass
class ProviderTimeoutError(ProviderError): pass


_STATUS_CODE_MAP: dict[int, type[ProviderError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AuthenticationError,
    413: InvalidRequestError,
    422: InvalidRequestError,
    429: RateLimitError,
    503: ProviderUnavailableError,
    504: ProviderTimeoutError,
    529: ProviderUnavailableError,
}


def map_provider_exception(
    e: Exception,
    provider_name: str,
    exception_map: dict[type[Exception], type[ProviderError]],
) -> ProviderError:
    status_code = getattr(e, "status_code", None) or getattr(e, "code", None)

    for exc_type, mapped_type in exception_map.items():
        if isinstance(e, exc_type):
            return mapped_type(str(e), provider_name=provider_name, status_code=status_code)

    if status_code and (mapped_type := _STATUS_CODE_MAP.get(status_code)):
        return mapped_type(str(e), provider_name=provider_name, status_code=status_code)

    return ProviderError(str(e), provider_name=provider_name, status_code=status_code)


class Provider(ABC):

    @abstractmethod
    def stream(self, request: CompletionRequest, model: str, request_id: str) -> AsyncGenerator[StreamChunk | UsageEvent, None]:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    async def close(self) -> None:
        pass

    def extract_system_prompt(self, messages: list[Message]) -> tuple[str | None, list[Message]]:
        if messages and messages[0].role == "system":
            return messages[0].content, messages[1:]
        return None, messages

    def normalize_finish_reason(self, raw: str) -> Literal["stop", "length", "error"]:
        if raw in ("end_turn", "stop_sequence", "stop", "STOP"):
            return "stop"
        if raw in ("max_tokens", "length", "MAX_TOKENS"):
            return "length"
        return "stop"
