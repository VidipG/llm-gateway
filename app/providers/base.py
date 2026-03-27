from abc import ABC, abstractmethod
from typing import AsyncIterator, Literal

from app.schemas.request import Message, CompletionRequest
from app.schemas.response import StreamChunk


class ProviderError(Exception):
    def __init__(self, message: str, provider_name: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.provider_name = provider_name
        self.status_code = status_code


class Provider(ABC):

    @abstractmethod
    async def stream(self, request: CompletionRequest, model: str, request_id: str) -> AsyncIterator[StreamChunk]:
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
