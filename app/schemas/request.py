from typing import Literal

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = True

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages: list[Message]) -> list[Message]:
        if not messages:
            raise ValueError("messages must not be empty")
        if messages[-1].role != "user":
            raise ValueError("last message must have role 'user'")
        return messages
