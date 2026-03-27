from typing import Literal

from pydantic import BaseModel


class StreamChunk(BaseModel):
    id: str
    model: str
    delta: str
    finish_reason: Literal["stop", "length", "error"] | None = None


class ErrorEvent(BaseModel):
    error: str
    code: int
