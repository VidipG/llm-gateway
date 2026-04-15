import pytest

from app.gateway.dispatcher import _format_event
from app.schemas.response import StreamChunk, UsageEvent


class TestFormatEvent:

    def test_stream_chunk_returns_sse_string(self):
        chunk = StreamChunk(id="req-1", model="claude-sonnet-4-6", delta="Hello")
        result = _format_event(chunk, "req-1")
        assert result == f"data: {chunk.model_dump_json()}\n\n"

    def test_usage_event_returns_none(self):
        event = UsageEvent(input_tokens=10, output_tokens=25, model="claude-sonnet-4-6", provider="anthropic")
        result = _format_event(event, "req-1")
        assert result is None

    def test_usage_event_logs(self, caplog):
        import logging
        event = UsageEvent(input_tokens=10, output_tokens=25, model="claude-sonnet-4-6", provider="anthropic")
        with caplog.at_level(logging.INFO, logger="app.gateway.dispatcher"):
            _format_event(event, "req-1")
        assert "req-1" in caplog.text
        assert "10" in caplog.text
        assert "25" in caplog.text

    def test_unregistered_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unhandled event type"):
            _format_event(object(), "req-1")
