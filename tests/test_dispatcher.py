import pytest
from unittest.mock import AsyncMock, MagicMock
from app.gateway.dispatcher import Dispatcher, _format_event, _format_error
from app.gateway.errors import UnknownModelError, ConfigurationError
from app.schemas.response import StreamChunk, UsageEvent, ErrorEvent


class TestDispatcherStream:
    @pytest.mark.asyncio
    async def test_yields_chunks_from_provider(self):
        mock_router = MagicMock()
        mock_provider = MagicMock()

        async def gen():
            yield StreamChunk(id="req-1", model="test", delta="Hello")
            yield StreamChunk(id="req-1", model="test", delta=" world")

        mock_provider.stream = lambda *args, **kwargs: gen()
        mock_router.resolve = AsyncMock(return_value=(mock_provider, "test-model"))

        dispatcher = Dispatcher(mock_router)
        from app.schemas.request import CompletionRequest, Message

        request = CompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="hi")],
        )

        results = [r async for r in dispatcher.stream(request, "req-1")]

        assert len(results) == 3
        assert "Hello" in results[0]
        assert "world" in results[1]
        assert "[DONE]" in results[2]

    @pytest.mark.asyncio
    async def test_yields_error_on_unknown_model(self):
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(side_effect=UnknownModelError("unknown model"))
        mock_router.providers = {"test": MagicMock()}

        dispatcher = Dispatcher(mock_router)
        from app.schemas.request import CompletionRequest, Message

        request = CompletionRequest(
            model="unknown",
            messages=[Message(role="user", content="hi")],
        )

        results = [r async for r in dispatcher.stream(request, "req-1")]

        assert len(results) == 1
        assert "error" in results[0]
        assert "404" in results[0]

    @pytest.mark.asyncio
    async def test_yields_error_on_configuration_error(self):
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(
            side_effect=ConfigurationError("missing provider")
        )

        dispatcher = Dispatcher(mock_router)
        from app.schemas.request import CompletionRequest, Message

        request = CompletionRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
        )

        results = [r async for r in dispatcher.stream(request, "req-1")]

        assert len(results) == 1
        assert "error" in results[0]
        assert "500" in results[0]


class TestFormatEvent:
    def test_stream_chunk_returns_sse_string(self):
        chunk = StreamChunk(id="req-1", model="claude-sonnet-4-6", delta="Hello")
        result = _format_event(chunk, "req-1")
        assert result == f"data: {chunk.model_dump_json()}\n\n"

    def test_usage_event_returns_none(self):
        event = UsageEvent(
            input_tokens=10,
            output_tokens=25,
            model="claude-sonnet-4-6",
            provider="anthropic",
        )
        result = _format_event(event, "req-1")
        assert result is None

    def test_unregistered_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unhandled event type"):
            _format_event(object(), "req-1")


class TestFormatError:
    def test_returns_sse_error_string(self):
        result = _format_error("something went wrong", 500)
        assert "event: error" in result
        assert "500" in result
        assert "something went wrong" in result
