import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderError
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.schemas.request import CompletionRequest, Message


def make_request(messages: list[dict], **kwargs) -> CompletionRequest:
    return CompletionRequest(
        model="test-model",
        messages=[Message(**m) for m in messages],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:

    def make_provider(self) -> AnthropicProvider:
        with patch("app.providers.anthropic.anthropic.AsyncAnthropic"):
            return AnthropicProvider(api_key="test", timeout=30.0)

    @pytest.mark.asyncio
    async def test_extracts_system_prompt(self):
        provider = self.make_provider()
        request = make_request([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ])

        text_event = MagicMock()
        text_event.type = "content_block_delta"
        text_event.delta.type = "text_delta"
        text_event.delta.text = "Hi"

        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(return_value=stream_cm)
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        stream_cm.__aiter__ = MagicMock(return_value=aiter([text_event]))
        provider.client.messages.stream = MagicMock(return_value=stream_cm)

        _ = [c async for c in provider.stream(request, "claude-sonnet-4-6", "req-1")]

        call_kwargs = provider.client.messages.stream.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    @pytest.mark.asyncio
    async def test_yields_stream_chunks(self):
        provider = self.make_provider()
        request = make_request([{"role": "user", "content": "Hello"}])

        events = []
        for text in ["Hello", " world"]:
            e = MagicMock()
            e.type = "content_block_delta"
            e.delta.type = "text_delta"
            e.delta.text = text
            events.append(e)

        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(return_value=stream_cm)
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        stream_cm.__aiter__ = MagicMock(return_value=aiter(events))  # async generator
        provider.client.messages.stream = MagicMock(return_value=stream_cm)

        chunks = [c async for c in provider.stream(request, "claude-sonnet-4-6", "req-1")]

        assert len(chunks) == 2
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"

    @pytest.mark.asyncio
    async def test_wraps_sdk_error_in_provider_error(self):
        import anthropic as anthropic_sdk

        provider = self.make_provider()
        request = make_request([{"role": "user", "content": "Hello"}])

        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(side_effect=anthropic_sdk.APIStatusError(
            message="Unauthorized",
            response=MagicMock(status_code=401),
            body={},
        ))
        stream_cm.__aexit__ = AsyncMock(return_value=False)
        provider.client.messages.stream = MagicMock(return_value=stream_cm)

        with pytest.raises(ProviderError) as exc_info:
            async for _ in provider.stream(request, "claude-sonnet-4-6", "req-1"):
                pass

        assert exc_info.value.provider_name == "anthropic"
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------

class TestGeminiProvider:

    def make_provider(self) -> GeminiProvider:
        with patch("app.providers.gemini.genai.Client"):
            return GeminiProvider(api_key="test", timeout=30.0)

    @pytest.mark.asyncio
    async def test_maps_assistant_role_to_model(self):
        provider = self.make_provider()
        request = make_request([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ])

        async def fake_stream(**kwargs):
            yield MagicMock(text="Good", candidates=[])

        provider.client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())

        chunks = [c async for c in provider.stream(request, "gemini-2.0-flash", "req-1")]

        call_kwargs = provider.client.aio.models.generate_content_stream.call_args.kwargs
        contents = call_kwargs["contents"]
        roles = [c.role for c in contents]
        assert "model" in roles
        assert "assistant" not in roles

    @pytest.mark.asyncio
    async def test_yields_stream_chunks(self):
        provider = self.make_provider()
        request = make_request([{"role": "user", "content": "Hello"}])

        async def fake_stream(**kwargs):
            for text in ["Hello", " there"]:
                chunk = MagicMock()
                chunk.text = text
                chunk.candidates = []
                yield chunk

        provider.client.aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())

        chunks = [c async for c in provider.stream(request, "gemini-2.0-flash", "req-1")]

        assert len(chunks) == 2
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " there"


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

class TestOllamaProvider:

    def make_provider(self) -> OllamaProvider:
        with patch("app.providers.ollama.AsyncClient"):
            return OllamaProvider(base_url="http://localhost:11434", timeout=30.0)

    @pytest.mark.asyncio
    async def test_passes_system_message_through(self):
        provider = self.make_provider()
        request = make_request([
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ])

        async def fake_chat(**kwargs):
            chunk = MagicMock()
            chunk.message.content = "Hi"
            chunk.done = True
            chunk.done_reason = "stop"
            yield chunk

        provider.client.chat = AsyncMock(return_value=fake_chat())

        chunks = [c async for c in provider.stream(request, "llama3.2", "req-1")]

        call_kwargs = provider.client.chat.call_args.kwargs
        roles = [m["role"] for m in call_kwargs["messages"]]
        assert "system" in roles

    @pytest.mark.asyncio
    async def test_yields_stream_chunks(self):
        provider = self.make_provider()
        request = make_request([{"role": "user", "content": "Hello"}])

        async def fake_chat(**kwargs):
            for text in ["Hello", " world"]:
                chunk = MagicMock()
                chunk.message.content = text
                chunk.done = False
                chunk.done_reason = None
                yield chunk
            final = MagicMock()
            final.message.content = ""
            final.done = True
            final.done_reason = "stop"
            yield final

        provider.client.chat = AsyncMock(return_value=fake_chat())

        chunks = [c async for c in provider.stream(request, "llama3.2", "req-1")]

        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert chunks[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_wraps_connection_error(self):
        provider = self.make_provider()
        request = make_request([{"role": "user", "content": "Hello"}])

        provider.client.chat = AsyncMock(side_effect=Exception("Connection refused"))

        with pytest.raises(ProviderError) as exc_info:
            async for _ in provider.stream(request, "llama3.2", "req-1"):
                pass

        assert exc_info.value.provider_name == "ollama"
        assert "localhost" in exc_info.value.message


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def aiter(items):
    for item in items:
        yield item
