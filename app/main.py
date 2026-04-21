import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request

from app.api.exceptions import register_exception_handlers
from app.api.routes import completions, health
from app.config import get_settings
from app.gateway.cache import build_semantic_cache
from app.gateway.dispatcher import Dispatcher
from app.gateway.guard import PromptGuard
from app.gateway.router import Router
from app.gateway.semantic_router import SemanticRouter
from app.providers.anthropic import AnthropicProvider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    providers = {
        "anthropic": AnthropicProvider(
            api_key=settings.anthropic_api_key,
            timeout=settings.anthropic_timeout,
        ),
        "gemini": GeminiProvider(
            api_key=settings.gemini_api_key,
            timeout=settings.gemini_timeout,
        ),
        "ollama": OllamaProvider(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout,
        ),
    }

    ollama_ok = await providers["ollama"].health_check()
    if not ollama_ok:
        logger.warning("Ollama unreachable at %s — ollama provider disabled", settings.ollama_base_url)

    logger.info("Providers initialized: %s", list(providers.keys()))

    semantic_router = SemanticRouter()
    await semantic_router.initialize()

    router = Router(
        settings=settings,
        providers=providers,
        semantic_router=semantic_router
    )

    semantic_cache = build_semantic_cache(
        redis_url=settings.redis_url,
        distance_threshold=settings.semantic_cache_threshold
    )

    prompt_guard = PromptGuard(settings) if settings.prompt_guard_enabled else None

    app.state.dispatcher = Dispatcher(
        router=router,
        semantic_cache=semantic_cache,
        prompt_guard=prompt_guard,
    )

    yield

    ollama = providers.get("ollama")
    if ollama:
        await ollama.close()
    logger.info("Shutdown complete")


app = FastAPI(title="llm-gateway", version="0.1.0", lifespan=lifespan)

register_exception_handlers(app)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(completions.router, prefix="/v1", tags=["completions"])
