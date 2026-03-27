import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import completions, health
from app.config import get_settings
from app.gateway.router import ConfigurationError, UnknownModelError
from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderError
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

    app.state.providers = providers
    logger.info("Providers initialized: %s", list(providers.keys()))

    yield

    ollama = providers.get("ollama")
    if ollama:
        await ollama.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="llm-gateway",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    logger.error("Provider error [%s]: %s", exc.provider_name, exc.message)
    return JSONResponse(
        status_code=502,
        content={"error": exc.message, "provider": exc.provider_name},
    )


@app.exception_handler(UnknownModelError)
async def unknown_model_handler(request: Request, exc: UnknownModelError):
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    logger.error("Configuration error: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc)})


app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(completions.router, prefix="/v1", tags=["completions"])
