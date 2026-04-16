import logging

from app.config import Settings
from app.gateway.errors import ConfigurationError, UnknownModelError
from app.gateway.semantic_router import SemanticRouter
from app.providers.base import Provider
from app.schemas.request import Message

logger = logging.getLogger(__name__)


class Router:
    def __init__(
        self,
        settings: Settings,
        providers: dict[str, Provider],
        semantic_router: SemanticRouter | None = None,
    ):
        self._settings = settings
        self._providers = providers
        self._semantic_router = semantic_router

    async def resolve(self, model: str, messages: list[Message]) -> tuple[Provider, str]:
        if model == "auto":
            if self._semantic_router is not None:
                return await self._resolve_semantic(messages)
            logger.warning("Falling back to static routing for 'auto'")
        return self._resolve_static(model)

    async def _resolve_semantic(self, messages: list[Message]) -> tuple[Provider, str]:
        provider_name, model = await self._semantic_router.resolve(messages)  # type: ignore[union-attr]
        return self._lookup_provider(provider_name), model

    def _resolve_static(self, model: str) -> tuple[Provider, str]:
        resolved = self._settings.model_aliases.get(model, model)
        provider_name = self._settings.model_routes.get(resolved)
        if not provider_name:
            raise UnknownModelError(f"No route for model '{resolved}'")
        return self._lookup_provider(provider_name), resolved

    def _lookup_provider(self, provider_name: str) -> Provider:
        provider = self._providers.get(provider_name)
        if not provider:
            raise ConfigurationError(f"Provider '{provider_name}' is not initialized")
        return provider
