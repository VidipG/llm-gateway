from app.config import Settings
from app.providers.base import Provider


class UnknownModelError(ValueError):
    pass


class ConfigurationError(RuntimeError):
    pass


class ModelRouter:
    def __init__(self, settings: Settings, providers: dict[str, Provider]):
        self.settings = settings
        self.providers = providers

    def resolve(self, model: str) -> tuple[Provider, str]:
        resolved = self.settings.model_aliases.get(model, model)

        provider_name = self.settings.model_routes.get(resolved)
        if not provider_name:
            raise UnknownModelError(f"No route for model '{resolved}'")

        provider = self.providers.get(provider_name)
        if not provider:
            raise ConfigurationError(f"Provider '{provider_name}' is not initialized")

        return provider, resolved
