class UnknownModelError(ValueError):
    pass


class ConfigurationError(RuntimeError):
    pass


class PromptInjectionError(ValueError):
    pass


class ContentPolicyError(ValueError):
    pass
