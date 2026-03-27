# Tests for ModelRouter
#
# test_resolves_known_model:
#   - router.resolve("gemini-2.0-flash") → (GeminiProvider, "gemini-2.0-flash")
#
# test_expands_alias:
#   - router.resolve("fast") → (GeminiProvider, "gemini-2.0-flash")
#
# test_raises_on_unknown_model:
#   - router.resolve("gpt-4o") → raises UnknownModelError
#
# test_raises_on_missing_provider:
#   - route table maps "some-model" → "anthropic"
#   - providers dict is empty
#   - router.resolve("some-model") → raises ConfigurationError
