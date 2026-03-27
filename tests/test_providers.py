# Tests for provider adapters (all using mocked SDK clients)
#
# AnthropicProvider:
#   test_extracts_system_prompt:
#     - messages with a system role → system passed as top-level param
#     - remaining messages passed in messages list
#
#   test_yields_stream_chunks:
#     - mock SDK stream yields content_block_delta events
#     - assert StreamChunk.delta matches expected text per event
#
#   test_wraps_sdk_error_in_provider_error:
#     - mock SDK raises anthropic.APIError
#     - assert ProviderError is raised with provider_name="anthropic"
#
# GeminiProvider:
#   test_maps_assistant_role_to_model:
#     - message with role "assistant" → sent to SDK with role "model"
#
#   test_yields_stream_chunks:
#     - mock SDK stream yields GenerateContentResponse chunks
#     - assert StreamChunk.delta matches chunk.text
#
# OllamaProvider:
#   test_passes_system_message_through:
#     - system role message is included in messages list as-is
#
#   test_parses_ndjson_stream:
#     - mock httpx response yields newline-delimited JSON lines
#     - assert correct StreamChunk per line
#
#   test_wraps_connection_error:
#     - mock httpx raises ConnectError
#     - assert ProviderError raised with useful message
