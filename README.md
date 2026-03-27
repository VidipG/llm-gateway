# llm-gateway

A FastAPI server that provides a single HTTP endpoint for routing LLM requests to multiple providers. Accepts a prompt via API, routes to the target model, and streams responses back via SSE.

**Supported providers:** Anthropic, Gemini, Ollama

---

## Architecture

```
POST /v1/chat/completions
        │
        ▼
   deps.py (auth)
        │
        ▼
  Dispatcher
        │
        ▼
  ModelRouter  ──► resolves model name / alias → Provider instance
        │
        ▼
  Provider.stream()  ──► Anthropic / Gemini / Ollama SDK
        │
        ▼
  StreamingResponse (SSE)
```

**Key components:**

- `app/config.py` — Pydantic `BaseSettings`. All configuration via environment variables or `.env` file. Model routing table and aliases are config-driven.
- `app/gateway/router.py` — maps model name (including aliases) to a `Provider` instance.
- `app/gateway/dispatcher.py` — orchestrates the call, formats SSE output, handles mid-stream errors.
- `app/providers/` — one class per provider. Each implements a common `Provider` ABC: `stream()` yields `StreamChunk` objects, `health_check()` returns a bool.
- `app/schemas/` — Pydantic models for request validation (`CompletionRequest`, `Message`) and response (`StreamChunk`, `ErrorEvent`).

Providers translate the normalized request format into their native SDK calls. System prompts are handled per-provider (Anthropic and Gemini take them as a top-level parameter; Ollama accepts them inline as a message role).

---

## Setup

**Requirements:** Python 3.14+, [uv](https://github.com/astral-sh/uv)

```bash
git clone <repo>
cd llm-gateway
uv sync
```

Create a `.env` file at the project root:

```bash
GATEWAY_API_KEY=your-gateway-key
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
OLLAMA_BASE_URL=http://localhost:11434  # optional, default
```

Start the server:

```bash
uv run python main.py
```

---

## Configuration

All settings are read from environment variables (or `.env`). The routing table and aliases can be overridden via env:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `RELOAD` | `false` | Uvicorn auto-reload |
| `GATEWAY_API_KEY` | required | Key for all `/v1/*` requests |
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `GEMINI_API_KEY` | required | Google Gemini API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `ANTHROPIC_TIMEOUT` | `60.0` | Timeout in seconds |
| `GEMINI_TIMEOUT` | `60.0` | Timeout in seconds |
| `OLLAMA_TIMEOUT` | `120.0` | Timeout in seconds |

**Default model routes:**

| Model | Provider |
|---|---|
| `claude-opus-4-6`, `claude-sonnet-4-6` | anthropic |
| `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro` | gemini |
| `llama3.2`, `mistral`, `qwen3.5` | ollama |

**Default aliases:** `fast` → `gemini-2.0-flash`, `smart` → `claude-opus-4-6`, `local` → `llama3.2`

To override the routing table via env:
```bash
MODEL_ROUTES='{"gemini-2.5-flash":"gemini","llama3.2":"ollama"}'
```

---

## API

### `POST /v1/chat/completions`

**Auth:** `x-api-key` header required.

**Request body:**

```json
{
  "model": "gemini-2.5-flash",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": true
}
```

`stream` defaults to `true`. `temperature` and `max_tokens` are optional.

**Response:** `text/event-stream`

```
data: {"id":"<request-id>","model":"gemini-2.5-flash","delta":"Hello","finish_reason":null}

data: {"id":"<request-id>","model":"gemini-2.5-flash","delta":"!","finish_reason":"stop"}

data: [DONE]
```

**Error event (mid-stream):**
```
event: error
data: {"error":"...","code":502}
```

---

### `GET /health`

Returns `{"status": "ok"}`. No auth required.

### `GET /health/providers`

Returns reachability status for each provider. No auth required.

```json
{
  "anthropic": "ok",
  "gemini": "ok",
  "ollama": "unreachable"
}
```

---

## curl Examples

**Gemini** (aliases `fast`, `smart`, `local` are also accepted in place of model names):
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-gateway-key" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "say hello"}]
  }' --no-buffer
```

**Anthropic:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-gateway-key" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "say hello"}]
  }' --no-buffer
```

**Ollama** (requires `ollama serve` and model pulled locally):
```bash
ollama pull qwen3.5

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-gateway-key" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "say hello"}]
  }' --no-buffer
```

**With system prompt:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-gateway-key" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {"role": "system", "content": "Reply only in haiku."},
      {"role": "user", "content": "describe the ocean"}
    ]
  }' --no-buffer
```

**Health check:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/providers
```

`--no-buffer` prevents curl from buffering SSE chunks — without it you won't see streaming output until the response completes.
