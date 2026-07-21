# OpenAI-compatible API

`diffuse serve` runs a small HTTP server on your machine that speaks the OpenAI
API. Any client that already talks to OpenAI (LibreChat, Open WebUI, Continue,
the official SDKs) can point at it and use the Diffuse network instead.

The server is only a facade over the same pipeline `diffuse query` uses.
Tokenization happens locally on your machine, and your prompt only ever leaves
over the encrypted, layer-by-layer route to the network. The server never sends a
prompt in clear text and never bypasses the encrypted pipeline.

## Starting the server

```
diffuse serve --model Qwen/Qwen2.5-0.5B-Instruct
```

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--port` | `8080` | TCP port to listen on. |
| `--host` | `127.0.0.1` | Address to bind. Loopback only by default. |
| `--model` | none | Default model when a request omits `model`. |
| `--bootstrap` | built-in sentinel | Comma-separated sentinels used to discover the network. |

The server discovers the network, starts a local tokenizer worker, and then
listens for HTTP requests. It prints the address it is serving on once ready.

### Why it listens on localhost by default

The API has no authentication. Binding it to `0.0.0.0` without an authenticating
proxy in front would let anyone who can reach the port spend your node's capacity
and use the models it can route to. For that reason `--host` defaults to
`127.0.0.1`, which only accepts connections from the same machine. If you pass a
different address the server prints a warning and keeps running, so exposing it is
a deliberate choice. See the note in the limitations page before doing so.

## Endpoints

### `GET /v1/models`

Lists the models the network can actually serve, meaning every layer of the model
is covered by at least one reachable peer. A model that is only partially held is
not listed.

```
curl http://localhost:8080/v1/models
```

```json
{
  "object": "list",
  "data": [
    { "id": "Qwen/Qwen2.5-0.5B-Instruct", "object": "model", "created": 1737460000, "owned_by": "diffuse" }
  ]
}
```

### `POST /v1/chat/completions`

Accepts the standard chat completion body: `model`, `messages` (each with `role`
and `content`), `max_tokens`, and `stream`. The full message history is passed to
the worker, which applies the model's chat template.

Non-streamed request:

```
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Name three colours."}],
    "max_tokens": 64
  }'
```

Non-streamed response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1737460000,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Red, green, and blue." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19 }
}
```

`finish_reason` is `stop` when the model emitted its end-of-sequence token and
`length` when it stopped because `max_tokens` was reached.

Streamed request (`stream: true`):

```
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Name three colours."}],
    "stream": true
  }'
```

The response is a Server-Sent Events stream. Each event is a `data:` line holding
a chunk with `choices[0].delta.content`, and the stream ends with `data: [DONE]`.
This is what LibreChat and Open WebUI expect by default.

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Red"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Supported and ignored parameters

Diffuse decodes greedily and does not sample, so the parameters that control
sampling have no effect. They are accepted and ignored rather than rejected, so
existing clients keep working without changes.

| Parameter | Status |
| --- | --- |
| `model` | Used. Falls back to `--model` when omitted. |
| `messages` | Used. Passed to the chat template with roles intact. |
| `max_tokens` | Used. Defaults to 512 when omitted. |
| `stream` | Used. |
| `temperature` | Ignored (greedy decoding). |
| `top_p` | Ignored (greedy decoding). |
| `n` | Ignored. One completion is always returned. |
| `presence_penalty` | Ignored. |
| `frequency_penalty` | Ignored. |
| Any other field | Ignored. |

## Sessions

Each request gets its own UUID session id, and the session is cleared on the
serving nodes when generation ends. Session ids are never reused between
requests, so one request cannot see another request's cached state.

## Errors

Errors use the OpenAI error shape with an appropriate HTTP status code:

```json
{ "error": { "message": "...", "type": "...", "code": "..." } }
```

| Status | When |
| --- | --- |
| `400` | No `model` in the request and no `--model` default configured. |
| `404` | The requested model is not present on the network. |
| `503` | The model is present but incomplete, or no route could be built. |

A `503` for an incomplete model names the missing layer ranges, for example
`model X is incomplete: no peer serves layers 6:64 (of 64 total)`.

## Connecting clients

The base URL is `http://localhost:8080/v1`. Any string works as the API key,
since the server does not check it.

### LibreChat

In `librechat.yaml`, add a custom endpoint:

```yaml
endpoints:
  custom:
    - name: Diffuse
      apiKey: "diffuse"
      baseURL: "http://localhost:8080/v1"
      models:
        default: ["Qwen/Qwen2.5-0.5B-Instruct"]
        fetch: true
      titleConvo: false
```

`fetch: true` lets LibreChat load the model list from `GET /v1/models`.

### Open WebUI

Under Settings, then Connections, add an OpenAI API connection:

- API Base URL: `http://localhost:8080/v1`
- API Key: any non-empty string, for example `diffuse`

Open WebUI streams by default, which the server supports.

### Continue

In `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Diffuse",
      "provider": "openai",
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "diffuse"
    }
  ]
}
```

### Official SDKs

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="diffuse")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Name three colours."}],
)
print(resp.choices[0].message.content)
```

```javascript
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://localhost:8080/v1", apiKey: "diffuse" });
const resp = await client.chat.completions.create({
  model: "Qwen/Qwen2.5-0.5B-Instruct",
  messages: [{ role: "user", content: "Name three colours." }],
});
console.log(resp.choices[0].message.content);
```
