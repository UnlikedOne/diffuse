# OpenAI API reference

`diffuse serve` exposes an OpenAI-compatible HTTP API. This page is the endpoint
and schema reference. For client setup, see the
[server guide](/guides/server).

Base URL: `http://localhost:8080/v1`. The API key is not checked; any string
works.

## `GET /v1/models`

Returns the models the network can fully serve.

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1737460000,
      "owned_by": "diffuse"
    }
  ]
}
```

Only servable models appear. A partially held model is omitted.

## `POST /v1/chat/completions`

### Request

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | falls back to the server `--model` if omitted |
| `messages` | array | objects with `role` and `content` |
| `max_tokens` | integer | defaults to 512 |
| `stream` | boolean | Server-Sent Events when true |
| `temperature`, `top_p`, `n`, `presence_penalty`, `frequency_penalty` | any | accepted and ignored |

### Non-streamed response

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
`length` when it hit `max_tokens`.

### Streamed response

With `"stream": true`, the response is a Server-Sent Events stream. Each event is
a `data:` line with a chunk carrying `choices[0].delta.content`, and the stream
ends with `data: [DONE]`.

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Red"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Errors

```json
{ "error": { "message": "...", "type": "...", "code": "..." } }
```

| Status | Code | When |
|--------|------|------|
| `400` | `model_required` | no model in request and no default |
| `404` | `model_not_found` | model not present on the network |
| `503` | `model_incomplete` | model present but missing layer ranges |
| `503` | `route_unavailable` | no route could be built |

## Notes

- Decoding is greedy, so sampling parameters have no effect.
- Each request uses its own session id and clears it when generation ends, so no
  cache state is shared between requests.
