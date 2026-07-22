# OpenAI-compatible server

`diffuse serve` runs a small HTTP server on your machine that speaks the OpenAI
API. Any client that already talks to OpenAI (LibreChat, Open WebUI, Continue,
the official SDKs) can point at it and use the network instead.

```bash
diffuse serve --model Qwen/Qwen2.5-0.5B-Instruct
```

The server is a facade over the same pipeline `diffuse query` uses. Tokenization
happens locally, and traffic to the network is encrypted end to end. It never
bypasses the encrypted pipeline.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--port <n>` | `8080` | TCP port to listen on |
| `--host <addr>` | `127.0.0.1` | address to bind, loopback only by default |
| `--model <id>` | none | default model when a request omits one |
| `--bootstrap <urls>` | built-in sentinels | sentinels to discover the network |

::: warning Localhost by default
The API has no authentication. It binds to `127.0.0.1` so it is not exposed by
accident. If you pass a different `--host`, the server warns and keeps running,
so exposing it is a deliberate choice. Do not put it on an untrusted network
without an authenticating proxy in front. See [limitations](/limitations).
:::

## Endpoints

### `GET /v1/models`

Lists the models the network can actually serve. A partially held model is not
listed.

```bash
curl http://localhost:8080/v1/models
```

### `POST /v1/chat/completions`

Standard chat completions with `model`, `messages`, `max_tokens`, and `stream`.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Name three colours."}],
    "max_tokens": 64
  }'
```

Set `"stream": true` for a Server-Sent Events stream ending with `data: [DONE]`,
which is what LibreChat and Open WebUI expect by default.

## Supported and ignored parameters

Diffuse decodes greedily and does not sample, so sampling parameters have no
effect. They are accepted and ignored rather than rejected.

| Parameter | Status |
|-----------|--------|
| `model`, `messages`, `max_tokens`, `stream` | used |
| `temperature`, `top_p`, `n`, `presence_penalty`, `frequency_penalty` | ignored |

## Errors

Errors use the OpenAI error shape with the right HTTP status.

| Status | When |
|--------|------|
| `400` | no model in the request and no `--model` default |
| `404` | the model is not present on the network |
| `503` | the model is incomplete, naming the missing slices |

## Connect a client

The base URL is `http://localhost:8080/v1`. Any string works as the API key.

::: code-group

```yaml [LibreChat]
endpoints:
  custom:
    - name: Diffuse
      apiKey: "diffuse"
      baseURL: "http://localhost:8080/v1"
      models:
        default: ["Qwen/Qwen2.5-0.5B-Instruct"]
        fetch: true
```

```text [Open WebUI]
Settings -> Connections -> add an OpenAI API connection:
  API Base URL: http://localhost:8080/v1
  API Key:      diffuse
```

```json [Continue]
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

```python [Python SDK]
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="diffuse")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Name three colours."}],
)
print(resp.choices[0].message.content)
```

:::

## Next

- [OpenAI API reference](/reference/api)
- [Choosing a model](/guides/choosing-a-model)
