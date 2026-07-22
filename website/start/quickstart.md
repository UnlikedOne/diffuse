# Quickstart

Once Diffuse is [installed](/start/installation), you are three commands away
from talking to the network.

## Talk to the network

```bash
diffuse chat
```

`diffuse chat` joins the network through the built-in sentinels, finds a servable
model, starts a local tokenizer, and streams the answer back token by token. Type
your message, or `/quit` to leave.

## See what is live

```bash
diffuse models
```

This lists the models the network can serve right now, meaning every layer is
covered by at least one reachable node. A model that is only partially held is
marked incomplete and is not listed as servable.

## Ask one question

```bash
diffuse query --prompt "Explain black holes in two sentences."
```

`query` is the one-shot form: it discovers a model, routes through it, prints the
answer, and exits.

## Contribute your machine

Give the network capacity by hosting a slice:

```bash
diffuse host --model Qwen/Qwen2.5-0.5B-Instruct
```

Your node profiles its own hardware, takes the slice the network needs most,
announces it, and starts serving. Leave it running to keep contributing.

## Use it from any OpenAI client

Run the local server and point any OpenAI-compatible tool at it:

```bash
diffuse serve --model Qwen/Qwen2.5-0.5B-Instruct
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello"}]}'
```

See the [OpenAI server guide](/guides/server) to connect LibreChat, Open WebUI,
or Continue.

## Where to go next

- [Chat with the network](/guides/chat)
- [Host a node](/guides/host)
- [Choosing a model](/guides/choosing-a-model)
- [CLI reference](/reference/cli)
