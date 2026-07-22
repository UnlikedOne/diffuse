# Chat with the network

`diffuse chat` is the interactive way to talk to a model on the network.

```bash
diffuse chat
```

It connects through the sentinels, picks a servable model (prompting you if there
is more than one), starts a local tokenizer, and streams the reply token by
token. Type a message and press enter. Type `/quit` to leave.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--bootstrap <urls>` | built-in sentinels | comma-separated sentinels to discover the network |
| `--memory` | off | keep conversation history across turns |

## Memory

By default each message is standalone: the model does not see earlier turns. Pass
`--memory` to carry the full history into each turn, so the conversation builds on
itself.

```bash
diffuse chat --memory
```

Without memory, a session is a series of independent questions. With it, it is a
continuous conversation. History lives only in your local process and is never
stored.

## Joining a specific network

To talk to a private network or your own sentinel instead of the public one:

```bash
diffuse chat --bootstrap http://your-sentinel:9440
```

## What stays local

Your machine runs the tokenizer, drives the generation loop, and decodes the
result. Traffic to the network is encrypted end to end. For exactly what a
serving node can and cannot see, read the [privacy and threat
model](/privacy).

## Next

- [Ask one question](/guides/query)
- [Choosing a model](/guides/choosing-a-model)
