# Ask one question

`diffuse query` is the one-shot form of chat. It discovers a model, routes through
it, prints the answer, and exits. It is the right tool for scripts and pipelines.

```bash
diffuse query --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain black holes in two sentences."
```

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <id>` | required | the model to query |
| `--prompt <text>` | required | the prompt to send |
| `--bootstrap <urls>` | built-in sentinels | sentinels to discover the network |
| `--max-tokens <n>` | 80 | maximum tokens to generate |

## Examples

A longer answer:

```bash
diffuse query --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Write a haiku about distributed systems." \
  --max-tokens 60
```

Against your own network:

```bash
diffuse query --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Ping" \
  --bootstrap http://your-sentinel:9440
```

## Errors you may see

- **Model not found on the network.** No node serves that model. Run
  `diffuse models` to see what is live.
- **Model is incomplete.** The model is present but some layer range is held by
  nobody. The error names the missing slices. See
  [replication](/concepts/replication).

## Next

- [Chat with the network](/guides/chat)
- [OpenAI-compatible server](/guides/server)
