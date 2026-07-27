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
| `--max-tokens <n>` | 512 | longest answer per turn |

`--max-tokens` means two different things depending on the model, and the
difference matters. A model that answers with words stops when it has finished,
so the number is a ceiling it usually never reaches. A model that answers with
audio or pixels has no equivalent of "finished": the number *is* the length of
the answer, and every step is a round trip through the network. On MusicGen,
512 steps is about ten seconds of music and several minutes of waiting. Start
small.

## Commands

| Command | What it does |
|---------|--------------|
| `/help` | list these |
| `/attach <file>` | attach a file; its kind comes from the extension |
| `/image <file>` · `/audio <file>` · `/video <file>` | attach, saying what it is |
| `/files` | show what is attached to the next message |
| `/detach` | drop the attachments |
| `/reset` | clear conversation memory |
| `/clear` | clear the screen |
| `/stats` | turns and tokens so far |
| `/save [file]` | write the transcript as markdown |
| `/quit` | leave |

## Sending pictures, sound and video

Attach a file, then ask your question. The attachment applies to the next
message only, and is dropped once the answer arrives.

```
● › /image ~/photos/receipt.jpg
  ◆ attached image   receipt.jpg · 240 KB
● › What is the total on this receipt?
```

The file never leaves your machine. It is read here, turned into hidden states
by the part of the model that lives on your side, and only those activations
travel — the same thing that would travel for a text prompt. See
[multimodal](/guides/multimodal) for which models accept what.

If you attach something the model cannot read, the turn fails with a message
saying so rather than quietly answering as if no file had been sent.

## Answers that are not words

Some models answer with audio or an image rather than text. Chat handles those
the same way you would expect: it says what the model produces, generates it,
writes the file next to you, and prints the path.

```
● › a calm piano melody
  → this model answers with audio on 4 streams
  → generating over encrypted channel...

  answer:
  36.3 KB of audio/wav after 31 steps
  /home/you/diffuse-0000019f.wav
```

The transcript records the path, so `/save` keeps a usable record of what was
produced.

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
