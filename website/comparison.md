# How Diffuse compares

An honest look at where Diffuse sits next to the alternatives. No option is best
at everything.

## At a glance

| | Diffuse | Hosted API | Local model | Other split inference |
|---|---|---|---|---|
| Your prompt leaves in clear text | never on the wire | yes, to the provider | no | often yes |
| A single party sees all traffic | no | yes | no | varies |
| Needs a powerful machine | no | no | yes | no |
| Runs models bigger than one machine | yes | yes | no | yes |
| Speed | slow | fast | fast | slow |
| Cost | free | paid | hardware | free |
| Setup | one binary | account | model download | varies |

## Versus a hosted API (OpenAI, Anthropic, others)

Hosted APIs are fast, reliable, and effortless. In exchange, every prompt you send
passes through one company that can log it, and you need an account and a budget.
Diffuse is slower and community-run, but no single party receives all of your
prompts and there is no account. Use a hosted API when you need speed and polish;
use Diffuse when you want decentralization and to keep your words off a central
server.

## Versus running a model locally

Running the model yourself (Ollama, llama.cpp, LM Studio) is the strongest privacy
option: nothing leaves your machine at all. The catch is you can only run models
that fit on your hardware. Diffuse exists for the case where the model is too big
for any one of your machines but a group of machines can hold it together. If your
machine can run the model alone, running it locally is simpler and more private.

## Versus other distributed inference

Other projects also split a model across peers. The usual difference is what a
peer sees: many send the prompt or its plain embeddings to another machine, so a
peer can read what you asked. Diffuse encrypts every hop end to end and is built so
the earliest, most invertible representations can stay on your device. It is also
explicit and honest about the current gap on the default path, where the entry
node still sees your token ids until local first layers land.

## Choosing

- Need speed and do not mind a provider seeing your prompts: **hosted API**.
- The model fits on your machine: **run it locally**.
- The model is too big for one machine and you value decentralization: **Diffuse**.
- You want a private network for a team or lab: **Diffuse**, self-hosted.
