# systemone-qwen

A self-hosted **System One** decision API backed by a Qwen model served by
[llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).

## What a System One model is

A System One model does not write text. It receives a **state** (a message, a record, a chat
log) and typed **questions**, and answers each one with a probability distribution your code
can act on directly:

- `noul`: a yes/no question, answered with the probability of yes;
- `choice`: pick one option from a set, with a probability per option and a confidence;
- `score`: rate against ordered levels, with the probability-weighted level.

TypeSafe's **Jev** is the reference model of this kind, available through
[TypeSafe](https://docs.typesafe.ai/api.md) and [OpenRouter](https://openrouter.ai/docs/client-sdks/python/sdks/systemone/README).

## What this project does

It exposes the same wire format as Jev (`POST /api/v1/systemone`, with the OpenRouter and
TypeSafe paths as aliases) on top of an open-weight Qwen model running on your own hardware.
A client written for Jev switches to it by changing only its base URL.

Nothing is generated. Each option is shown to the model with a letter (A, B, C…), and its
probability is the model's first-token logprob for that letter, read from llama-server's
`/v1/completions`. Reasoning is closed by default; a reasoning budget can be enabled per model
for harder decisions.

## Why

- **Data stays on your infrastructure**: no state leaves your network.
- **No per-decision cost** beyond the hardware you already run.
- **Measured, not assumed**: on [JevBench](https://github.com/fstandhartinger/jevbench)'s 231
  public tasks, Qwen3.5-9B read this way scores 0.827 against Jev's 0.857, and 0.853 with a
  512-token reasoning budget.

## Design

- The service is plain Python and holds no weights: **no GPU, CUDA or ROCm in its image**. The
  GPU stack lives in llama-server (ROCm, CUDA, Vulkan or CPU builds all work).
- One multi-arch container image (`linux/amd64`, `linux/arm64`), published to GHCR by CI.
- A YAML registry maps request `model` names to llama-server backends, each with its own
  settings: option wording, calibration temperature, reasoning budget, aliases such as
  `jev-latest`.

## Status

The first implementation (service, tests, container, CI and documentation) is under review in
the initial pull request.

This is an independent project, not affiliated with TypeSafe or OpenRouter.

## License

MIT. Model weights keep their own licences (Qwen: Apache-2.0).
