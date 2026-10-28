# systemone-qwen

A self-hosted **System One** decision API backed by a Qwen model on
[llama-server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server). It answers the
same wire format as TypeSafe's Jev (on [OpenRouter](https://openrouter.ai/docs/client-sdks/python/sdks/systemone/README)
or [TypeSafe](https://docs.typesafe.ai/api.md)), so a Jev client switches by changing its base URL.

Send a state and typed questions (`noul`, `choice`, `score`); get probability distributions
back. Nothing is generated: each option gets a letter, and its probability is the model's
first-token logprob for that letter.

On JevBench's 231 public tasks, Qwen3.5-9B through this method scores **0.827** against Jev's
**0.857**, and **0.853** with a 512-token reasoning budget. See [benchmarks](docs/benchmarks.md).

This is an independent project, not affiliated with TypeSafe or OpenRouter.

## What it needs

- A llama-server serving a Qwen GGUF (Qwen3.5/3.6/3.8 ChatML with `<think>`), reachable over HTTP.
- Nothing else: **no GPU, CUDA or ROCm in this image**. The GPU stack lives in llama-server.
  The image is plain Python on Debian slim, for `linux/amd64` and `linux/arm64`.

## Quick start

```sh
# llama-server already running on :8080 (see docs/deployment.md for decision-friendly flags)
podman run --rm -p 127.0.0.1:8000:8000 -e SYSTEMONE_BACKEND_URL=http://host.containers.internal:8080 \
  ghcr.io/dohr-michael/systemone-qwen:latest

curl -s localhost:8000/api/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "qwen-decision",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages"}}
  }
}'
```

```json
{"id": "gen-dec-…", "model": "qwen-decision", "provider": "systemone-qwen",
 "answers": {"urgent": {"type": "noul", "noul": 0.989},
             "department": {"type": "choice", "choice": "technical",
                            "probabilities": {"billing": 0.30, "technical": 0.70}, "confidence": 0.40}},
 "usage": {"input_tokens": 219, "output_tokens": 2, "cost": 0}}
```

`compose.yaml` and `scripts/run.sh` do the same; `CONTAINER_ENGINE=docker` selects Docker.

## Documentation

- [API](docs/api.md): routes, request and response schema, errors.
- [Configuration](docs/configuration.md): environment, the YAML model registry, why each default.
- [Deployment](docs/deployment.md): llama-server flags, Kubernetes with pod affinity, LiteLLM pass-through.
- [Benchmarks](docs/benchmarks.md): JevBench results, settings, calibration, French.

## Development

```sh
uv sync
uv run pytest            # fake llama-server, no model needed
SYSTEMONE_BACKEND_URL=http://localhost:8080 uv run systemone-qwen
scripts/build.sh         # local image: localhost/systemone-qwen:dev
benchmark/jevbench.sh local http://127.0.0.1:8000 qwen-decision
```

Images are built, smoke-tested without a model server, and published to
`ghcr.io/dohr-michael/systemone-qwen` by [GitHub Actions](.github/workflows/image.yml):
`latest` from `main`, `X.Y.Z` / `X.Y` from `vX.Y.Z` tags, and `sha-…` for every build.

## License

MIT. Model weights keep their own licences (Qwen: Apache-2.0).
