# API

systemone-qwen speaks the System One wire format published by
[OpenRouter](https://openrouter.ai/docs/client-sdks/python/sdks/systemone/README) and
[TypeSafe](https://docs.typesafe.ai/api.md). A client written for Jev switches to this service
by changing its base URL (and, unless an alias covers it, the `model`).

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/api/v1/systemone` | OpenRouter SDK path (`server_url` ending in `/api/v1`) |
| `POST` | `/v1/systemone` | TypeSafe path, also used by the JevBench `typesafe` adapter |
| `POST` | `/api/alpha/decisions` | OpenRouter's earlier alpha path |
| `GET` | `/v1/models`, `/api/v1/models` | Configured models and aliases |
| `GET` | `/health` | Liveness: the process answers |
| `GET` | `/ready` | Readiness: every configured llama-server answers `/health` (503 otherwise) |

| Client | Base URL |
| --- | --- |
| Jev on OpenRouter | `https://openrouter.ai/api/v1` |
| Jev on TypeSafe | `https://api.typesafe.ai/v1` |
| This service | `http://<host>:8000/api/v1` (OpenRouter style) or `http://<host>:8000/v1` (TypeSafe style) |

## Request

```json
{
  "model": "qwen-decision",
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations"}
    },
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
```

- `state`: string, object or array.
- `questions`: 1 to 255 named questions. `instructions` and every criterion accept a string,
  an object or an array.
- `noul`: optional `criteria.true` / `criteria.false`.
- `choice`: 2 to 255 options; a `null` description means the key speaks for itself.
- `score`: 2 to 10 ordered levels.
- `model`: a configured name or alias; omitted means the default model.
- OpenRouter-only fields (`provider`, `session_id`, `user`, `trace`) are accepted and ignored.

## Response

```json
{
  "id": "gen-dec-1c9e…",
  "model": "qwen-decision",
  "provider": "systemone-qwen",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.989},
    "department": {"type": "choice", "choice": "technical",
                   "probabilities": {"billing": 0.30, "technical": 0.70}, "confidence": 0.40},
    "frustration": {"type": "score", "score": 1.0, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.01, "1": 0.98, "2": 0.01}, "confidence": 0.98}
  },
  "usage": {"input_tokens": 219, "output_tokens": 3, "cost": 0}
}
```

- `noul`: probability of yes.
- `choice`: the most probable option, the full distribution and `confidence`.
- `score`: the probability-weighted level (it can land between levels), the `legend`, the
  per-level distribution and `confidence`.
- `confidence` = (n·p_max − 1)/(n − 1): the top probability rescaled so that a uniform
  distribution is 0. This matches Jev's answers.
- `usage.output_tokens` counts reasoning tokens when thinking is enabled; `cost` is always 0.

Probabilities are not rounded (Jev rounds to two decimals).

## Errors

| Status | When |
| --- | --- |
| 404 | `model` is neither configured nor an alias (and `allow_unknown_model` is false) |
| 422 | Invalid request, or more than 26 options without llama-server `/tokenize` access |
| 502 | The llama-server failed or returned no usable logprobs |

Error bodies follow `{"error": {"message": "...", "code": 404}}`; FastAPI's own validation
errors (422) use its `{"detail": [...]}` body.
