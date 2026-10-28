# Configuration

## Single model from the environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `SYSTEMONE_BACKEND_URL` | — | llama-server root URL (required without `SYSTEMONE_CONFIG`) |
| `SYSTEMONE_MODEL` | `qwen-decision` | Name clients send in `model` |
| `SYSTEMONE_SERVED_MODEL` | — | `model` forwarded to the backend (needed behind LiteLLM) |
| `SYSTEMONE_API_KEY_ENV` | — | Name of the variable holding the backend API key |
| `SYSTEMONE_HOST` / `SYSTEMONE_PORT` | `0.0.0.0` / `8000` | Listen address |
| `SYSTEMONE_LOG_LEVEL` | `info` | uvicorn log level |

## Model registry

`SYSTEMONE_CONFIG=/path/config.yaml` takes precedence and supports several models, aliases and
every per-model setting. [`config.example.yaml`](../config.example.yaml) documents each field.

| Field | Default | Meaning |
| --- | --- | --- |
| `base_url` | — | llama-server root URL |
| `served_model` | — | `model` forwarded to the backend |
| `api_key_env` | — | Variable holding the backend key |
| `noul_labels` | `prefixed` | `plain` (Yes/No), `criteria`, or `prefixed` (`Yes: <criterion>`) |
| `debias` | `false` | Average noul over both option orders |
| `calibration_temperature` | `1.0` | p ∝ p^(1/T) on every distribution |
| `think_tokens` | `0` | Reasoning budget before reading the answer |
| `top_logprobs` | `100` | First-token alternatives requested |
| `concurrency` | `1` | Parallel backend requests per decision request |
| `timeout_s` | `120` | Per backend request |
| `extra_body` | `{cache_prompt: true}` | Merged into every completion request |
| `system_prompt` | `Choose one option. Answer only with its code.` | Keep it: verbose prompts lost 3 points |

## How each setting was chosen

All numbers are JevBench public-set accuracy (231 tasks) for Qwen3.5-9B; see
[benchmarks.md](benchmarks.md).

- **`noul_labels`**: `plain` 0.831, `criteria` 0.827, `prefixed` 0.827, all within run-to-run
  noise at 9B. `prefixed` is the default because it was the only safe choice for small models:
  Qwen3-0.6B answered "yes" to almost everything with `plain` and "no" with `criteria`.
- **`debias`**: no change at 9B (0.827) for 1.5× the latency. It only rescues sub-1B models.
- **Choice option keys**: always shown (`billing: Payments…`); hiding them cost 1.7 points.
- **`system_prompt`**: a longer, more explicit prompt scored 0.797 (−3 points, −5.4 on the hard tier).
- **`think_tokens: 512`**: 0.853, level with Jev (0.857), but ~21.8 s per decision instead of
  0.36 s. Offer it as a separate model name (see `qwen-decision-think` in the example).
- **`calibration_temperature`**: does not change accuracy. Fitted by NLL on half the task
  groups and scored on the other half: Qwen3.5-9B T≈1.5 (ECE 0.062 → 0.054), with thinking
  T≈1.75 (0.066 → 0.050). Fit your own on data that resembles your traffic.

## Options and the 26-letter limit

Options are lettered A, B, C… and the answer is the first token's logprob for each letter.
Up to 26 options this needs nothing beyond OpenAI-style `logprobs`. Beyond 26 the codes become
AA, AB…, which the service scores token by token using llama-server's `/tokenize` and one extra
request per shared prefix. Behind a proxy without `/tokenize` (LiteLLM), such questions fail
with a 422 instead of returning a distorted distribution.

An option missing from the returned top-N gets the lowest returned logprob, an upper bound on
its true value; raise `top_logprobs` if many options compete.
