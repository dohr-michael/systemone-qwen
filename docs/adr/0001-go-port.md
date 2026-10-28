# ADR 0001: Port the service from Python to Go

- Status: accepted
- Date: 2026-10-28

## Context

The first version (v0.1.0, commit `44f1083`) was Python: FastAPI, uvicorn, httpx, pydantic and
PyYAML on `python:3.12-slim`. The service is thin glue (HTTP and JSON in, prompts out to
llama-server, probabilities back), about 600 lines. Its image weighed 187 MB, and scanners
report a steady stream of findings against such images, most of them from the Debian packages
of the base image rather than from the service's own code.

Rust was considered. It handles the hardest part of the port better (`serde_json` with
`preserve_order` and `arbitrary_precision`), but a usual HTTP stack pulls 100+ crates and
multi-arch static builds need extra CI tooling, against the goal of fewer dependencies.

## Decision

Rewrite the service in Go, keeping the wire format, configuration and behaviour:

- **Dependencies**: Go's standard library (`net/http` server and client, `encoding/json`,
  `log/slog`) plus `go.yaml.in/yaml/v3` for the registry. Configuration in JSON would remove
  that last one, at the cost of comments in the registry file.
- **Image**: `FROM scratch` with the static binary and root certificates, running as UID 10001:
  7.5 MB, no shell, no OS packages. Multi-arch by cross-compiling, without QEMU.
- **Parity**: the prompt is part of the benchmarked model. `ParseJSON` keeps what Python's
  `json` keeps (key order, a repeated key's first position and last value, number literals), and
  `PythonJSON` renders as `json.dumps(ensure_ascii=False, separators=(",", ":"))` (Python's float
  repr, no `<>&` escaping). A generic ordered map was rejected for this: tested on a chat-log
  state, `cornelk/orderedmap` reordered objects nested in arrays, rounded large integers and
  escaped `<>&`.
- **Proof**: `testdata/parity.json` was recorded from the Python service against a deterministic
  fake llama-server (`testdata/generate_parity.py`); the Go test replays it and requires
  byte-identical prompts, identical backend requests, and the same answers in the same key order.
  Deliberate changes to the float formatting or to debias make it fail.

Two fixes from the v0.1.0 review are included, outside the benchmarked path:

1. Beyond 26 options, every option gets a two-letter code. Mixing A..Z with AA.. made `A` an
   upper bound of every `A?` code, favouring the first 26 options.
2. An option missing from an incomplete top-N (fewer alternatives than requested) is a 502, not
   the floor: a backend ignoring `logprobs` made every option equal, silently.

Small contract changes: validation errors use the service's `{"error": ...}` body instead of
FastAPI's `{"detail": [...]}`; bodies over 16 MB get a 413; nesting is capped at 512 levels.

## Consequences

- One static binary to patch: `govulncheck` runs in CI, and a Go security release means a rebuild.
- Validation that pydantic did declaratively is hand-written in `schema.go`.
- A JevBench run against the real llama-server must confirm 0.827 before tagging a release.
- No shell in the image: debugging goes through `/health`, `/ready` and the request logs.
