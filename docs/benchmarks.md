# Benchmarks

Measured in October 2026 with [JevBench](https://github.com/fstandhartinger/jevbench)'s own
harness (adapter `typesafe`, 231 public tasks: 72 original, 48 easy, 111 hard; 74 noul,
139 choice, 18 score), plus a smaller in-house yes/no set in English and French. With 231
tasks one standard deviation is about ±2.3 accuracy points, and repeated runs against the same
llama-server differ by about one task. Reproduce with [`benchmark/jevbench.sh`](../benchmark/jevbench.sh).

## Models on JevBench

The same method for every open model: lettered options, answer read from first-token logprobs,
no reasoning, `prefixed` noul labels, no debias.

| System | Served by | All | Hard | Noul | Choice | Score | Brier ↓ | ECE ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 26B-A4B | MLX 4-bit (Mac) | 0.866 | 0.739 | 0.851 | 0.871 | 0.889 | 0.237 | 0.111 |
| Jev 1.13 (TypeSafe) | OpenRouter | 0.857 | 0.712 | 0.838 | 0.878 | 0.778 | 0.177 | 0.030 |
| DeepSeek V4 Flash | API (chat logprobs) | 0.844 | 0.712 | 0.838 | 0.849 | 0.833 | 0.266 | 0.118 |
| **Qwen3.5-9B** | **llama-server (ROCm)** | **0.827** | **0.667** | **0.838** | **0.827** | **0.778** | **0.268** | **0.062** |
| Qwen3.8-27B | MLX 4-bit (Mac) | 0.827 | 0.676 | 0.770 | 0.863 | 0.778 | 0.241 | 0.059 |
| Qwen3.6-35B-A3B | MLX 4-bit (Mac) | 0.823 | 0.694 | 0.770 | 0.863 | 0.722 | 0.255 | 0.082 |
| Gemma 4 E4B | MLX 4-bit (Mac) | 0.745 | 0.523 | 0.716 | 0.755 | 0.778 | 0.466 | 0.214 |

Jev scores about 200/231 on the official leaderboard; 198/231 here.

**This service** (v0.1.0, default settings, calibration off) against the same llama-server
reproduces the lab adapter it was ported from exactly: 0.827 overall, identical on every tier
and question type, Brier 0.268, ECE 0.062, p50 0.35 s.

## Qwen3.5-9B settings

| Variant | All | Hard | Brier | ECE | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `think_tokens: 512` | **0.853** | 0.694 | **0.205** | 0.066 | 21.8 s |
| `noul_labels: plain` | 0.831 | 0.667 | 0.252 | 0.050 | 0.36 s |
| `noul_labels: criteria` | 0.827 | 0.667 | 0.261 | 0.059 | 0.36 s |
| `debias: true` | 0.827 | 0.667 | 0.263 | 0.067 | 0.54 s |
| **default** (`prefixed`) | **0.827** | 0.667 | 0.268 | 0.062 | 0.36 s |
| choice options without keys | 0.810 | 0.640 | 0.295 | 0.072 | 0.36 s |
| verbose system prompt | 0.797 | 0.613 | 0.287 | 0.065 | 0.36 s |

Latency is end to end through LiteLLM and HTTPS to a Strix Halo llama-server, one request at a time.

## Calibration

Temperature scaling never changes accuracy. T was fitted by NLL on half the task groups and
scored on the other half (paraphrase groups never straddle the split).

| System | Fitted T | Brier (T=1 → T) | ECE (T=1 → T) |
| --- | ---: | --- | --- |
| Qwen3.5-9B | 1.4–1.55 | 0.268 → 0.263 | 0.062 → 0.054 |
| Qwen3.5-9B, thinking 512 | 1.6–1.9 | 0.205 → 0.199 | 0.066 → 0.050 |
| Gemma 4 26B-A4B | 3.35–3.85 | 0.237 → 0.184 | 0.111 → 0.040 |
| Jev 1.13 | 1.05–1.1 | 0.177 → 0.176 | 0.028 → 0.035 |

## Yes/no in English and French

40 hard yes/no cases (sarcasm, hypotheticals, reported speech, negation scope, severity),
translated: message in French with English questions, and everything in French. Accuracy with
criteria, the input Jev receives:

| System | English | FR message | All French |
| --- | ---: | ---: | ---: |
| Gemma 4 26B-A4B | 0.90 | 0.88 | 0.85 |
| Jev 1.13 | 0.93 | 0.80 | 0.80 |
| Qwen3.6-35B-A3B | 0.82 | 0.75 | 0.72 |
| **Qwen3.5-9B** | **0.85** | **0.70** | **0.68** |

Qwen3.5-9B matches the larger models in English and loses about 15 points in French. If most
traffic is French, a larger or Gemma-family backend is worth the switch (Gemma needs its own
prompt template; this service is Qwen-specific).
