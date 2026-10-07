#!/usr/bin/env bash
# Runs JevBench's official harness (231 public tasks) against a System One endpoint.
#   benchmark/jevbench.sh <label> <endpoint> [model]      e.g. local http://127.0.0.1:8000 qwen-decision
# Needs python3 and git; results land in benchmark/results/<label>/ and are never overwritten.
set -euo pipefail
cd "$(dirname "$0")"
label=${1:?usage: benchmark/jevbench.sh <label> <endpoint> [model]}
endpoint=${2:?usage: benchmark/jevbench.sh <label> <endpoint> [model]}
model=${3:-qwen-decision}

bench=${JEVBENCH_DIR:-$HOME/.cache/jevbench}
[[ -d "$bench" ]] || git clone -q https://github.com/fstandhartinger/jevbench.git "$bench"
tasks="$bench/datasets/public/all.jsonl"
cat "$bench"/datasets/public/{original,easy,hard}.jsonl > "$tasks"

out="$PWD/results/$label"
[[ ! -e "$out" ]] || { echo "$out exists: runs are never overwritten" >&2; exit 1; }
mkdir -p "$PWD/results"

# The harness uses urllib; python.org builds ship without root certificates.
if [[ -z "${SSL_CERT_FILE:-}" ]] && python3 -c 'import certifi' 2>/dev/null; then
  export SSL_CERT_FILE="$(python3 -m certifi)"
fi
ledger="$PWD/results/ledger.jsonl"
cd "$bench"
python3 -m jevbench.cli run --tasks "$tasks" --adapter typesafe --endpoint "$endpoint" --model "$model" \
  --key-env "${JEVBENCH_KEY_ENV:-}" --reserve-usd 0 --cost-basis local_open_weights --run-label "$label" \
  --results "$out/results.jsonl" --raw-dir "$out/raw" --ledger "$ledger" --manifest "$out/manifest.json"
python3 -m jevbench.cli summarize --tasks "$tasks" --results "$out/results.jsonl" --public-export "$out/summary.json" \
  >/dev/null
python3 - "$out/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"accuracy {d['accuracy']:.3f}  brier {d['brier_mean']:.3f}  ece {d['ece']['ece']:.3f}  p50 {d['latency']['p50_s']:.2f}s")
PY
