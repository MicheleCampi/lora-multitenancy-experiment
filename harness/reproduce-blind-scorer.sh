#!/usr/bin/env bash
# Reproduces, on a CPU box in about a minute, what the routing chain
# cannot see. No GPU, no model weights, no adapters downloaded.
#
# What it shows:
#   1. eight adapters loaded, running_lora_adapters="" at rest — the
#      metric tracks requests, not residency (vllm-project/vllm#45325);
#   2. six requests on one adapter and one each on two others produce
#      three indistinguishable label sets — the load per adapter is
#      computed internally and dropped at export;
#   3. every label combination seen stays exposed, so a scraper finds
#      several series and nothing in them says which is current.
#
# The exact series count varies between runs: which requests overlap in
# time decides which combinations occur. Runs where two adapters are
# concurrent produce a combined label such as
# running_lora_adapters="a3,a1" — one value holding a comma-separated
# list, which is how the count per adapter is lost and also what breaks
# a naive label parser (see inferscope commit 05faba1). Two label
# values vary independently, running and waiting, so eight adapters
# admit up to 65535 distinct label combinations — an upper bound on the
# series a long-lived server can accumulate, not a count of what one
# run produces. This is the cardinality objection raised in vllm#45325,
# visible in under a minute.
#
# What it does NOT show: any cost. The simulator's latencies are
# declared, not computed. This reproduces the blindness, not its price.
set -euo pipefail

PORT="${PORT:-8010}"
SIM_DIR="${SIM_DIR:-/root/llm-d-inference-sim}"
BIN="$SIM_DIR/bin/llm-d-inference-sim"
N_HEAVY="${N_HEAVY:-6}"

if [[ ! -x "$BIN" ]]; then
  echo "simulator binary not found at $BIN" >&2
  echo "clone https://github.com/llm-d/llm-d-inference-sim and run: make build" >&2
  exit 1
fi

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "port ${PORT} is in use; set PORT=<free port>" >&2
  exit 1
fi

MODULES=()
for i in 1 2 3 4 5 6 7 8; do
  MODULES+=("{\"name\":\"a${i}\",\"path\":\"/x/a${i}\"}")
done

"$BIN" --model Qwen/Qwen2.5-7B-Instruct --port "$PORT" \
  --max-loras 8 --max-cpu-loras 8 --max-num-seqs 64 --seed 42 \
  --lora-modules "${MODULES[@]}" > /tmp/sim-repro.log 2>&1 &
SIM_PID=$!
trap 'kill "$SIM_PID" 2>/dev/null || true' EXIT

for _ in $(seq 20); do
  curl -sf "http://127.0.0.1:${PORT}/metrics" >/dev/null 2>&1 && break
  sleep 0.5
done

echo "== 1. eight adapters loaded, nothing running =="
curl -s "http://127.0.0.1:${PORT}/v1/models" \
  | python3 -c 'import sys,json;print("   loaded:", ", ".join(m["id"] for m in json.load(sys.stdin)["data"]))'
curl -s "http://127.0.0.1:${PORT}/metrics" | grep 'lora_requests_info{' | sed 's/^/   /'

echo
echo "== 2. ${N_HEAVY} requests on a1, one each on a2 and a3 =="
for _ in $(seq "$N_HEAVY"); do
  curl -s "http://127.0.0.1:${PORT}/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"a1","prompt":"hello","max_tokens":200}' >/dev/null &
done
for a in a2 a3; do
  curl -s "http://127.0.0.1:${PORT}/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${a}\",\"prompt\":\"hello\",\"max_tokens\":200}" >/dev/null &
done
sleep 1
curl -s "http://127.0.0.1:${PORT}/metrics" | grep 'lora_requests_info{' | sed 's/^/   /'
echo "   a1 carried ${N_HEAVY}x the load of a2 and a3. Nothing above says so."

# Not `wait`: a curl can outlive its response and hang the script.
# The requests have answered by now; give the gauge a moment to settle.
sleep 3
echo
echo "== 3. at rest: every combination seen is still exposed =="
curl -s "http://127.0.0.1:${PORT}/metrics" | grep 'lora_requests_info{' | sed 's/^/   /'
echo "   series: $(curl -s "http://127.0.0.1:${PORT}/metrics" | grep -c 'lora_requests_info{')"
