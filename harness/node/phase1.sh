#!/usr/bin/env bash
# Phase 1 - isolated venv with vllm 0.30.0 and ninja; in parallel, the pinned downloads.
# The downloads run in a separate venv: pip install vllm must not be able to
# change huggingface_hub under a download in progress.
set -u
B="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/lora-run"
mkdir -p "$R/logs"
fail() { echo "PHASE1 FAILED: $*"; exit 1; }

if ! python3 -m venv /tmp/venv-probe >/dev/null 2>&1; then
  echo "== installing python3-venv"
  sudo apt-get update -qq \
    && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv \
    || fail "installing python3-venv"
fi
rm -rf /tmp/venv-probe

echo "== the downloads' venv"
python3 -m venv "$R/venv-dl" || fail "creating venv-dl"
"$R/venv-dl/bin/pip" install -q --upgrade pip huggingface_hub || fail "huggingface_hub"
"$R/venv-dl/bin/python" "$B/download.py" > "$R/logs/download.log" 2>&1 &
DL=$!
echo "downloads started, PID $DL, log $R/logs/download.log"

echo "== the venv of the server and the load generator"
python3 -m venv "$R/venv" || fail "creating venv"
"$R/venv/bin/pip" install -q --upgrade pip || fail "upgrading pip"
"$R/venv/bin/pip" install "vllm==0.30.0" ninja > "$R/logs/pip.log" 2>&1 \
  || { tail -30 "$R/logs/pip.log"; fail "pip install vllm==0.30.0 ninja"; }

echo "== waiting for the downloads"
wait "$DL"
dlrc=$?
tail -14 "$R/logs/download.log"
[ "$dlrc" = 0 ] || fail "download (rc $dlrc)"

echo "== checking the venv"
PATH="$R/venv/bin:$PATH" "$R/venv/bin/python" - <<'PY' || fail "checking the venv"
import importlib.metadata as m
import shutil
import torch
v = m.version("vllm")
print("vllm", v, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu", torch.cuda.is_available(), "| ninja", shutil.which("ninja"))
assert v == "0.30.0", v
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.cuda.is_available(), "torch does not see the GPU"
assert shutil.which("ninja"), "ninja not in the venv's PATH"
PY
echo "PHASE1 OK"
