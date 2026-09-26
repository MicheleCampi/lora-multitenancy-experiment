#!/usr/bin/env bash
# Fase 1 - venv isolato con vllm 0.30.0 e ninja; in parallelo, i download pinnati.
# I download girano in un venv separato: pip install vllm non deve poter
# cambiare huggingface_hub sotto un download in corso.
set -u
B="$(cd "$(dirname "$0")" && pwd)"
R="$HOME/lora-run"
mkdir -p "$R/logs"
fail() { echo "FASE1 FALLITA: $*"; exit 1; }

if ! python3 -m venv /tmp/venv-probe >/dev/null 2>&1; then
  echo "== installo python3-venv"
  sudo apt-get update -qq \
    && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv \
    || fail "installazione di python3-venv"
fi
rm -rf /tmp/venv-probe

echo "== venv dei download"
python3 -m venv "$R/venv-dl" || fail "creazione di venv-dl"
"$R/venv-dl/bin/pip" install -q --upgrade pip huggingface_hub || fail "huggingface_hub"
"$R/venv-dl/bin/python" "$B/download.py" > "$R/logs/download.log" 2>&1 &
DL=$!
echo "download avviati, PID $DL, log $R/logs/download.log"

echo "== venv del server e del generatore"
python3 -m venv "$R/venv" || fail "creazione di venv"
"$R/venv/bin/pip" install -q --upgrade pip || fail "aggiornamento di pip"
"$R/venv/bin/pip" install "vllm==0.30.0" ninja > "$R/logs/pip.log" 2>&1 \
  || { tail -30 "$R/logs/pip.log"; fail "pip install vllm==0.30.0 ninja"; }

echo "== attendo i download"
wait "$DL"
dlrc=$?
tail -14 "$R/logs/download.log"
[ "$dlrc" = 0 ] || fail "download (rc $dlrc)"

echo "== verifica del venv"
PATH="$R/venv/bin:$PATH" "$R/venv/bin/python" - <<'PY' || fail "verifica del venv"
import importlib.metadata as m
import shutil
import torch
v = m.version("vllm")
print("vllm", v, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu", torch.cuda.is_available(), "| ninja", shutil.which("ninja"))
assert v == "0.30.0", v
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.cuda.is_available(), "torch non vede la GPU"
assert shutil.which("ninja"), "ninja non nel PATH del venv"
PY
echo "FASE1 OK"
