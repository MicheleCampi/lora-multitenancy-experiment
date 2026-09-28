#!/usr/bin/env bash
# Fase 0 - controlli d'ambiente sul nodo. Non installa nulla.
# Esce con codice diverso da 0 al primo controllo che fallisce.
set -u
B="$(cd "$(dirname "$0")" && pwd)"
fail() { echo "FASE0 FALLITA: $*"; exit 1; }

echo "== integrita' del pacchetto"
(cd "$B" && sha256sum -c --quiet SHA256SUMS) || fail "SHA256SUMS non corrisponde"
echo "ok"

echo "== GPU"
q=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) \
  || fail "nvidia-smi non risponde"
echo "$q"
[ "$(printf '%s\n' "$q" | grep -c .)" = 1 ] || fail "attesa una sola GPU"
name=$(printf '%s' "$q" | cut -d, -f1 | xargs)
drv=$(printf '%s' "$q" | cut -d, -f2 | xargs)
[ "$name" = "NVIDIA A10" ] || fail "GPU attesa NVIDIA A10, trovata: $name"
[ "${drv%%.*}" -ge 580 ] || fail "driver $drv < 580: torch 2.13.0 usa CUDA 13.0"

echo "== memoria della GPU (ECC)"
# Il 28/9 un A10 con 373452 errori DRAM non correggibili nella sua storia ha
# superato questa fase e ha fatto fallire vllm serve al primo forward ("CUDA
# error: uncorrectable ECC error encountered"). Si legge nvidia-smi -q -d ECC,
# nel formato osservato con il driver 580.105.08: si ferma se l'ECC non e'
# abilitato, se un contatore non correggibile (Volatile o Aggregate) e' > 0, se
# una riparazione e' in sospeso o una soglia superata, o se il formato non e'
# riconosciuto. Gli errori correggibili si riportano soltanto.
ecc=$(nvidia-smi -q -d ECC) || fail "nvidia-smi -q -d ECC non risponde"
printf '%s\n' "$ecc" | awk -F' : ' '
  { key = $1; sub(/^ +/, "", key); sub(/ +$/, "", key); val = $2 }
  NF == 1 && key != "" { sec = key; next }
  sec == "ECC Mode" && key == "Current" { mode = val }
  (sec == "Volatile" || sec == "Aggregate") && key ~ /Uncorrectable/ {
    unc[sec] += val; if (key == "DRAM Uncorrectable") seen[sec] = 1
    if (val + 0 > 0) bad = bad " " sec ": " key " = " val ";"
  }
  (sec == "Volatile" || sec == "Aggregate") && key ~ /Correctable/ && key !~ /Uncorrectable/ {
    cor[sec] += val
  }
  (key ~ /Repair Pending$/ || key ~ /Threshold Exceeded$/) && val != "No" {
    bad = bad " " key " = " val ";"
  }
  END {
    printf "ECC %s | non correggibili: volatile %d, aggregate %d | correggibili: volatile %d, aggregate %d\n",
      mode, unc["Volatile"], unc["Aggregate"], cor["Volatile"], cor["Aggregate"]
    if (mode != "Enabled") { print "ECC non abilitato: i contatori non provano nulla"; exit 1 }
    if (!seen["Volatile"] || !seen["Aggregate"]) { print "formato ECC non riconosciuto"; exit 1 }
    if (bad != "") { print "errori:" bad; exit 1 }
  }' || fail "memoria della GPU non sana o non verificabile"

echo "== sistema"
. /etc/os-release
echo "$PRETTY_NAME"
glibc=$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')
echo "glibc $glibc"
python3 -c 'import sys; sys.exit(0 if tuple(map(int, sys.argv[1].split("."))) >= (2, 34) else 1)' "$glibc" \
  || fail "glibc $glibc < 2.34 (richiesta dal binario di inferscope)"
pyv=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
echo "python3 $pyv"
python3 -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)' \
  || fail "python3 $pyv fuori da >=3.10,<3.15 (requires_python di vllm 0.30.0)"
if python3 -m venv /tmp/venv-probe >/dev/null 2>&1; then
  echo "venv: disponibile"
else
  echo "venv: ASSENTE, la fase 1 installa python3-venv"
fi
rm -rf /tmp/venv-probe

echo "== disco"
df -h "$HOME" | tail -1
avail=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
[ "$avail" -ge 60 ] || fail "spazio libero ${avail}G < 60G"

echo "== variabili VLLM_*"
if env | grep -q '^VLLM_'; then
  env | grep '^VLLM_' | cut -d= -f1
  fail "variabili VLLM_* impostate"
fi
echo "(nessuna)"

echo "== inferscope"
# --version e --help stampano su stderr, con il prefisso "inferscope: ";
# --version esce con codice 2 (stesso binario: codice 2 sul nodo, stderr su
# optim-dev, 26/9). Si controlla il testo, catturando stderr, non il codice.
v=$("$B/inferscope" --version 2>&1)
echo "$v"
case "$v" in
  *"inferscope 0.5.0"*) ;;
  *) fail "inferscope non eseguibile o versione diversa da 0.5.0: $v" ;;
esac
"$B/inferscope" --help 2>&1 | grep -q -- '--gpu' || fail "inferscope senza --gpu"

echo "== token Hugging Face"
if [ -n "${HF_TOKEN:-}" ]; then echo "presente (non stampato)"; else echo "assente: download anonimo"; fi

echo "FASE0 OK | gpu=$name | driver=$drv | glibc=$glibc | python=$pyv | disco=${avail}G"
