#!/usr/bin/env bash
# Phase 0 - environment checks on the node. Installs nothing.
# Exits non-zero at the first check that fails.
set -u
B="$(cd "$(dirname "$0")" && pwd)"
fail() { echo "PHASE0 FAILED: $*"; exit 1; }

echo "== bundle integrity"
(cd "$B" && sha256sum -c --quiet SHA256SUMS) || fail "SHA256SUMS does not match"
echo "ok"

echo "== GPU"
q=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) \
  || fail "nvidia-smi does not answer"
echo "$q"
[ "$(printf '%s\n' "$q" | grep -c .)" = 1 ] || fail "expected a single GPU"
name=$(printf '%s' "$q" | cut -d, -f1 | xargs)
drv=$(printf '%s' "$q" | cut -d, -f2 | xargs)
[ "$name" = "NVIDIA A10" ] || fail "expected GPU NVIDIA A10, found: $name"
[ "${drv%%.*}" -ge 580 ] || fail "driver $drv < 580: torch 2.13.0 uses CUDA 13.0"

echo "== GPU memory (ECC)"
# On 2026-09-28 an A10 with 373452 uncorrectable DRAM errors in its history
# passed this phase and made vllm serve fail at its first forward ("CUDA
# error: uncorrectable ECC error encountered"). It reads nvidia-smi -q -d ECC,
# in the format seen with driver 580.105.08, and stops if ECC is not enabled,
# if an uncorrectable counter (Volatile or Aggregate) is > 0, if a repair is
# pending or a threshold exceeded, or if the format is not recognised.
# Correctable errors are only reported.
ecc=$(nvidia-smi -q -d ECC) || fail "nvidia-smi -q -d ECC does not answer"
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
    printf "ECC %s | uncorrectable: volatile %d, aggregate %d | correctable: volatile %d, aggregate %d\n",
      mode, unc["Volatile"], unc["Aggregate"], cor["Volatile"], cor["Aggregate"]
    if (mode != "Enabled") { print "ECC not enabled: the counters prove nothing"; exit 1 }
    if (!seen["Volatile"] || !seen["Aggregate"]) { print "ECC format not recognised"; exit 1 }
    if (bad != "") { print "errors:" bad; exit 1 }
  }' || fail "GPU memory not healthy or not verifiable"

echo "== system"
. /etc/os-release
echo "$PRETTY_NAME"
glibc=$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$')
echo "glibc $glibc"
python3 -c 'import sys; sys.exit(0 if tuple(map(int, sys.argv[1].split("."))) >= (2, 34) else 1)' "$glibc" \
  || fail "glibc $glibc < 2.34 (required by the inferscope binary)"
pyv=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
echo "python3 $pyv"
python3 -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else 1)' \
  || fail "python3 $pyv outside >=3.10,<3.15 (requires_python of vllm 0.30.0)"
if python3 -m venv /tmp/venv-probe >/dev/null 2>&1; then
  echo "venv: available"
else
  echo "venv: MISSING, phase 1 installs python3-venv"
fi
rm -rf /tmp/venv-probe

echo "== disk"
df -h "$HOME" | tail -1
avail=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc 0-9)
[ "$avail" -ge 60 ] || fail "free space ${avail}G < 60G"

echo "== VLLM_* variables"
if env | grep -q '^VLLM_'; then
  env | grep '^VLLM_' | cut -d= -f1
  fail "VLLM_* variables set"
fi
echo "(none)"

echo "== inferscope"
# --version and --help print on stderr, prefixed "inferscope: ", and exit
# with code 2 (crates/inferscope/src/main.rs:36-42 at acd21ec, the commit the
# pinned binary is built from). The text is checked, stderr included.
v=$("$B/inferscope" --version 2>&1)
echo "$v"
case "$v" in
  *"inferscope 0.5.0"*) ;;
  *) fail "inferscope not executable or not version 0.5.0: $v" ;;
esac
"$B/inferscope" --help 2>&1 | grep -q -- '--gpu' || fail "inferscope without --gpu"

echo "== Hugging Face token"
if [ -n "${HF_TOKEN:-}" ]; then echo "present (not printed)"; else echo "absent: anonymous download"; fi

echo "PHASE0 OK | gpu=$name | driver=$drv | glibc=$glibc | python=$pyv | disk=${avail}G"
