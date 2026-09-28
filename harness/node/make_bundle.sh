#!/usr/bin/env bash
# Genera il pacchetto piatto per il nodo da un commit di questo repository.
# Uso: harness/node/make_bundle.sh <cartella di destinazione> <binario inferscope>
# Il pacchetto corrisponde a un commit: il worktree deve essere pulito, e
# l'hash del commit va in BUNDLE_COMMIT. Il binario di inferscope non sta nel
# repository: viene accettato solo con lo sha256 pinnato. SHA256SUMS e' calcolato
# qui ed e' quello che phase0.sh verifica con sha256sum -c.
set -eu
INFERSCOPE_SHA256=5a18aae3d8fb4994d86ef6ba7d33bcab1e0272c4b86d87988295929c9cccdc69
fail() { echo "BUNDLE FALLITO: $*" >&2; exit 1; }

[ $# -eq 2 ] || fail "uso: $0 <cartella di destinazione> <binario inferscope>"
out=$1
bin=$2
repo=$(cd "$(dirname "$0")/../.." && pwd)

[ -z "$(git -C "$repo" status --porcelain)" ] || fail "worktree con modifiche: il pacchetto deve corrispondere a un commit"
[ ! -e "$out" ] || fail "$out esiste gia': un pacchetto non si sovrascrive"
got=$(sha256sum "$bin" | cut -d' ' -f1)
[ "$got" = "$INFERSCOPE_SHA256" ] || fail "sha256 di $bin = $got, atteso $INFERSCOPE_SHA256"

mkdir -p "$out"
cp "$repo/harness/run_cell.py" "$out/"
for f in download.py phase0.sh phase1.sh phase2.py phase3.py plan.py seeds.json \
         campaign.py campaign_plan.py campaign-seeds.json; do
  cp -p "$repo/harness/node/$f" "$out/"
done
cp -p "$bin" "$out/inferscope"
git -C "$repo" rev-parse HEAD > "$out/BUNDLE_COMMIT"
(cd "$out" && sha256sum BUNDLE_COMMIT download.py inferscope phase0.sh phase1.sh \
  phase2.py phase3.py plan.py run_cell.py seeds.json \
  campaign.py campaign_plan.py campaign-seeds.json > SHA256SUMS)
echo "pacchetto $out dal commit $(cat "$out/BUNDLE_COMMIT")"
cat "$out/SHA256SUMS"
