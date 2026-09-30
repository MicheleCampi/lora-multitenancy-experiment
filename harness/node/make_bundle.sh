#!/usr/bin/env bash
# Builds the flat bundle for the node from a commit of this repository.
# Usage: harness/node/make_bundle.sh <destination directory> <inferscope binary>
# The bundle corresponds to a commit: the worktree must be clean, and the
# commit's hash goes into BUNDLE_COMMIT. The inferscope binary is not in the
# repository: it is accepted only with the pinned sha256. SHA256SUMS is computed
# here, and it is what phase0.sh checks with sha256sum -c.
set -eu
INFERSCOPE_SHA256=5a18aae3d8fb4994d86ef6ba7d33bcab1e0272c4b86d87988295929c9cccdc69
fail() { echo "BUNDLE FAILED: $*" >&2; exit 1; }

[ $# -eq 2 ] || fail "usage: $0 <destination directory> <inferscope binary>"
out=$1
bin=$2
repo=$(cd "$(dirname "$0")/../.." && pwd)

[ -z "$(git -C "$repo" status --porcelain)" ] || fail "worktree has changes: the bundle must correspond to a commit"
[ ! -e "$out" ] || fail "$out already exists: a bundle is not overwritten"
got=$(sha256sum "$bin" | cut -d' ' -f1)
[ "$got" = "$INFERSCOPE_SHA256" ] || fail "sha256 of $bin = $got, expected $INFERSCOPE_SHA256"

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
echo "bundle $out from commit $(cat "$out/BUNDLE_COMMIT")"
cat "$out/SHA256SUMS"
