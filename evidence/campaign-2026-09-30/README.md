# Campaign, 2026-09-30

The campaign of `PROTOCOL.md`: one NVIDIA A10 rented from Lambda, vLLM
0.30.0, the base model and the eight adapters of `PROTOCOL.md`, and the 56
cells of `harness/node/campaign_plan.py`, run by `harness/node/campaign.py`
in one node session. The rules it is judged by are those of the amendments
of 2026-09-29 at the end of `PROTOCOL.md`.

Of the files here, 123 are identical to a file in the archive downloaded
from the node, `campagna-node.tgz`, sha256
`dcbebcef6c3ac6064da930c4c846c1946c4d338d3c7391c0a87d160906883d2e`:
`phase0.out`, `phase2.out` and `campaign.out` at the archive's top level,
every other one at the same path under `lora-run/`. The archive is not in
this repository. The 57 `load-extract.json.gz` files are derived from it by
`extract_load.py`, which rewrites them byte for byte.

Left out of the archive: `phase1.out`, the installation; `lora-run/logs/`,
the logs of the download, of pip and of the server; `session.json` and
`server.pid`; `results/<cell>.stdout.txt` and `.stderr.txt`, and the stderr
of the idle measurements; and in every cell `load.json`, `load.stdout.txt`,
`load.stderr.txt` and `measure.stderr.txt`. `load.json` carries the
generated text of every request; everything else in it is in
`load-extract.json.gz`.

## What is here

- `pins.json`: the base model and the eight adapters `a1` to `a8`, each
  with its full revision, and the two reserves; the same content as
  `evidence/dry-run-2026-09-28/pins.json`.
- `phase0.out`: the node's checks, printed by `harness/node/phase0.sh`,
  among them the GPU's memory errors.
- `phase2.out`: the server's start, printed by `harness/node/phase2.py`.
- `campaign.out`: printed by `campaign.py`, one line per cell and per idle
  measurement in the order they ran, then the campaign's own summary.
- `results/gpu.csv`: the GPU's name, driver and memory per `nvidia-smi`,
  written by `campaign.py` before the warm-up.
- `results/<cell>/manifest.json`: written by `harness/run_cell.py` for each
  cell: configuration, commands as issued, timing, energy, containment.
- `results/<cell>/measure.json`: inferscope's measurement of the cell.
- `results/<cell>/load-extract.json.gz`: the benchmark's `load.json`
  without `generated_texts`: the per-request lists and the benchmark's own
  summary figures.
- `results/d8-idle-start.json`, `d8-idle-middle.json`, `d8-idle-end.json`:
  inferscope for 60 s with the server idle, after the warm-up, after the
  second round and after the last cell.
- `results/summary.json`: written by `campaign.py` after every cell.
- `extract_load.py`: writes the extracts from the archive.

The cells: `warmup`, 24 prompts over the eight adapters at concurrency 8,
discarded by design; then `r<round>-<configuration>-c<concurrency>`, four
rounds of the seven configurations at concurrency 128 and 64, 336 prompts
each. `n1u`, `n2u`, `n4u` and `n8u` spread the requests evenly over 1, 2, 4
and 8 adapters; `n2s`, `n4s` and `n8s` give one adapter 75% of them
(`harness/node/campaign_plan.py:34-42`). Each round runs the fourteen
combinations rotated three positions from the round before (lines 46-50).

## Recomputing

    python3 harness/analyze.py evidence/campaign-2026-09-30/results

prints the idle power, the verdict of every cell, cost per token by
configuration and the H1 and H2 judgements, from the manifests and the
idle measurements alone.

    python3 harness/latency.py evidence/campaign-2026-09-30/results

prints ttft, tpot, itl and e2el by configuration, from the extracts, after
checking every cell against the benchmark's own mean, median and p99 of
ttft, tpot and itl.

    python3 harness/check_campaign.py evidence/campaign-2026-09-30/results harness/node/campaign-seeds.json

checks fields the verdicts rest on that `analyze.py` does not check: that
the energy comes from the GPU's counter and is above zero, that inferscope
was given the cell's window, that the prefix-cache counter was read, that
the requests generated completed x 256 tokens in all, and that each cell
ran the seed and the list of adapters `harness/node/seedcheck.py` checked.

With the archive, the extracts are rewritten byte for byte by

    python3 evidence/campaign-2026-09-30/extract_load.py <archive>/lora-run/results <directory>
