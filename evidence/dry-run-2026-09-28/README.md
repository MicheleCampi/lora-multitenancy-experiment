# Second dry run, 2026-09-28

Files from the second dry run: one NVIDIA A10 rented from Lambda, vLLM
0.30.0, the base model and the eight adapters of `PROTOCOL.md`. The
amendments of 2026-09-29 at the end of `PROTOCOL.md` rest on them. This is
a dry run, not the campaign: its numbers fixed the campaign's design and
are not its result.

Each of the 35 files is identical to a file in the archive downloaded from
the node, `seconda-prova-node.tgz`, sha256
`3e989b041f5a01c02b99c515d2dd69d4b1d2451e765e8c369ccda3768969901b`:
`phase2.out` at the archive's top level, every other file at the same path
under `lora-run/`. The archive is not in this repository. Left out are the
benchmark's `load.json` files, which carry the generated text of every
request, and the server's log.

## What is here

- `pins.json`: the base model and the eight adapters `a1` to `a8`, in the
  order of the table in `PROTOCOL.md`, each with its full revision, and the
  two reserves.
- `phase2.out`: the server's start, printed by `harness/node/phase2.py`,
  with the two lines of vLLM's log that give the KV cache size.
- `results/<cell>/manifest.json`: written by `harness/run_cell.py` for each
  cell: configuration, commands as issued, timing, energy, containment.
- `results/<cell>/measure.json`: inferscope's measurement of the cell.
- `results/d8-idle.json`: inferscope for 60 s with the server idle.
- `results/metrics-final.txt`: vLLM's `/metrics`, saved from the server
  after the last cell; it is not written by the harness.
- `results/summary.json`: written by `harness/node/phase3.py`.

The cells: `warmup`, discarded by design; `d2-c64`, `d2-c96`, `d2-c128`
and `d2-c160`, N=8 uniform with 672 prompts at four concurrencies;
`d3-n1-r1` to `r4` and `d3-n8-r1` to `r4`, N=1 and N=8 uniform at
concurrency 128 with 336 prompts; `r30-a` and `r30-b`, the same seed twice
with 24 prompts, so that `r30-b` must record prefix-cache hits.

## Recomputing

    python3 harness/analyze.py evidence/dry-run-2026-09-28/results

prints the idle power, the verdict of every cell, cost per token by
configuration, the concurrency sweep, the preamble range and the H1
judgement, from the manifests and `d8-idle.json` alone.

The counts that amendment A8 reads from `/metrics`:

    cd evidence/dry-run-2026-09-28/results
    grep -c '^vllm:lora_requests_info{' metrics-final.txt
    grep -oE 'running_lora_adapters="[^"]*",waiting_lora_adapters="[^"]*"' \
        metrics-final.txt | awk -F'"' '{n++} $2 != $4 {d++} END {print n, d+0}'
    grep -oE 'model_name="[^"]*"' metrics-final.txt | sort | uniq -c

The first counts the label combinations the gauge recorded; the second
prints how many label pairs it read and how many of them differ; the third
counts the series carrying a `model_name` label, by value.
