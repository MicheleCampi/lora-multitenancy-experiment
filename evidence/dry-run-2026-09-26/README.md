# First dry run, 2026-09-26

Files from the first dry run: one NVIDIA A10 rented from Lambda, vLLM
0.30.0, the base model and the eight adapters of `PROTOCOL.md`. None of the
run's files records the name of the GPU, so the GPU and the provider are
stated here rather than read from them. This run found defects in the harness
that later commits fix, and their messages cite it. It is a dry run, not the
campaign, and the amendments of 2026-09-29 take their figures from the
second dry run, in `dry-run-2026-09-28/`.

Each of the 33 files copied from the node is identical to a file in the
archive downloaded from it, `prova-generale-node.tgz`, sha256
`4d6209b900bc8c2ccff5ac06ac352cf0e2868dd131096df6268689ee9e44f750`:
`phase2.out` and `phase2-tentativo2.out` at the archive's top level, every
other file at the same path under `lora-run/`. The archive is not in this
repository. Left out are:

- the benchmark's `load.json` files, which carry the generated text of
  every request; the two fields the evidence needs are extracted;
- the server log `logs/serve.log`, and the logs of the downloads and of pip;
- the standard output and error saved for each cell;
- `phase1.out`, `phase3.out`, `session.json`, `session-tentativo1.json`,
  `server.pid` and `gpu-limits.csv`.

## What is here

- `pins.json`: the base model and the eight adapters `a1` to `a8`, each with
  its full revision, and the two reserves.
- `phase2.out`: the start of the server that ran every cell, printed by
  `harness/node/phase2.py`, with the two lines of vLLM's log that give the KV
  cache size. Its PID, 5087, is the `server_pid` of every manifest.
- `phase2-tentativo2.out` and `logs/serve-tentativo2.log`: an earlier start
  that failed (below).
- `results/<cell>/manifest.json`: written by `harness/run_cell.py` for each
  cell: configuration, commands as issued, timing, energy, containment.
- `results/<cell>/measure.json`: inferscope's report on the cell, as
  `harness/run_cell.py` saves it.
- `results/d8-idle.json`: inferscope with the server idle.
- `results/metrics-final.txt`: vLLM's `/metrics`, saved from the server; it
  is not written by the harness.
- `results/summary.json`: written by `harness/node/phase3.py`, with the
  verdict of every cell.
- `load-extract/<cell>.json`: the lists `input_lens` and `errors` of the
  cell's `load.json`, unchanged and in order, written from the archive by
  `extract_load.py`.
- `regen.out`: the output of `harness/node/regen.py` (below).

## The cells

All 13 ran on the same server, with 256 tokens in and 256 out:

| cells | adapters | concurrency | prompts | window | seeds |
|---|---|---|---|---|---|
| `warmup` | `a1` to `a8` | 8 | 24 | 120 s | 1001 |
| `d2-c64`, `d2-c32`, `d2-c16`, `d2-c8` | `a1` to `a8` | 64, 32, 16, 8 | 168 | 180, 228, 372, 647 s | 1002 to 1005 |
| `d3-n1-r1`, `r2`, `r3` | `a1` | 64 | 168 | 124 s | 1006, 1008, 1010 |
| `d3-n8-r1`, `r2`, `r3` | `a1` to `a8` | 64 | 168 | 124 s | 1007, 1009, 1011 |
| `r30-a`, `r30-b` | `a1` to `a8` | 8 | 24 | 120 s | 777, both |

Of the 1,752 requests declared, 1,751 completed. `summary.json` discards
three cells:

- `d3-n1-r3` recorded 240 prefix-cache hits; `regen.out` shows that its
  request 140 has the same first 256 tokens as its request 114, both served
  by `a1`.
- `d3-n8-r3` completed 167 requests: its request 36 was 257 tokens long and
  failed with `Bad Request`.
- `r30-b`, sent the seed `r30-a` had just used, recorded 5,760 hits.

## The failed start

The names containing `tentativo2`, Italian for "attempt 2", are those given
on the node. That start's server, PID 4962, stopped on its port:
`OSError: [Errno 98] Address already in use`, line 40 of
`logs/serve-tentativo2.log`. `phase2.py` nonetheless reported it ready in
0 s (`phase2-tentativo2.out`, line 3). The last line of the log, after
13,103 NUL bytes, was written by another server, PID 4385, answering a
request for `/v1/models`, the endpoint `phase2.py` polls to decide the
server is ready (line 99).

## Recomputing

`harness/analyze.py` does not read these manifests: they have no
`prompt_lengths` field, which later versions of `harness/run_cell.py` write
and which the tool reads at its line 63.

`regen.out` is recomputed from the published files alone. `regen.py` reads
each cell's `load.json` for `input_lens` only (line 43), so the extracts
stand in for it:

    D=$(mktemp -d)
    cp -r evidence/dry-run-2026-09-26/results "$D"/
    for f in evidence/dry-run-2026-09-26/load-extract/*.json; do
        cp "$f" "$D"/results/$(basename "$f" .json)/load.json
    done
    python3 harness/node/regen.py "$D"/results \
        | grep -vE '^(INFO|WARNING) [0-9]{2}-[0-9]{2} [0-9:]{8} \[' \
        | diff - evidence/dry-run-2026-09-26/regen.out

It needs vLLM 0.30.0 and the base model's tokenizer in the local Hugging
Face cache, since it runs offline (line 14). vLLM writes its log to
standard output by default (`vllm/envs.py:836`), which is why its lines are
filtered out before the comparison; `diff` prints nothing when the output
matches.

The script regenerates each cell's prompts with the benchmark's own code
and prints in Italian: per cell, `seme` is the seed, `richieste` the number
of requests, `fedelta' OK` that the regenerated prompt lengths equal
`input_lens`, `adattatori` the adapter list. An indented line
`cell[i] adapter: k blocchi = n token in comune con other[j]` says that
request i of the cell shares its first k blocks of 16 tokens, n tokens,
with request j of the same or an earlier cell, served by the same adapter.
`FINE` is the end.
