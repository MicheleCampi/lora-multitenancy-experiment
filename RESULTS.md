# Results

The campaign of `PROTOCOL.md` ran on 2026-09-30 on one NVIDIA A10: 56
cells, all kept, none rerun and none missing. Its files are in
`evidence/campaign-2026-09-30/`, and every figure below comes from the
commands at the end.

## The answer

- **H1 is not falsified.** Serving eight adapters costs more per generated
  token than serving one: +7.05% energy net of idle at concurrency 128 and
  +5.96% at 64, both above the 5% threshold, with no band wider than 1.45%.
- **H2 is falsified at every level.** At two, four and eight adapters,
  sending 75% of the requests to one adapter changes the cost per token by
  +0.11% to +0.60% against an even split, inside the band of repetitions
  in each of the six judgements.

On this configuration, then, the per-adapter request counts that vLLM
computes and does not export (`PROTOCOL.md`, A8) would not have told two
pods with the same adapters apart by cost per token. What moved the cost
was how many adapters were served, not how the requests were spread among
them.

## H1: the number of adapters

Decided, as A1 and A2 fix, on energy per generated token net of idle, with
an idle power of 57.08 W: the mean of three measurements of 60 s, 57.08,
57.61 and 56.54 W, at the start, middle and end of the session.

| Concurrency | N=1, J/token | N=8, J/token | Margin | Bands N=1, N=8 | Judgement |
|---|---|---|---|---|---|
| 128 | 0.107749 | 0.115340 | +7.05% | 1.45%, 0.67% | not falsified |
| 64 | 0.155812 | 0.165093 | +5.96% | 0.50%, 0.62% | not falsified |

At 64 the margin clears the threshold by 0.96 points, and both bands are
under 1%, so the comparison resolves it. At 128 the second dry run had
measured +7.42% (A3).

The metric decides, as A1 said it would. On the same cells time per token
gives +6.11% at 128 and +6.05% at 64, and raw energy per token +2.90% and
+2.40%. Raw energy also counts the part of each window in which no request
is in flight, which does not depend on the adapters; judged on raw energy,
H1 would be falsified at both concurrencies.

Where the cost appears, against N=1 at the same concurrency, energy net of
idle. This is descriptive: no hypothesis was stated about it.

| Concurrency | N=2 | N=4 | N=8 |
|---|---|---|---|
| 128 | +0.05% | +2.21% | +7.05% |
| 64 | +0.64% | +1.01% | +5.96% |

Output throughput at N=8 is 5.80% lower than at N=1 at concurrency 128,
and 5.70% lower at 64.

## H2: the imbalance

d = (skewed - uniform) / uniform, falsified where |d| is smaller than the
larger of the two bands (A2), each level judged on its own.

| N | Concurrency | d | Bands uniform, skewed | Judgement |
|---|---|---|---|---|
| 2 | 128 | +0.52% | 0.60%, 0.73% | falsified |
| 2 | 64 | +0.11% | 0.95%, 1.75% | falsified |
| 4 | 128 | +0.60% | 0.72%, 0.61% | falsified |
| 4 | 64 | +0.46% | 1.02%, 2.01% | falsified |
| 8 | 128 | +0.32% | 0.67%, 0.43% | falsified |
| 8 | 64 | +0.13% | 0.62%, 0.51% | falsified |

The skewed lists give one adapter 75% of the requests, interleaved through
the list rather than in blocks (`harness/run_cell.py:94-125`): the volume
per adapter changes, not the timing of its requests.

## Latency, descriptive

`PROTOCOL.md:130` reads H2 first in the tails of itl, the gaps between
consecutive tokens. Skewed against uniform, from `harness/latency.py`:

- itl p99 differs by -0.10% to +0.13%, inside both bands at all six levels.
- itl p90 at concurrency 128 is higher by +1.35%, +1.29% and +1.11% at
  N=2, 4 and 8, each larger than both bands, which are at most 0.41%. That
  is under 0.8 ms on an itl p90 of 56 to 62 ms. At 64 it is higher by
  +0.30%, +0.81% and +0.39%, and only at N=8 is the difference larger than
  the bands.

No rule was fixed for latency before the campaign, so these are
observations, not judgements.

## The first cell

`r1-n1u-c128` was the first cell measured, after a warm-up of 24 prompts at
concurrency 8 and the first idle measurement. Its mean time to first token
was 3209.5 ms, against 2436.1 to 2447.4 ms in the three other repetitions
of the same configuration, cells 26, 37 and 48 of 56; its p99 was 9512.3 ms
against 7466.4 to 7480.5 ms. Its mean time per output token, 73.369 ms
against 72.467 to 72.633 ms, differs far less. It is why the band of N=1's
time per token at 128 is 4.79%, where no other configuration's exceeds
2.12%. These files do not establish the cause.

The rules keep the cell (A2, A7), and every judgement above includes it.
Leaving it out, an analysis chosen after seeing the data and reported only
to show its weight: the band of N=1's time per token at 128 falls to 0.22%,
the H1 margin at 128 rises from +7.05% to +7.33%, and N=2 uniform, which
with it reads 0.84% faster per token than N=1, reads 0.32% slower. No
judgement changes.

## What this does not show

- One NVIDIA A10, one base model, eight adapters of one rank and one set of
  target modules, vLLM 0.30.0 under the invariants of
  `PROTOCOL.md:162-219`, in one node session.
- Equal generative work: 256 tokens in and 256 out for every request (A6),
  so not tenants whose traffic differs in shape (`PROTOCOL.md:371-380`).
- Imbalance in volume, not in bursts (`PROTOCOL.md:407-411`).
- Two concurrencies, 128 and 64, four repetitions of each configuration.
- The GPU's energy, not the host's: the delta of the device's NVML counter
  over the window (inferscope at `acd21ec`,
  `crates/is-sysmon/src/gpu_nvidia.rs:179`, `201-202`).

## Recomputing

From the root of the repository:

    python3 harness/analyze.py evidence/campaign-2026-09-30/results

prints the idle power, the verdict of every cell, cost per token by
configuration, output throughput by configuration and the H1 and H2
judgements. The margins of the second table are its J/token means divided
by N=1's at the same concurrency.

    python3 harness/latency.py evidence/campaign-2026-09-30/results

prints the latency figures; with `--json` it also gives them cell by cell,
those of the first cell included.

    python3 harness/check_campaign.py evidence/campaign-2026-09-30/results harness/node/campaign-seeds.json

checks the fields the verdicts rest on. The figures without the first cell
are those of `analyze.py --json`'s rows for each cell, recomputed without
`r1-n1u-c128`.
