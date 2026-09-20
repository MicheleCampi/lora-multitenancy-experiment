# Protocol — what multi-tenant LoRA costs

Written 2026-09-20, before any data was collected and before a node was
booked. Every configuration fact below was read from vLLM at
`origin/main` `e378275a8f` (2026-09-20); the line references are to that
revision.

## Question

Serving N LoRA adapters on one GPU costs more per generated token than
serving one — and that cost depends on how unevenly traffic is spread
across the adapters, not only on how many there are.

The second half is the part worth measuring. That more adapters cost
more is expected; whether an 80/20 split costs differently from 50/50 at
the same N is not, and it is the question multi-tenancy actually poses.

## Hypotheses and what would falsify them

### H1 — the number of adapters costs
Cost per generated token at N=8 exceeds cost at N=1.

**Falsified if** the difference stays under 5%, with the band taken from
repetitions. Then multi-tenant LoRA is free within noise on this
hardware, which is a publishable result and closes the question.

### H2 — the imbalance costs, at fixed N
At fixed N, a skewed traffic distribution costs differently per token
than a uniform one.

**Falsified if** the two distributions differ by less than the
repetition band. Then only the count of active adapters matters, not how
load spreads across them — which would also explain why vLLM exports
adapter names and discards the per-adapter request counts it holds
(`vllm/v1/metrics/stats.py:643-644` computes them;
`vllm/v1/metrics/loggers.py:1100-1105` keeps only the keys).

H1 is the control without which H2 cannot be read.

## The threshold, fixed before the data

**5% on cost per token**, band from repetitions. Below 5% no deployment
decision would change, so calling a smaller difference an effect would
be noise dressed as a finding. Same discipline as the +3% threshold in
vllm-coldstart-operator's phase C and the 5-point threshold in
agentic-kv's adr0010-arrival.

A limit of this number, declared now rather than discovered later: a
threshold is only meaningful against the available margin, and that
margin is unknown here. Phase C of the operator ran a +3% test against a
margin of about 6% and said so. If multi-tenant LoRA turns out to cost
tens of percent, 5% is a slack test; if it costs three, 5% falsifies H1
by construction. The dry run measures the margin, and the threshold is
revisited once — before the campaign, never after seeing a cell.

## Invariants, and why each one is held

- **`enforce_eager`.** `vllm/config/lora.py:68` declares
  `specialize_active_lora`, whose docstring says separate CUDA graphs are
  captured "for different counts of active LoRAs (powers of 2 up to
  max_loras)". With graphs captured, cost per token would depend on which
  power of two covers the currently active adapters — a capture detail,
  not multi-tenancy. Eager mode captures no graphs at all, so the effect
  cannot arise however that option is gated.
- **`max_loras` constant across every cell, equal to max(N).** LoRA
  weights are preallocated per slot, not per loaded adapter:
  `vllm/lora/layers/base_linear.py:132-149` creates
  `torch.zeros(max_loras, 1, ...)` for both `lora_a_stacked` and
  `lora_b_stacked`. Holding it constant makes the memory footprint
  identical in every cell, so a difference between N=1 and N=8 cannot
  come from VRAM pressure. A cell at N=1 therefore carries seven empty
  slots — that is the control, not waste.
- **`max_cpu_loras` at its default.** `vllm/config/lora.py:117-118` sets
  it equal to `max_loras` when unset. With N never exceeding `max_loras`,
  the LRU eviction path never runs, so what is measured is multi-tenancy
  and not adapter swapping.
- **`max_lora_rank` fixed.** It is a dimension of the preallocated
  tensors (same lines as above); varying it would change cost for
  reasons orthogonal to the question.
- **Fixed target modules, one base model, adapters loaded before load
  begins.**

## Factors

| Factor | Levels | Serves |
|---|---|---|
| N, active adapters | 1, 2, 4, 8 | H1 |
| traffic distribution at fixed N | uniform, skewed | H2 |

Repetitions per cell: more than one, count fixed after the dry run.

## What the dry run must establish before the campaign

None of the following can be settled away from the node, and each one
changes the cell count or the hour budget:

1. whether `max_loras=8` fits in the A10's VRAM alongside the base
   model, read from the server at startup rather than computed;
2. the concurrency level at which the GPU is the bottleneck rather than
   the load generator;
3. the duration a cell needs for its counters to move enough to
   difference;
4. whether `vllm:lora_requests_info` is exposed and carries the adapters
   the run actually drives — it exists only when `lora_config` is not
   None (`vllm/v1/metrics/loggers.py:982`).

The dry run happens on a node that is paid for, so it is part of the
budget, not a preliminary outside it.
