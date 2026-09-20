# Protocol — what multi-tenant LoRA costs

Written 2026-09-20, before any data was collected and before a node was
booked. Every configuration fact below was read from vLLM at
`origin/main` `27757dde02` (2026-09-20); the line references are to that
revision.

## What this measures, and for whom

A routing component in production scores two pods identically when one
serves an adapter with 90% of its traffic and the other with 5%. It has
no way to tell them apart, because the number that would is computed and
then dropped three times along the way.

Read at the revisions below, on 2026-09-20:

| Where | What happens | Line |
|---|---|---|
| vLLM computes | `running_lora_adapters[name] = len(stats.running)` | `vllm/v1/metrics/stats.py:644` |
| vLLM exports | `",".join(...keys())` — the counts are dropped | `vllm/v1/metrics/loggers.py:1100` |
| llm-d-router receives | `m[trimmed] = 0` into a `map[string]int` | `extractor.go:312` |
| llm-d-router decides | `_, active := m.ActiveModels[request.TargetModel]` | `lora_affinity.go:89` |

vLLM at `27757dde02`, llm-d-router at `5367d054`.

The type survives the whole way. `ActiveModels` is a `map[string]int`
with room for a weight per adapter, and every entry in it is zero. The
scorer then discards the value with `_` and uses the map as a set: 1.0
if the adapter is active on that endpoint, 0.8 if there is spare
capacity (`unionCount < MaxActiveModels`), 0.6 if waiting, 0.0
otherwise.

None of the three components is written badly. Each one is reasonable
given what reaches it. The question is whether what does not reach it
matters.

### Both answers are worth having

**If imbalance costs**, `loraaffinity` is choosing between endpoints
that differ on a dimension it cannot see, and the place to propose a fix
is already marked: the scorer's own comment says "This may change later
if vLLM adds native support" (`lora_affinity.go:92-94`).

**If imbalance does not cost**, three independent components were right
to drop the signal, and a question left open since June — issue
vllm-project/vllm#45325, still unanswered by any maintainer and marked
stale — has an empirical answer.

### What this is not

One A10, synthetic load, no real tenants. This measures a decision
operators face, on hardware they use, under conditions stated in full
below. It is not a production deployment and does not claim to
generalise beyond the configuration it names.

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

## Response variables

Reported in the vocabulary the field already reads, so the result is
comparable without translation. The names are those of
`vllm/benchmarks/serve.py:327` at `27757dde02`:

- **Latency**: `ttft`, `tpot`, `itl`, `e2el`, each with mean, median and
  percentiles.
- **Throughput**: `request_throughput`, `output_throughput`,
  `total_token_throughput`.

`tpot` and `itl` are not the same measurement and both are kept. `tpot`
is a per-request mean — `latency_minus_ttft / (output_len - 1)`, line
627 — so it flattens whatever happens between individual tokens. `itl`
is the list of gaps between consecutive tokens, concatenated across
requests (line 631). If multi-tenancy costs occasional stalls rather
than a uniform slowdown, `tpot` hides them and the tail percentiles of
`itl` show them. H2 is read there first.

`goodput` is not reported. It is computed only when a caller supplies
`goodput_config_dict` (lines 578, 639) with `ttft` or `tpot` keys (643,
648) — that is, against an SLO the operator declares. One chosen here
would be an arbitrary parameter deciding the outcome. The distributions
are published instead, so a reader with an SLO can apply it.

### The layer this campaign adds

None of the above says what the work costs in energy. Energy per
generated token, and tokens per joule, come from inferscope's NVML
counter and are the reason this campaign exists rather than being a
rerun of a published benchmark.

The relation between the two layers is the finding, not either alone:
throughput falling while tokens-per-joule stays flat means multi-tenancy
costs time and not energy — opposite conclusions on a rented node and on
owned hardware. agentic-kv already met that split: generation cost held at
$0.00911-0.00916 per trajectory across all fifteen cells, a 0.5% band,
while $/M token moved +56%. All of that difference was waiting, not
generating.

A dollar figure, if reported, is derived from wall-clock time times a
declared rate and is labelled as derived. On a rented node it is
throughput wearing a currency sign.

### Discard criterion

A cell with any `failed` request is not comparable and is rerun, not
averaged. `completed` and `failed` are both recorded.

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
- **One rank, shared by every adapter.** vLLM does not require this:
  `vllm/lora/peft_helper.py:128` rejects an adapter only when its rank
  *exceeds* `max_lora_rank`, so adapters of mixed rank coexist happily.
  The constraint is this design's own. Slots are preallocated at
  `max_lora_rank` so memory would not move, but the kernel work does
  depend on each adapter's actual rank, and a set of mixed ranks would
  vary cost for a reason that has nothing to do with multi-tenancy.
  `max_lora_rank` itself is one of `Literal[1, 8, 16, 32, 64, 128, 256,
  320, 512]` (`vllm/config/lora.py:27`), default 16.
- **Fixed target modules, one base model, adapters loaded before load
  begins.**

Two loading constraints the adapter selection must satisfy, read at the
same revision: an adapter whose rank exceeds `max_lora_rank` is rejected
(`peft_helper.py:128`), and an adapter whose `bias` is anything but
`"none"` contributes "Adapter bias is not supported." to an error list
that is then raised as a `ValueError` (`peft_helper.py:133-136`).

## Configuration

**Base model: `Qwen/Qwen2.5-7B-Instruct`.** Three reasons, none of them
availability — the Hugging Face API returns adapters against this base
past the query limit, as it does for Mistral 7B, Llama 3.1 8B and Qwen
2.5 3B, so the choice is not forced by scarcity.

- It is the model in inferscope's canonical vLLM fixture
  (`crates/is-metrics/tests/fixtures/vllm-prometheus-client-exposition.txt`),
  so the instrument has already been exercised against this exposition.
- It is the family used in earlier campaigns in this portfolio, which
  makes the numbers comparable with them rather than free-standing.
- Llama is gated, which has already constrained one earlier choice.

7B on a 24 GB A10 is deliberately tight: the published safetensors
total 14.19 GiB (summed from the Hugging Face model API), leaving the
rest for KV cache and eight preallocated LoRA slots. A smaller base
would leave more headroom and measure a less representative case.

**Declared fallback.** If the dry run shows that eight slots plus the
KV cache do not fit, the base drops to `Qwen/Qwen2.5-3B-Instruct`,
which the same query shows has adapters past the limit too. Recorded
here so the fallback is a plan rather than an improvisation on a paid
node.

**Adapter selection criteria**, to be satisfied before any is named:

- `library_name: peft` — the base-model filter also returns quantised
  checkpoints that merely declare this base, which are not adapters;
- one rank shared by all eight, read from each `adapter_config.json`
  rather than from the model card;
- `bias: none`, since an adapter carrying a bias is refused
  (`peft_helper.py:133-136`);
- rank not exceeding `max_lora_rank` (`peft_helper.py:128`).

The eight are named in this document before the first cell runs.

## The load, and what holding it fixed costs

Every request carries the same prompt and generates the same number of
tokens, whichever adapter serves it: `ignore_eos=True` with a fixed
`max_tokens`. Both are exposed by vLLM's own serving benchmark
(`vllm/benchmarks/serve.py:802`, passed through to the request payload
in `lib/endpoint_request_func.py:135-136`), and `ignore_eos=True` is
what `vllm/benchmarks/latency.py:100` already does when measuring
latency.

The reason is that adapters differ by task. The public adapters
published against this base cover materials analysis, Vietnamese law,
physics in Bengali, IELTS essay writing and more; whichever eight are
selected, each would answer its own natural prompt with a different
output length.
Output length drives `tpot`, `itl` and throughput directly, so letting
it vary would fold the difference between tasks into the difference
this campaign attributes to multi-tenancy. With `ignore_eos` the
generated length is identical in every cell and the only thing that
changes is how many adapters serve the same work.

**What that costs, stated here rather than found later.** The answers
are nonsense: an adapter trained on Vietnamese law is being asked
whatever the shared prompt says, and forced to keep generating past its
own stopping point. Computational cost does not depend on the answer
being sensible, which is why this is sound for the question asked — but
it means the campaign measures multi-tenancy **at equal generative
work**, not multi-tenancy as deployed, where different tenants send
different traffic with different shapes. A result here does not carry
to a fleet where adapter A answers in 30 tokens and adapter B in 800.
That is a separate experiment with a separate design.

`min_tokens` is left at its default: with `ignore_eos` set, generation
does not stop before the cap, so a floor would constrain nothing.

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
