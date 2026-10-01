# lora-multitenancy-experiment

A routing component in production cannot tell apart two pods where the
same LoRA adapter carries 90% and 5% of the traffic. The number that
would separate them is computed, then dropped three times before any
decision is made.

| Where | What happens | Line |
|---|---|---|
| vLLM computes it | `running_lora_adapters[name] = len(stats.running)` | `vllm/v1/metrics/stats.py:644` |
| vLLM drops it | `",".join(...keys())` keeps only the names | `vllm/v1/metrics/loggers.py:1100` |
| the router stores zeros | `m[trimmed] = 0` into a `map[string]int` | `extractor.go:312` |
| the scorer discards even that | `_, active := m.ActiveModels[...]` | `lora_affinity.go:89` |
| the simulator repeats it | `maps.Keys(snap.Running)` joined by commas | `pkg/engine/vllm/metrics.go:815` |

vLLM at `27757dde02`, llm-d-router at `5367d054`, llm-d-inference-sim at
`e924683`, all read 2026-09-20.

Three projects, two languages, three teams. Each holds the per-adapter
counts internally and each drops them at the same step — so a router
developed against the simulator cannot see them even in development.

## Reproduce the premise in a minute

No GPU, no weights, no adapters downloaded. Build
[llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim)
with `make build`, then:

```
./harness/reproduce-blind-scorer.sh
```

Eight adapters are loaded and the metric reads
`running_lora_adapters=""`. Six requests go to one adapter and one
each to two others, and the three label sets are indistinguishable. At
rest every combination seen is still exposed, timestamps identical.

The script reproduces the blindness, not its cost: the simulator's
latencies are declared rather than computed. What it costs is what the
campaign is for.

This repository measures whether that blindness costs anything.

**If imbalance costs**, `loraaffinity` is choosing between endpoints on
a dimension it cannot see, and its own comment already marks where a fix
belongs: "This may change later if vLLM adds native support".

**If imbalance does not cost**, three independent components were right
to drop the signal, and vllm-project/vllm#45325 — open since June, no
maintainer reply, marked stale — has an empirical answer.

## Status

**Measured.** The campaign ran on 2026-09-30 on one NVIDIA A10: 56
cells, all kept.

- Serving eight adapters costs more per generated token than serving
  one: +7.05% energy net of idle at concurrency 128 and +5.96% at 64. H1
  is not falsified.
- At two, four and eight adapters, sending 75% of the requests to one
  adapter changes that cost by +0.11% to +0.60% against an even split,
  inside the band of repetitions at every level. H2 is falsified.

For cost per token, on this configuration, the answer is the second of
the two above.

[`RESULTS.md`](RESULTS.md) gives the judgements, what they rest on and
what they do not show. The files are in
[`evidence/campaign-2026-09-30/`](evidence/campaign-2026-09-30/), and
`harness/analyze.py` recomputes every judgement from them.

[`PROTOCOL.md`](PROTOCOL.md) was committed before any measurement
existed, which is the point of committing it first. It carries two
hypotheses with the result that would falsify each, a threshold fixed in
advance with its own limit declared, response variables named in the
field's own vocabulary, and invariants each justified by the source line
behind it. The rules the campaign is judged by were added to it on
2026-09-29, as dated amendments, before the campaign's first cell.

## What this is not

One A10, synthetic load, no real tenants. A decision operators face, on
hardware they use, under conditions stated in full. Not a production
deployment, and no claim beyond the configuration named.

Measurement is by [inferscope](https://github.com/MicheleCampi/inferscope).
