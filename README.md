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

This repository measures whether that blindness costs anything.

**If imbalance costs**, `loraaffinity` is choosing between endpoints on
a dimension it cannot see, and its own comment already marks where a fix
belongs: "This may change later if vLLM adds native support".

**If imbalance does not cost**, three independent components were right
to drop the signal, and vllm-project/vllm#45325 — open since June, no
maintainer reply, marked stale — has an empirical answer.

## Status

**Protocol only. No harness, no node booked, no data.**

[`PROTOCOL.md`](PROTOCOL.md) was committed before any measurement
existed, which is the point of committing it first. It carries two
hypotheses with the result that would falsify each, a threshold fixed in
advance with its own limit declared, response variables named in the
field's own vocabulary, and invariants each justified by the source line
behind it.

## What this is not

One A10, synthetic load, no real tenants. A decision operators face, on
hardware they use, under conditions stated in full. Not a production
deployment, and no claim beyond the configuration named.

Measurement is by [inferscope](https://github.com/MicheleCampi/inferscope).
