# lora-multitenancy-experiment

What multi-tenant LoRA costs: per generated token, as the number of
active adapters grows and as traffic spreads unevenly across them.

**Status: protocol only. No harness, no node booked, no data.**
[`PROTOCOL.md`](PROTOCOL.md) was committed before any measurement
existed — that is the point of committing it first. It carries two
hypotheses with the result that would falsify each, a threshold fixed
in advance with its own limit declared, and invariants each justified
by the vLLM source line behind it (read at `e378275a8f`, 2026-09-20).

Measurement is by [inferscope](https://github.com/MicheleCampi/inferscope).

## What is not here yet

The harness, the dry run, and the data. The dry run has four questions
to settle on a paid node before the campaign runs; they are listed at
the end of the protocol, and each one changes the cell count or the
hour budget.
