# Evidence from the bench

What is here backs numbers that appear in the documents of this repository.
None of it is a measurement of the campaign: it comes from
llm-d-inference-sim at `e924683`, whose latencies are declared rather than
computed, on a CPU-only VM. It exists so the figures quoted elsewhere can
be recomputed rather than believed.

## `bench-runs/`

Six manifests written by `harness/run_cell.py` on 2026-09-23, from three
sessions: `cells-*` before the timing fields were split, `cells2-*` and
`cells3-*` after. The inventory of the harness quotes a load-generator
preamble ranging from 10.07 s to 14.033 s over six runs; those are these
six. In the older two the preamble is recomputed as process time minus the
benchmark's own active duration, which is what the later manifests record
directly.

## `count-proxy.py`

The HTTP proxy that counted requests per model in front of the simulator.
It is what established that `--lora-assignment round-robin` distributes
twelve requests 4/4/4 over three adapters, and that the benchmark's
readiness check sends one inference request carrying the base model rather
than an adapter. Both claims appear in the protocol and in the harness
inventory. It is a one-off verification tool, not part of the harness: the
distribution is decided entirely by the load generator, so observing it
once is enough.

Run it as `python3 evidence/count-proxy.py http://<server> <port>` and
point the benchmark at the proxy's port; it prints its counts on SIGINT.
