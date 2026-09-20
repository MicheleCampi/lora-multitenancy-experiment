# Inventory — what was read before designing the LoRA reading

Phase-2 artifact. inferscope at 05faba1, vLLM at 27757dde02,
llm-d-router at 5367d054, llm-d-inference-sim at e924683, all
2026-09-20.

## Read

**The schema and its shape.** `EngineSchema`, `Series`, `Aggregation`
with their doc comments (schema.rs:26-144). Three aggregation variants,
all reducing lines to one numeric value. `model_name` deliberately
outside the schema (schema.rs:17-20).

**The timelines.** `KvCacheTimeline` and `KvCacheSample`,
`PhaseTimeline`, `SpecTimeline` and `SpecSample` with its `push`
contract. All three are series of monotonic counters; only the KV one
carries a declarative field (`accounting`).

**The signal path, end to end.** `Engine::accounting()`
(config.rs:51) to timeline construction (scrape.rs:132) to the
timeline field (kvcache.rs:102) to the derived metric
(derive.rs:254) to the report field (metrics.rs:307) to rendering
(render.rs:257). Six points, all read.

**The scrape loop.** `scrape_during` swallows per-tick errors, so
timelines carry silent gaps. `scrape_once` produces nothing rather
than a sample the type cannot express (scrape.rs:89-92).

**The provenance model.** `HitRateProvenance` with three states and
a resolution that is explicitly not a default: a recorded fact wins,
an absence on an unversioned report is an inference rendered as one,
an absence on a versioned report asserts nothing.

**ADR-014 D1, D2, D3, D4, D7** in full text.

**The report.** `ResourceReport`, twelve fields, module header, no
grouping that separates run context from measurement.

**The producers.** vLLM's `lora_requests_info` declaration and write
path, the Rust core's equivalent, llm-d-router's extractor and
scorer, llm-d-inference-sim's `reportLoras`.

## Not read

- `metrics::Report` in full — the other report type. A decision that
  fits `ResourceReport` and not this one would split the model.
- `parse_phase` and `parse_spec` bodies. Only `parse_kvcache` was
  read, and generalising from one of three parsers is the error this
  protocol names by name.
- `PhaseSample` fields.
- `render_resource_json` and the serialisation layer.
- ADR-011, ADR-012, ADR-016 full texts.
- `MetricsConfig` construction beyond its use sites.

Nothing in the not-read column is load-bearing for the decision
recorded in the ADR that follows: the decision turns on what the
schema can express and on how provenance travels, both of which are
in the read column. Where a conclusion would need something from the
second column, it is not drawn.
