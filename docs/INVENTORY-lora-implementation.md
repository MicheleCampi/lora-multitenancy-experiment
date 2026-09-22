# Inventory — implementing ADR-017

Phase-2 artifact for the implementation, measured against what the code,
its tests and its commit message will say. inferscope at 80764cb, vLLM at
27757dde02, llm-d-router at dc6538a1, llm-d-inference-sim at e924683,
2026-09-21.

## What the code will do, and what each part rests on

**One shared line splitter.** `parse.rs` holds the same splitting logic
twice: in `parse_series`, from `rsplit_once` at line 120 to the end of the
`match` at 135, and in `parse_seconds_as_nanos`, lines 207 to 219. Both
prepare the line identically before it (`trim`, then skip empty and `#`
lines: 114-116 and 202-204). After that preparation the two copies
coincide line for line; they differ only in comments. The splitter is
extracted there, so each caller keeps its own loop and preparation.

**The value read as `f64`.** `parse_counter_value` truncates toward zero
(line 78), which its doc states is exact for counters. A timestamp is not
a counter: truncation would make two series that a producer set within
one second tie, where the producer distinguished them.

**Not through `parse_series`.** It keeps only lines whose `model_name`
equals the configured model (doc at lines 83-84, filter at line 140).
`lora_requests_info` carries no `model_name` on any producer read: the
Python labels are built at `loggers.py:1106-1110`, the Rust ones at
`metrics.rs:385-388`, and every simulator body captured shows only
`max_lora` and the two adapter lists.

**Which series.** A read can hold several. The reader takes the largest
value and records a tie among the largest rather than choosing, as the
ADR-017 postscript decides.

**Where the types live.** The reading follows `SpecReading`
(`parse.rs:299`): returned by a parse function in `is-metrics`, with what
it means decided elsewhere. It is an enum rather than a tuple, because
its outcomes are distinct and because `SpecReading`'s own doc names the
tuple's cost, a silent transposition (lines 295-298). The recorded value
lives in `is-core`, in a module of its own re-exported from the root as
every other domain is (`lib.rs:20-26`), with the derive and serde
convention of `HitRateAccounting` (`kvcache.rs:34-35`). Its resolution
against the schema version lives in `is-report`, after
`HitRateProvenance`.

**Exports.** `parse_lora` beside the other parsers (`is-metrics`
`lib.rs:45`), `scrape_lora_once` beside the other scrapes (46-49). No
`_during` variant: ADR-017 D6 decides one read, not a loop.

**The report field.** `ResourceReport` gains one field. Six struct
literals construct it, counted by command: `main.rs:626` in production,
and `main.rs:881`, `cost_end_to_end.rs:66`, `resource_report.rs:107`,
`140` and `185` in tests. A seventh grep hit, `main.rs:880`, is a
function signature.

**Where the read happens.** In `run_sample_only` the window closes when
the sleep at `main.rs:533` returns; the three scrape loops are cancelled
at 566-568 and the report is built at 626. The read goes between the end
of the sleep and the cancels.

**The version.** `REPORT_SCHEMA_VERSION` rises from 1 to 2; the readers
were audited in `AUDIT-schema-version.md`.

## Fixtures, every one from a producer

llm-d-inference-sim at e924683, captured live against the event path:

- `sim-e924683-lora-tie-all.txt`, sha256 `4410b6aba1405084...`:
  four series, one value: a tie across all, empty set first.
- `sim-e924683-lora-tie-top.txt`, sha256 `41b095242dd912fc...`:
  three series, two values: `a1` strictly older, empty set and `a2` tied on
  top.
- `sim-e924683-lora-unique-latest.txt`, sha256 `91c1b27a16e1d706...`:
  two series, two values, captured with a request verified in flight: `a1`
  strictly the most recent.

llm-d-inference-sim v0.8.2, already committed: one series, at lines
65-67 of `llm-d-inference-sim-v0.8.2-metrics.txt`.

vLLM's Rust frontend: the label shape from its own test expectations,
`rust/src/engine-core-client/src/metrics.rs:455-469`, with no
`max_lora`. Those expectations normalise the value to `<ts>`, so a test
built on them inserts a value and says so.

## Not read, and what each limits

- **The Python frontend's exposed value format**, which
  `prometheus_client` sets outside vLLM. No Python exposition of
  `lora_requests_info` exists anywhere in vLLM: a whole-tree search finds
  four lines, all in the Rust test file. No test can carry a real Python
  body, and the reader's handling of that format is untested against its
  producer.
- **The Rust frontend's real exposed value**, normalised in its tests.
- **`metrics::Report`**, out of scope by ADR-017 D7.
- **The simulator's fake-metrics path**, whose values come from
  configuration. The reader is tested on the event path only.
