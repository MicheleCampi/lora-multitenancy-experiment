# Inventory — the cell harness

Phase-2 artifact, written before the script. It runs one cell and archives
its evidence; it does not start the server. inferscope at acd21ec, vLLM
0.30.0, llm-d-inference-sim at e924683, 2026-09-23.

## What the script does, and what each step rests on

- **Waits for the server itself**, then starts both children. The benchmark's
  own readiness check stays on (`--ready-check-timeout-sec`, default 0, so it
  must be passed explicitly) as a second guard.

- **Starts the measurement**: `inferscope --sample-only --pid <server>
  --duration-secs <window> --metrics-endpoint <url> --engine vllm --model
  <base> --json`. `--model` is required with a metrics endpoint since
  acd21ec, and names the base model: it selects the `model_name` label
  series.

- **Starts the load**: `vllm bench serve` with the cell's adapter list,
  `--lora-assignment round-robin`, `--ignore-eos`, a fixed
  `--random-output-len`, and `--save-result --save-detailed`.

- **Collects both**, checks containment, and archives the two JSON files with
  the configuration that produced them.

## Decisions, and why

- **The window is fixed and containment is verified afterwards.**
  `--duration-secs` is required with `--sample-only` and there is no way to
  end the window on a signal, so the script cannot make the measurement last
  exactly as long as the load. It instead checks, from the artifacts
  themselves, that the load ran inside the window, and records the idle
  fraction alongside the result: a joules-per-token figure should be read
  knowing how much of its window served nothing. A cell whose load is not
  contained is discarded and re-run, as the protocol already prescribes for
  `failed > 0`.

- **The PID is an argument, never discovered.** Matching a process by name
  works until a second vLLM runs on the same host, and then the measurement
  is of another process with nothing to say so.

- **`--include-descendants` is passed.** In vLLM 0.30.0 the inference work
  runs in engine-core processes that are forked rather than in the process a
  user points at (`vllm/v1/engine/core_client.py:101`, `664`, `767`), which
  is the case ADR-006 describes. Whether they are children of the serving
  process or sit deeper is not established here, and the dry run must check
  it: the flag sums direct children only.

- **The measurement window holds more than the requests, and the manifest
  says how much.** Inside it fall the benchmark's readiness request, one
  inference call on the base model; the tokenizer probe; and the load
  process's own start-up, which imports vLLM and torch and builds the dataset
  before sending anything. Measured on the bench over six runs, that preamble
  ranged from 10.07 s to 14.03 s while the requests themselves took between
  0.024 s and 0.037 s. The harness therefore records three separate figures -
  the benchmark's own active duration, the load process's wall time, and
  their difference - and never collapses them into one idle number. Starting
  the measurement after the preamble would instead lose the beginning of the
  load, which is where H1 expects a cost.

## Observed, on the bench

- `round-robin` over three adapters distributes twelve requests 4/4/4,
  counted from the request bodies by a proxy in front of the simulator.

- `--ignore-eos` with a fixed output length yields exactly `num_prompts x
  output_len` generated tokens.

- The readiness check, when enabled, sends one `/v1/completions` request
  carrying the base model, not an adapter.

- Prompt alignment happens before the benchmark starts and stops after one
  request when the tokenizers agree; on the simulator they disagree and it
  costs one request per prompt.

- inferscope records the LoRA read on this path: with `--model` the report
  carries `lora` and `kvcache_timeline`, and a read against a closed port
  records `read_failed`.

## Not read, and what each limits

- **Whether engine-core forks further workers.** `--include-descendants` sums
  the parent with the PIDs in `/proc/<pid>/task/<pid>/children`, its direct
  children. Grandchildren, if any, are outside the measurement. The dry run
  must check this on the node; the simulator is a single Go process, so the
  bench cannot exercise it.

- **The node's process layout under the campaign's serving flags.**
  `core_client.py:630` names a configuration where engines are managed
  externally and not forked, so the layout is not fixed by the version alone.

- **Whether the simulator's metric vocabulary matches vLLM's for every family
  the campaign reads.** It is known not to for KV cache.

- **How long a cell takes to run**, which the dry run establishes and which
  sets the window.
