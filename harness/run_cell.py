#!/usr/bin/env python3
"""Run one cell of the campaign and archive its evidence.

Starts nothing but the two children: the measurement (inferscope in
--sample-only, attached to a server that is already running) and the load
(vLLM's own serving benchmark). Collects both artifacts, checks that the
load ran inside the measurement window, and writes a manifest holding the
configuration, the two commands as issued, and the containment figures.

The rationale for each decision is in docs/INVENTORY-harness.md. The two
that shape this file: the window is fixed because --sample-only offers no
way to end it on a signal, so containment is verified after the fact
rather than assumed; and the PID is an argument, never discovered by name.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request


def wait_for_server(metrics_url: str, base_url: str, model: str,
                    timeout_s: float) -> float:
    """Blocks until the server completes a request on the base model, and
    returns how long that took.

    The benchmark's own readiness check is left at its default, off: when on,
    it sends an inference request on the base model inside the measurement
    window. This guard runs before the window opens, and asks for a generated
    token rather than only /metrics, so an endpoint that answers before its
    engine serves does not pass.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    body = json.dumps({"model": model, "prompt": "Hi", "max_tokens": 1}).encode()
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(metrics_url, timeout=5) as r:
                if r.status == 200:
                    req = urllib.request.Request(
                        base_url.rstrip("/") + "/v1/completions", data=body,
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=30) as c:
                        if c.status == 200:
                            return time.monotonic() - started
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise SystemExit(f"server did not complete a request at {base_url} in {timeout_s}s")


def counter_sum(metrics_url: str, name: str) -> float | None:
    """Sums every series of a Prometheus counter, or None if it is absent."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as r:
            text = r.read().decode()
    except (urllib.error.URLError, OSError):
        return None
    values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
              if line.startswith(name + "{") or line.startswith(name + " ")]
    return sum(values) if values else None


def prompt_lengths(load_result: dict | None, input_len: int,
                   num_prompts: int) -> dict:
    """Checks that every prompt the benchmark sent had the declared length.

    The random dataset stops adjusting a prompt after a fixed number of
    retries and keeps whatever length it reached, so a prompt can leave longer
    or shorter than requested. input_lens holds one entry per request sent,
    failed ones included. A cell whose prompts are not all of the declared
    length did not run its declared load; a result without input_lens cannot
    show that it did.
    """
    lens = (load_result or {}).get("input_lens")
    off = None if lens is None else sum(1 for n in lens if n != input_len)
    return {
        "expected": input_len,
        "recorded": None if lens is None else len(lens),
        "off_target": off,
        "exact": lens is not None and len(lens) == num_prompts and off == 0,
    }


def adapter_list(adapters: list[str], weights: list[int]) -> list[str]:
    """Expands adapters into the list the benchmark consumes.

    Imbalance comes from repetition, not from a patch: round-robin indexes
    the list it is given, so an adapter repeated n times receives n times
    the requests.
    """
    if not weights:
        return list(adapters)
    if len(weights) != len(adapters):
        raise SystemExit("--weights must have one entry per adapter")
    # Interleaved by weight class. Adapters of equal weight rotate among
    # themselves in the order given; classes are merged heaviest first, each
    # spread at equal intervals over the list built so far. The benchmark walks
    # the list cyclically (i % len), so a heavy adapter's repetitions are kept
    # apart across the wrap as well: at 21+7x1 no more than three consecutive
    # requests go to it. Contiguous blocks would send it up to its weight in a
    # row and change how many adapters share a batch. Every adapter still
    # appears exactly `weight` times.
    classes: dict[int, list[str]] = {}
    for adapter, weight in zip(adapters, weights):
        classes.setdefault(weight, []).append(adapter)
    out: list[str] = []
    for weight in sorted(classes, reverse=True):
        block = classes[weight] * weight
        if not block:
            continue
        m = len(out) + len(block)
        slots = {(j + 1) * m // len(block) - 1 for j in range(len(block))}
        rest, new_items = iter(out), iter(block)
        out = [next(new_items) if pos in slots else next(rest) for pos in range(m)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server-pid", type=int, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--metrics-url", required=True)
    ap.add_argument("--model", required=True,
                    help="base model, as the server names it")
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=int, default=[],
                    help="repetitions per adapter; omitted means one each")
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--max-concurrency", type=int, required=True)
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--window-secs", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--inferscope", default="inferscope")
    ap.add_argument("--bench-python", default="vllm")
    ap.add_argument("--server-wait-secs", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-gpu", action="store_true",
                    help="measure the GPU and discard a cell without energy")
    ap.add_argument("--tokenizer", required=True,
                    help="path of the pinned model snapshot; the benchmark "
                         "otherwise resolves the tokenizer by name, unpinned")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modules = adapter_list(args.adapters, args.weights)

    waited = wait_for_server(args.metrics_url, args.base_url, args.model,
                             args.server_wait_secs)

    measure_cmd = [
        args.inferscope, "--sample-only",
        "--pid", str(args.server_pid),
        "--include-descendants",
        "--duration-secs", str(args.window_secs),
        "--metrics-endpoint", args.metrics_url,
        "--engine", "vllm",
        "--model", args.model,
        "--json",
    ]
    if args.require_gpu:
        measure_cmd.append("--gpu")
    load_cmd = [
        args.bench_python, "bench", "serve",
        "--backend", "openai",
        "--base-url", args.base_url,
        "--endpoint", "/v1/completions",
        "--model", args.model,
        "--tokenizer", args.tokenizer,
        "--dataset-name", "random",
        "--num-prompts", str(args.num_prompts),
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--max-concurrency", str(args.max_concurrency),
        "--ignore-eos",
        "--seed", str(args.seed),
        "--lora-modules", *modules,
        "--lora-assignment", "round-robin",
        "--save-result", "--save-detailed",
        "--result-dir", str(out), "--result-filename", "load.json",
    ]

    hits_name = "vllm:prefix_cache_hits_total"
    hits_before = counter_sum(args.metrics_url, hits_name)
    t0 = time.monotonic()
    measure = subprocess.Popen(
        measure_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    load_started = time.monotonic() - t0
    load = subprocess.Popen(
        load_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    load_out, load_err = load.communicate()
    load_finished = time.monotonic() - t0
    # Containment means the measurement was still sampling when the load
    # ended. Comparing the two finish times cannot show that: the second is
    # taken after the first by construction.
    measure_alive_at_load_end = measure.poll() is None
    measure_out, measure_err = measure.communicate()
    measure_finished = time.monotonic() - t0
    hits_after = counter_sum(args.metrics_url, hits_name)
    # Any hit here is a prompt this cell sent before or another cell left in
    # the cache: the random dataset can repeat a prompt inside one cell as well
    # as across cells. Either way the cell did not pay its own prefill. None
    # means the server does not export the counter.
    prefix_hits = (None if hits_before is None or hits_after is None
                   else hits_after - hits_before)
    # The benchmark's warning that its tokenizer and the server's disagree. It
    # does not cover prompt length, which prompt_lengths checks.
    tokenizer_mismatch = "tokenizer mismatch" in load_out

    (out / "load.stdout.txt").write_text(load_out)
    (out / "load.stderr.txt").write_text(load_err)
    (out / "measure.stderr.txt").write_text(measure_err)

    report = None
    if measure.returncode == 0 and measure_out.strip():
        try:
            report = json.loads(measure_out)
            (out / "measure.json").write_text(json.dumps(report, indent=1))
        except json.JSONDecodeError as e:
            (out / "measure.stdout.txt").write_text(measure_out)
            print(f"warning: measurement stdout is not JSON: {e}",
                  file=sys.stderr)

    load_result = None
    load_path = out / "load.json"
    if load_path.exists():
        load_result = json.loads(load_path.read_text())

    # The benchmark's own `duration` covers the requests alone: it is
    # perf_counter() around the send loop (serve.py:1017, 1096). The load
    # process lives longer than that - importing vLLM, building the
    # tokenizer, generating the dataset - and that preamble sits inside the
    # measurement window without touching the server. Both are recorded,
    # because one figure cannot say both things.
    active_secs = load_result.get("duration") if load_result else None
    lengths = prompt_lengths(load_result, args.input_len, args.num_prompts)
    preamble_secs = (
        round(load_finished - load_started - active_secs, 3)
        if active_secs is not None else None
    )
    gpu = (report or {}).get("gpu") or {}
    energy_mj = gpu.get("energy_millijoules")
    energy_source = gpu.get("energy_source")
    contained = (
        load.returncode == 0
        and measure.returncode == 0
        and measure_alive_at_load_end
        and active_secs is not None
    )
    # The fraction of the window during which no request was in flight. It
    # holds the preamble and the tail after the last response, so it is read
    # alongside preamble_secs rather than on its own.
    no_request_fraction = None
    if contained and args.window_secs:
        no_request_fraction = round(1.0 - (active_secs / args.window_secs), 4)

    manifest = {
        "schema": "lora-cell/1",
        "cell": {
            "model": args.model,
            "adapters": args.adapters,
            "weights": args.weights or [1] * len(args.adapters),
            "lora_modules_as_passed": modules,
            "num_prompts": args.num_prompts,
            "max_concurrency": args.max_concurrency,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "window_secs": args.window_secs,
            "seed": args.seed,
            "server_pid": args.server_pid,
            "tokenizer": args.tokenizer,
        },
        "commands": {"measure": measure_cmd, "load": load_cmd},
        "timing_s": {
            "server_wait": round(waited, 3),
            "load_started_after_measure": round(load_started, 3),
            "load_finished": round(load_finished, 3),
            "measure_finished": round(measure_finished, 3),
            "measure_alive_at_load_end": measure_alive_at_load_end,
            "load_process": round(load_finished - load_started, 3),
            "active_reported_by_benchmark": active_secs,
            "preamble": preamble_secs,
        },
        "exit_codes": {"measure": measure.returncode, "load": load.returncode},
        "gpu": {"required": args.require_gpu, "energy_millijoules": energy_mj,
                "energy_source": energy_source},
        "prefix_cache_hits_in_cell": prefix_hits,
        "tokenizer_mismatch": tokenizer_mismatch,
        "prompt_lengths": lengths,
        "containment": {
            "load_inside_window": contained,
            "no_request_fraction_of_window": no_request_fraction,
        },
        "load_summary": (
            {
                k: load_result[k]
                for k in (
                    "completed", "failed", "total_output_tokens",
                    "output_throughput", "mean_ttft_ms", "mean_itl_ms",
                )
                if k in load_result
            }
            if load_result else None
        ),
        "lora_observation": report.get("lora") if report else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    failed = load_result.get("failed") if load_result else None
    completed = load_result.get("completed") if load_result else None
    # A cell that sent fewer requests than it declared did not run the load
    # it claims to have measured, however cleanly it exited.
    ok = (contained and failed == 0 and completed == args.num_prompts
          and (energy_mj is not None or not args.require_gpu)
          and not (prefix_hits or 0) > 0
          and lengths["exact"])
    print(json.dumps({
        "out_dir": str(out),
        "contained": contained,
        "no_request_fraction_of_window": no_request_fraction,
        "preamble_secs": preamble_secs,
        "completed": completed,
        "failed_requests": failed,
        "energy_millijoules": energy_mj,
        "prefix_cache_hits_in_cell": prefix_hits,
        "tokenizer_mismatch": tokenizer_mismatch,
        "prompt_lengths_off_target": lengths["off_target"],
        "lora": manifest["lora_observation"],
        "verdict": "keep" if ok else "discard",
    }, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
