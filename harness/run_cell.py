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


def wait_for_server(metrics_url: str, timeout_s: float) -> float:
    """Blocks until the server answers, and returns how long that took.

    The benchmark has its own readiness check, kept on as a second guard,
    but it runs inside the measurement window; this one runs before it.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(metrics_url, timeout=5) as r:
                if r.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise SystemExit(f"server did not answer {metrics_url} in {timeout_s}s")


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
    out = []
    for adapter, weight in zip(adapters, weights):
        out.extend([adapter] * weight)
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
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modules = adapter_list(args.adapters, args.weights)

    waited = wait_for_server(args.metrics_url, args.server_wait_secs)

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
    load_cmd = [
        args.bench_python, "bench", "serve",
        "--backend", "openai",
        "--base-url", args.base_url,
        "--endpoint", "/v1/completions",
        "--model", args.model,
        "--dataset-name", "random",
        "--num-prompts", str(args.num_prompts),
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--max-concurrency", str(args.max_concurrency),
        "--ignore-eos",
        "--seed", str(args.seed),
        "--ready-check-timeout-sec", "60",
        "--lora-modules", *modules,
        "--lora-assignment", "round-robin",
        "--save-result", "--save-detailed",
        "--result-dir", str(out), "--result-filename", "load.json",
    ]

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
    measure_out, measure_err = measure.communicate()
    measure_finished = time.monotonic() - t0

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
    preamble_secs = (
        round(load_finished - load_started - active_secs, 3)
        if active_secs is not None else None
    )
    contained = (
        load.returncode == 0
        and measure.returncode == 0
        and load_finished <= measure_finished
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
        },
        "commands": {"measure": measure_cmd, "load": load_cmd},
        "timing_s": {
            "server_wait": round(waited, 3),
            "load_started_after_measure": round(load_started, 3),
            "load_finished": round(load_finished, 3),
            "measure_finished": round(measure_finished, 3),
            "load_process": round(load_finished - load_started, 3),
            "active_reported_by_benchmark": active_secs,
            "preamble": preamble_secs,
        },
        "exit_codes": {"measure": measure.returncode, "load": load.returncode},
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
    ok = contained and failed == 0 and completed == args.num_prompts
    print(json.dumps({
        "out_dir": str(out),
        "contained": contained,
        "no_request_fraction_of_window": no_request_fraction,
        "preamble_secs": preamble_secs,
        "completed": completed,
        "failed_requests": failed,
        "lora": manifest["lora_observation"],
        "verdict": "keep" if ok else "discard",
    }, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
