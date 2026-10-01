#!/usr/bin/env python3
"""Latency per configuration from the benchmark's own load.json files.

Descriptive only: H1 and H2 are decided by analyze.py on energy per token
net of idle (PROTOCOL.md, A1). This reports what PROTOCOL.md:119-120
promises beside them - ttft, tpot, itl and e2el with mean, median and
percentiles - and, for H2, the tails of itl that line 130 says are read
first. No falsification rule was fixed for latency, and none is applied.

Per request, as vLLM 0.30.0 computes them (vllm/benchmarks/serve.py):
ttft and e2el are the request's ttft and latency; tpot is
(latency - ttft) / (output_len - 1) (line 622); itl is the request's list
of inter-token gaps, concatenated over the cell's requests (line 626).
Per cell: mean, median, p90, p99, with numpy's default (linear)
percentile, which the benchmark uses. The benchmark writes its own mean,
median and p99 of ttft, tpot and itl into load.json (lines 1391-1397,
1400-1402); every cell is checked against them, and the script stops on
any disagreement.

Per configuration: mean over the kept repetitions and band = (max - min) /
mean, the definition of A2. Cells are kept or not by analyze.verdict, the
rule analyze.py applies. Warm-up excluded.

Each cell's load.json is read if present; otherwise load-extract.json.gz, the
same file without generated_texts, as published under evidence/.

Usage: latency.py <results directory> [--json]
"""
import glob
import gzip
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import stats, verdict  # noqa: E402

METRICS = ("ttft", "tpot", "itl", "e2el")
STATS = ("mean", "median", "p90", "p99")
REL_TOL = 1e-9


def percentile(values: list[float], p: float) -> float:
    """numpy.percentile with its default method, 'linear'."""
    v = sorted(values)
    h = (len(v) - 1) * p / 100
    lo = math.floor(h)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (h - lo) * (v[hi] - v[lo])


def describe(values: list[float]) -> dict:
    return {"mean": sum(values) / len(values), "median": percentile(values, 50),
            "p90": percentile(values, 90), "p99": percentile(values, 99)}


def cell_latency(load: dict) -> dict:
    n = load["completed"]
    if load.get("failed") != 0 or not (len(load["ttfts"]) == len(load["latencies"])
                                        == len(load["itls"]) == len(load["output_lens"]) == n):
        raise SystemExit("load.json: failed requests or lists of unequal length")
    ttft, e2el = load["ttfts"], load["latencies"]
    tpot = [(e2el[i] - ttft[i]) / (load["output_lens"][i] - 1)
            for i in range(n) if load["output_lens"][i] > 1]
    itl = [x for gaps in load["itls"] for x in gaps]
    # seconds, as the benchmark stores them; reported in milliseconds
    return {m: {k: 1000 * v for k, v in describe(vals).items()}
            for m, vals in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2el", e2el))}


def check_against_benchmark(tag: str, lat: dict, load: dict) -> None:
    for m in ("ttft", "tpot", "itl"):
        for ours, theirs in (("mean", "mean"), ("median", "median"), ("p99", "p99")):
            b = load[f"{theirs}_{m}_ms"]
            if not math.isclose(lat[m][ours], b, rel_tol=REL_TOL):
                raise SystemExit(f"{tag}: {ours} {m} {lat[m][ours]} ms, benchmark {b} ms")


def analyze_latency(res: str) -> dict:
    groups, cells = {}, []
    for mf in sorted(glob.glob(os.path.join(res, "*", "manifest.json"))):
        tag = os.path.basename(os.path.dirname(mf))
        if tag == "warmup":
            continue
        m = json.load(open(mf))
        ok, _ = verdict(m)
        if not ok:
            cells.append({"tag": tag, "keep": False})
            continue
        c = m["cell"]
        d = os.path.dirname(mf)
        if os.path.exists(os.path.join(d, "load.json")):
            load = json.load(open(os.path.join(d, "load.json")))
        else:
            load = json.load(gzip.open(os.path.join(d, "load-extract.json.gz"), "rt"))
        lat = cell_latency(load)
        check_against_benchmark(tag, lat, load)
        n, skewed = len(set(c["adapters"])), len(set(c["weights"])) > 1
        key = f"n{n}{'s' if skewed else 'u'}-c{c['max_concurrency']}-p{c['num_prompts']}"
        cells.append({"tag": tag, "keep": True, "config": key, "latency_ms": lat})
        groups.setdefault(key, []).append(lat)
    by_config = {k: {m: {s: stats([r[m][s] for r in rows]) for s in STATS} for m in METRICS}
                 for k, rows in sorted(groups.items())}
    h2_itl = {}
    for k in by_config:
        for n in (2, 4, 8):
            if not k.startswith(f"n{n}u-"):
                continue
            tail = k[len(f"n{n}u-"):]
            skw = by_config.get(f"n{n}s-" + tail)
            if skw is None:
                continue
            uni = by_config[k]
            h2_itl[f"n{n}-{tail}"] = {
                s: {"difference": (skw["itl"][s]["mean"] - uni["itl"][s]["mean"]) / uni["itl"][s]["mean"],
                    "band_uniform": uni["itl"][s]["band"], "band_skewed": skw["itl"][s]["band"],
                    "reps": [uni["itl"][s]["n"], skw["itl"][s]["n"]]}
                for s in ("p90", "p99")}
    checked = sum(c["keep"] for c in cells)
    return {"results": os.path.abspath(res), "cells_checked_against_benchmark": checked,
            "cells": cells, "by_config": by_config, "h2_itl_tails": h2_itl}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def report(r: dict) -> None:
    print(f"cells kept and checked against the benchmark's own mean, median, p99: "
          f"{r['cells_checked_against_benchmark']} of {len(r['cells'])}")
    for k, st in r["by_config"].items():
        for m in METRICS:
            print(k, m, " | ".join(f"{s} {st[m][s]['mean']:.3f} ms (n={st[m][s]['n']}, "
                                   f"band {pct(st[m][s]['band'])})" for s in STATS))
    print("H2, itl tails, descriptive (no rule fixed for latency):")
    for k, h in r["h2_itl_tails"].items():
        print(f"  {k}: " + "; ".join(
            f"{s} skewed vs uniform {100 * h[s]['difference']:+.2f}%, bands uniform "
            f"{pct(h[s]['band_uniform'])}, skewed {pct(h[s]['band_skewed'])}, reps {h[s]['reps']}"
            for s in ("p90", "p99")))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: latency.py <results directory> [--json]")
    result = analyze_latency(sys.argv[1])
    if "--json" in sys.argv[2:]:
        print(json.dumps(result, indent=1))
    else:
        report(result)
