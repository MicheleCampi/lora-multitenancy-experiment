#!/usr/bin/env python3
"""Recompute cost per token and the H1/H2 judgements from a results directory.

Reads only primary files: the manifest.json run_cell.py writes for each cell
and the idle measurements (d8-idle*.json, inferscope output). Summaries
written by phase3.py or campaign.py are not read, so this is an independent
recomputation of what they report.

For each cell directory (warmup excluded):
- the verdict is recomputed with the six conditions of run_cell.py: load
  contained, no failed request, completed equal to declared, energy present
  when the cell required the GPU, no prefix-cache hit, every prompt of the
  declared length;
- the configuration comes from the manifest itself: distinct adapters N,
  skewed if the weights differ, concurrency and prompt count;
- tokens = completed x output length (--ignore-eos, max_tokens = length);
- J/token raw = energy / tokens; s/token = active / tokens;
- J/token net = (energy - P_idle x (window - active)) / tokens, with P_idle
  the mean power of the idle measurements in the directory.

Decision rules, fixed on 2026-09-29 before the campaign (PROTOCOL.md,
amendments). The deciding metric is J/token net; band = (max - min) / mean
over the kept repetitions of one configuration.
- H1, at each concurrency and prompt count: m = (N=8 uniform - N=1) / N=1.
  Inconclusive if either band is 5% or more or either side has fewer than two
  repetitions; falsified if m < 5%; otherwise not falsified.
- H2, at each N in (2, 4, 8) and each concurrency and prompt count:
  d = (skewed - uniform) / uniform. Inconclusive if either side has fewer
  than two repetitions; falsified if |d| is below the larger of the two bands;
  otherwise not falsified. Each level is reported on its own.

Usage: analyze.py <results directory> [--json]
"""
import glob
import json
import os
import sys

THRESHOLD = 0.05
METRICS = ("j_net", "s", "j_raw")


def verdict(m: dict) -> tuple[bool, list[str]]:
    cell = m["cell"]
    s = m.get("load_summary") or {}
    why = []
    if not m["containment"]["load_inside_window"]:
        why.append("load not contained")
    if s.get("failed") != 0:
        why.append(f"failed={s.get('failed')}")
    if s.get("completed") != cell["num_prompts"]:
        why.append(f"completed={s.get('completed')} of {cell['num_prompts']}")
    if m["gpu"]["energy_millijoules"] is None and m["gpu"]["required"]:
        why.append("no energy")
    if (m.get("prefix_cache_hits_in_cell") or 0) > 0:
        why.append(f"prefix hits={m['prefix_cache_hits_in_cell']}")
    if not m["prompt_lengths"]["exact"]:
        why.append(f"prompts off length={m['prompt_lengths']['off_target']}")
    return not why, why


def idle_watts(res: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(res, "d8-idle*.json"))):
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        e = (d.get("gpu") or {}).get("energy_millijoules")
        secs = d.get("duration_secs")
        if e is not None and secs:
            out.append({"file": os.path.basename(f), "watts": e / 1000 / secs})
    return out


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "band": None}
    mean = sum(values) / len(values)
    band = (max(values) - min(values)) / mean if len(values) >= 2 and mean else None
    return {"n": len(values), "mean": mean, "band": band}


def rel(a, b):
    return (a - b) / b if a is not None and b else None


def analyze(res: str) -> dict:
    idle = idle_watts(res)
    p_idle = sum(i["watts"] for i in idle) / len(idle) if idle else None
    cells, groups = [], {}
    for mf in sorted(glob.glob(os.path.join(res, "*", "manifest.json"))):
        tag = os.path.basename(os.path.dirname(mf))
        if tag == "warmup":
            continue
        m = json.load(open(mf))
        c = m["cell"]
        ok, why = verdict(m)
        n = len(set(c["adapters"]))
        skewed = len(set(c["weights"])) > 1
        key = f"n{n}{'s' if skewed else 'u'}-c{c['max_concurrency']}-p{c['num_prompts']}"
        row = {"tag": tag, "config": key, "keep": ok, "why": why}
        if ok:
            tokens = m["load_summary"]["completed"] * c["output_len"]
            e = m["gpu"]["energy_millijoules"] / 1000
            active = m["timing_s"]["active_reported_by_benchmark"]
            row["j_raw"] = e / tokens
            row["s"] = active / tokens
            row["j_net"] = ((e - p_idle * (c["window_secs"] - active)) / tokens
                            if p_idle is not None else None)
            groups.setdefault(key, []).append(row)
        cells.append(row)
    by_config = {k: {mt: stats([r[mt] for r in rows if r[mt] is not None]) for mt in METRICS}
                 for k, rows in sorted(groups.items())}

    h1 = {}
    for k in by_config:
        if not k.startswith("n1u-"):
            continue
        tail = k[len("n1u-"):]
        one, eight = by_config[k], by_config.get("n8u-" + tail)
        if eight is None:
            continue
        a, b = one["j_net"], eight["j_net"]
        m = rel(b["mean"], a["mean"])
        if a["n"] < 2 or b["n"] < 2 or m is None:
            judgement = "inconclusive: fewer than two repetitions"
        elif a["band"] >= THRESHOLD or b["band"] >= THRESHOLD:
            judgement = "inconclusive: band of 5% or more"
        elif m < THRESHOLD:
            judgement = "falsified"
        else:
            judgement = "not falsified"
        h1[tail] = {"margin": {mt: rel(eight[mt]["mean"], one[mt]["mean"]) for mt in METRICS},
                    "band_n1": a["band"], "band_n8": b["band"],
                    "reps": [a["n"], b["n"]], "judgement": judgement}

    h2 = {}
    for k in by_config:
        for n in (2, 4, 8):
            if not k.startswith(f"n{n}u-"):
                continue
            tail = k[len(f"n{n}u-"):]
            uni, skw = by_config[k], by_config.get(f"n{n}s-" + tail)
            if skw is None:
                continue
            u, s = uni["j_net"], skw["j_net"]
            d = rel(s["mean"], u["mean"])
            if u["n"] < 2 or s["n"] < 2 or d is None:
                judgement = "inconclusive: fewer than two repetitions"
            elif abs(d) < max(u["band"], s["band"]):
                judgement = "falsified"
            else:
                judgement = "not falsified"
            h2[f"n{n}-{tail}"] = {
                "difference": {mt: rel(skw[mt]["mean"], uni[mt]["mean"]) for mt in METRICS},
                "band_uniform": u["band"], "band_skewed": s["band"],
                "reps": [u["n"], s["n"]], "judgement": judgement}

    return {"results": os.path.abspath(res), "idle": idle, "p_idle_watts": p_idle,
            "cells": cells, "by_config": by_config, "h1": h1, "h2": h2}


def pct(x):
    return "n/a" if x is None else f"{100 * x:+.2f}%"


def band(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def report(r: dict) -> None:
    print("idle:", ", ".join(f"{i['file']} {i['watts']:.2f} W" for i in r["idle"]) or "none",
          "| P_idle used:", "n/a" if r["p_idle_watts"] is None else f"{r['p_idle_watts']:.2f} W")
    kept = sum(c["keep"] for c in r["cells"])
    print(f"cells: {len(r['cells'])}, kept {kept}, discarded {len(r['cells']) - kept}")
    for c in r["cells"]:
        if not c["keep"]:
            print(f"  discarded {c['tag']}: {'; '.join(c['why'])}")
    for k, st in r["by_config"].items():
        print(k, " | ".join(
            f"{mt} n={st[mt]['n']} mean={st[mt]['mean']:.6g} band={band(st[mt]['band'])}"
            for mt in METRICS if st[mt]["mean"] is not None))
    for k, h in r["h1"].items():
        print(f"H1 {k}: margin j_net {pct(h['margin']['j_net'])}, s {pct(h['margin']['s'])}, "
              f"j_raw {pct(h['margin']['j_raw'])}; bands N=1 {band(h['band_n1'])}, "
              f"N=8 {band(h['band_n8'])}; reps {h['reps']} -> {h['judgement']}")
    for k, h in r["h2"].items():
        print(f"H2 {k}: difference j_net {pct(h['difference']['j_net'])}, "
              f"s {pct(h['difference']['s'])}, j_raw {pct(h['difference']['j_raw'])}; "
              f"bands uniform {band(h['band_uniform'])}, skewed {band(h['band_skewed'])}; "
              f"reps {h['reps']} -> {h['judgement']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: analyze.py <results directory> [--json]")
    result = analyze(sys.argv[1])
    if "--json" in sys.argv[2:]:
        print(json.dumps(result, indent=1))
    else:
        report(result)
