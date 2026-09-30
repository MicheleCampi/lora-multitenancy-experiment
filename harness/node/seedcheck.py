"""Chooses the seeds of a plan's cells before the node.

The plan is a module with IN_LEN, BLOCK and CELLS: plan (the second dry run,
the default) or campaign_plan (the campaign). For every cell of CELLS it
looks, from the starting seed up, for a main seed and a spare for the rerun
(r30 of plan.py has only one, used twice by design).
A seed is accepted if:
- every prompt has exactly IN_LEN tokens;
- no request shares its first block of BLOCK tokens with
  another request of the same adapter in the cell, in the main seed
  of the same cell (for the spare) or in the plan's two cells before.

The first block is enough: block hashes are chained, and a missing block
makes the following ones miss (vllm/v1/core/single_type_kv_cache_manager.py:
790-791, vLLM 0.30.0). Two cells are enough: the free-block queue is LRU
(v1/core/kv_cache_utils.py:255-257), cached blocks go back to its tail
(v1/core/block_pool.py:792-798) and new ones are taken from its head (678);
every measured cell allocates at least 336 x 512 = 172032 tokens, against
89968 of cache measured on the node. That one cell is enough to evict the
ones before is an inference: the second cell is the margin, and the hit
check of run_cell.py remains the guard.

The adapter of request i is adapters[i % len], as in the benchmark's
round-robin; for the skewed cells it is the expanded list that run_cell.py
passes to --lora-modules. The prompts are generated with the benchmark's code:
the parser (benchmarks/serve.py:1601), get_tokenizer as at 2107-2113,
get_samples as at 2180, with the tokenizer of the pinned snapshot. Usage:
  seedcheck.py <base model snapshot> <seeds.json> [starting seed] [plan]
"""
import hashlib
import importlib
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

from vllm.benchmarks.datasets import get_samples
from vllm.benchmarks.serve import add_cli_args
from vllm.tokenizers import get_tokenizer
from vllm.utils.argparse_utils import FlexibleArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PLAN_NAME = sys.argv[4] if __name__ == "__main__" and len(sys.argv) > 4 else "plan"
plan = importlib.import_module(PLAN_NAME)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
WINDOW = 2
SNAP = None
tok = None


def parse(adapters, n, seed):
    p = FlexibleArgumentParser()
    add_cli_args(p)
    return p.parse_args([
        "--backend", "openai", "--model", MODEL, "--tokenizer", SNAP,
        "--dataset-name", "random", "--num-prompts", str(n),
        "--random-input-len", str(plan.IN_LEN), "--random-output-len", str(plan.OUT_LEN),
        "--seed", str(seed), "--lora-modules", *adapters, "--lora-assignment", "round-robin"])


def load(snap):
    """Builds the tokenizer as the benchmark does, from the pinned snapshot."""
    global SNAP, tok
    SNAP = snap
    first = parse(["a1"], 1, 0)
    tok = get_tokenizer(first.tokenizer, tokenizer_mode=first.tokenizer_mode,
                        trust_remote_code=first.trust_remote_code)


def keys(adapters, n, seed):
    """Keys (adapter, first block) of the requests, or the reason for rejection."""
    reqs = get_samples(parse(adapters, n, seed), tok)
    if len(reqs) != n:
        return None, f"{len(reqs)} requests instead of {n}"
    off = sum(1 for r in reqs if r.prompt_len != plan.IN_LEN)
    if off:
        return None, f"{off} prompts off length"
    out = []
    for i, r in enumerate(reqs):
        ids = tok.encode(r.prompt, add_special_tokens=False)
        out.append((adapters[i % len(adapters)], tuple(ids[:plan.BLOCK])))
    if len(set(out)) != len(out):
        return None, "first block repeated within the cell"
    return set(out), None


def choose(start):
    history: list[set] = []
    chosen: dict = {}
    rejected: list = []
    seed = start
    for tag, adapters, n in plan.CELLS:
        window = set().union(*history[-WINDOW:])
        picks, cell_keys = [], set()
        while len(picks) < (1 if tag == "r30" else 2):
            k, why = keys(adapters, n, seed)
            if why is None and k & (window | cell_keys):
                why = "first block already used in the window"
            if why is None:
                cell_keys |= k
                picks.append(seed)
            else:
                rejected.append({"cell": tag, "seed": seed, "reason": why})
            seed += 1
        history.append(cell_keys)
        chosen[tag] = {"seed": picks[0], "spare": picks[1] if len(picks) > 1 else None,
                       "adapters": adapters, "num_prompts": n}
        print(tag, chosen[tag]["seed"], chosen[tag]["spare"], flush=True)
    return {"plan": PLAN_NAME,
            "plan_sha256": hashlib.sha256(open(plan.__file__, "rb").read()).hexdigest(),
            "snapshot": SNAP, "start": start, "block": plan.BLOCK, "in_len": plan.IN_LEN,
            "window_cells": WINDOW, "cells": chosen, "rejected": rejected}


if __name__ == "__main__":
    load(sys.argv[1])
    out = sys.argv[2]
    result = choose(int(sys.argv[3]) if len(sys.argv) > 3 else 2001)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print("rejected:", len(result["rejected"]), "| written", out)
    for r in result["rejected"]:
        print("  rejected", r["cell"], r["seed"], r["reason"])
