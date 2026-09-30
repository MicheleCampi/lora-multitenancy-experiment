"""Campaign plan (decided on 2026-09-28).

The seven configurations of PROTOCOL.md:417-431 at two concurrencies: 128,
the c* of the second dry run, and 64. Four repetitions in rounds: each round
runs the 14 combinations once, rotated ROTATION positions from the round
before. 336 prompts per cell (a multiple of 168, PROTOCOL.md:434, and of 28,
the skewed list at N=8). Windows from the 2026-09-28 durations: the N=8 cells
at c=128 ran 64.55-64.6 s; at c=64, 88 s scales d2-c64 (672 prompts) to 336.
The single source for seedcheck.py and campaign.py; the per-adapter lists
are those run_cell.adapter_list passes to the benchmark.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here)):
    if os.path.exists(os.path.join(_p, "run_cell.py")):
        sys.path.insert(0, _p)
        break
from run_cell import adapter_list  # noqa: E402

IN_LEN = 256
OUT_LEN = 256
BLOCK = 16
PROMPTS = 336
REPS = 4
CONCURRENCY = (128, 64)
WINDOW = {128: 165, 64: 216}
WARMUP_PROMPTS = 24
ROTATION = 3
ALL8 = [f"a{i}" for i in range(1, 9)]

# (name, adapters, weights): uniform at every N, skewed to 75% at every N > 1.
CONFIGS = [
    ("n1u", ALL8[:1], [1]),
    ("n2u", ALL8[:2], [1, 1]),
    ("n2s", ALL8[:2], [3, 1]),
    ("n4u", ALL8[:4], [1, 1, 1, 1]),
    ("n4s", ALL8[:4], [9, 1, 1, 1]),
    ("n8u", ALL8, [1] * 8),
    ("n8s", ALL8, [21] + [1] * 7),
]
COMBOS = [(name, ads, w, c) for c in CONCURRENCY for name, ads, w in CONFIGS]

RUNS = []
for r in range(REPS):
    k = (r * ROTATION) % len(COMBOS)
    for name, ads, w, c in COMBOS[k:] + COMBOS[:k]:
        RUNS.append({"tag": f"r{r + 1}-{name}-c{c}", "config": name, "adapters": ads,
                     "weights": w, "c": c, "round": r + 1})

# For seedcheck.py: (tag, list passed to the benchmark, prompts), in run order.
CELLS = ([("warmup", ALL8, WARMUP_PROMPTS)]
         + [(x["tag"], adapter_list(x["adapters"], x["weights"]), PROMPTS) for x in RUNS])
