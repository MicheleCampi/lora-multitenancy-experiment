"""Piano della campagna (deciso il 28/9/2026).

Le sette configurazioni di PROTOCOL.md:417-431 a due concorrenze: 128, il c*
della seconda prova generale, e 64. Quattro ripetizioni in turni: ogni turno
esegue le 14 combinazioni una volta, ruotate di ROTATION posizioni rispetto al
turno precedente. 336 prompt per cella (multiplo di 168, PROTOCOL.md:434, e di
28, la lista sbilanciata a N=8). Le finestre vengono dalle durate misurate il
28/9: 64.6 s a c=128 con N=8, 88 s stimati a c=64. Unica fonte per seedcheck.py
e campaign.py; le liste per adattatore sono quelle che run_cell.adapter_list
passa al benchmark.
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

# (nome, adattatori, pesi): uniforme a ogni N, sbilanciata al 75% a ogni N > 1.
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

# Per seedcheck.py: (tag, lista passata al benchmark, prompt), in ordine di esecuzione.
CELLS = ([("warmup", ALL8, WARMUP_PROMPTS)]
         + [(x["tag"], adapter_list(x["adapters"], x["weights"]), PROMPTS) for x in RUNS])
