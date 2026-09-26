"""Sceglie i semi delle celle del piano prima del nodo.

Per ogni cella di plan.CELLS cerca, dal seme iniziale in su, un seme
principale e uno di riserva per il rilancio (r30 ne ha uno solo, usato due
volte per disegno). Un seme e' accettato se:
- ogni prompt ha esattamente plan.IN_LEN token;
- nessuna richiesta condivide il primo blocco di plan.BLOCK token con
  un'altra richiesta dello stesso adattatore nella cella, nel seme principale
  della stessa cella (per la riserva) o nelle due celle precedenti del piano.

Il primo blocco basta: gli hash dei blocchi sono concatenati e un blocco
mancante fa mancare i successivi (vllm/v1/core/single_type_kv_cache_manager.py:
790-791, vLLM 0.30.0). Due celle bastano: la coda dei blocchi liberi e' LRU
(v1/core/kv_cache_utils.py:255-257), i blocchi in cache tornano in fondo
(v1/core/block_pool.py:792-798) e i nuovi si prendono dalla testa (678); ogni
cella D2 o D3 alloca almeno 336 x 512 = 172032 token, contro 89968 di cache
misurati sul nodo. Che una cella basti a espellere le precedenti e'
un'inferenza: la seconda cella e' il margine, e il controllo delle hit di
run_cell.py resta la guardia.

L'adattatore della richiesta i e' adattatori[i % len], come nel round-robin
del benchmark. I prompt si generano con il codice del benchmark: parser
(benchmarks/serve.py:1601), get_tokenizer come a 2107-2113, get_samples come
a 2180, con il tokenizer dello snapshot pinnato. Uso:
  seedcheck.py <snapshot del modello base> <seeds.json> [seme iniziale]
"""
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

from vllm.benchmarks.datasets import get_samples
from vllm.benchmarks.serve import add_cli_args
from vllm.tokenizers import get_tokenizer
from vllm.utils.argparse_utils import FlexibleArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan  # noqa: E402

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
    """Crea il tokenizer come il benchmark, dallo snapshot pinnato."""
    global SNAP, tok
    SNAP = snap
    first = parse(["a1"], 1, 0)
    tok = get_tokenizer(first.tokenizer, tokenizer_mode=first.tokenizer_mode,
                        trust_remote_code=first.trust_remote_code)


def keys(adapters, n, seed):
    """Chiavi (adattatore, primo blocco) delle richieste, o il motivo del rifiuto."""
    reqs = get_samples(parse(adapters, n, seed), tok)
    if len(reqs) != n:
        return None, f"{len(reqs)} richieste invece di {n}"
    off = sum(1 for r in reqs if r.prompt_len != plan.IN_LEN)
    if off:
        return None, f"{off} prompt fuori misura"
    out = []
    for i, r in enumerate(reqs):
        ids = tok.encode(r.prompt, add_special_tokens=False)
        out.append((adapters[i % len(adapters)], tuple(ids[:plan.BLOCK])))
    if len(set(out)) != len(out):
        return None, "primo blocco ripetuto nella cella"
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
                why = "primo blocco gia' usato in finestra"
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
    return {"snapshot": SNAP, "start": start, "block": plan.BLOCK, "in_len": plan.IN_LEN,
            "window_cells": WINDOW, "cells": chosen, "rejected": rejected}


if __name__ == "__main__":
    load(sys.argv[1])
    out = sys.argv[2]
    result = choose(int(sys.argv[3]) if len(sys.argv) > 3 else 2001)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print("scartati:", len(result["rejected"]), "| scritto", out)
    for r in result["rejected"]:
        print("  scartato", r["cell"], r["seed"], r["reason"])
