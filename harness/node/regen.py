"""Regenerates the 13 cells' random prompts offline and finds shared prefixes.

Uses the code of vllm bench serve 0.30.0: parser (serve.py:60, 1601),
argument aliases (2135-2146), get_tokenizer (2107-2113), get_samples (2180).
Fidelity: the regenerated prompt_len == input_lens of load.json, cell by cell.
Adapter of request i: lora_modules[i % len] (round-robin).
For every request: full blocks of 16 tokens shared with an earlier request
(same cell or cells before) served by the same adapter. Prints in Italian.
"""
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

from vllm.benchmarks.datasets import get_samples
from vllm.benchmarks.serve import add_cli_args
from vllm.tokenizers import get_tokenizer
from vllm.utils.argparse_utils import FlexibleArgumentParser

RES = sys.argv[1]
ORDER = ["warmup", "d2-c64", "d2-c32", "d2-c16", "d2-c8",
         "d3-n1-r1", "d3-n8-r1", "d3-n1-r2", "d3-n8-r2", "d3-n1-r3", "d3-n8-r3",
         "r30-a", "r30-b"]
BLOCK = 16

seen = {}  # (adapter, prefix of k blocks) -> (cell, index)
for cell in ORDER:
    man = json.load(open(f"{RES}/{cell}/manifest.json"))
    argv = man["commands"]["load"][3:]
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(argv)
    if args.input_len is not None:
        args.random_input_len = args.input_len
    if args.output_len is not None:
        args.random_output_len = args.output_len
    tok_id = args.tokenizer if args.tokenizer is not None else args.model
    tok = get_tokenizer(tok_id, tokenizer_mode=args.tokenizer_mode,
                        trust_remote_code=args.trust_remote_code)
    reqs = get_samples(args, tok)
    lens = [r.prompt_len for r in reqs]
    ref = json.load(open(f"{RES}/{cell}/load.json"))["input_lens"]
    mods = args.lora_modules
    print(f"{cell}: seme {args.seed} | richieste {len(reqs)} | fedelta' "
          f"{'OK' if lens == ref else 'DIVERSA'} | adattatori {mods}", flush=True)
    for i, r in enumerate(reqs):
        ad = mods[i % len(mods)]
        ids = tok.encode(r.prompt, add_special_tokens=False)
        best, src = 0, None
        for k in range(1, len(ids) // BLOCK + 1):
            hit = seen.get((ad, tuple(ids[:k * BLOCK])))
            if hit is None:
                break
            best, src = k, hit
        if best:
            print(f"  {cell}[{i}] {ad}: {best} blocchi = {best * BLOCK} token "
                  f"in comune con {src[0]}[{src[1]}]")
        for k in range(1, len(ids) // BLOCK + 1):
            seen.setdefault((ad, tuple(ids[:k * BLOCK])), (cell, i))
print("FINE")
