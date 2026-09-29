#!/usr/bin/env python3
"""Extract the per-request fields of each cell's load.json that the evidence needs.

load.json, written by vllm bench serve with --save-detailed, also carries
the generated text of every request, which is not published. This keeps two
per-request lists, unchanged and in order: input_lens (the prompt tokens the
server reported, or the client's count for a request without a response
usage) and errors (empty for a request that completed).

Usage: extract_load.py <results directory of the archive> <output directory>
Writes <output directory>/<cell>.json for every cell that has a load.json.
"""
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
for cell in sorted(os.listdir(src)):
    path = os.path.join(src, cell, "load.json")
    if not os.path.isfile(path):
        continue
    load = json.load(open(path))
    out = {"input_lens": load["input_lens"], "errors": load["errors"]}
    with open(os.path.join(dst, cell + ".json"), "w") as f:
        json.dump(out, f)
        f.write("\n")
    print(cell, len(out["input_lens"]), sum(1 for e in out["errors"] if e))
