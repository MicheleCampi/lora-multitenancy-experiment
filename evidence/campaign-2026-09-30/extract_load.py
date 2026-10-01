#!/usr/bin/env python3
"""Write each cell's load.json without the generated text, gzip-compressed.

load.json, written by vllm bench serve with --save-detailed, also carries
the generated text of every request (generated_texts), which is not
published. Every other key is kept, with its values unchanged and in order:
the per-request lists harness/latency.py reads (ttfts, itls, latencies,
output_lens) and the benchmark's own summary figures it checks them
against. Floats are written with Python's shortest round-trip repr, so each
value reads back identical. The output is deterministic: compact
separators, gzip level 9, no timestamp and no file name in the header.

Usage: extract_load.py <results directory of the archive> <results directory of the evidence>
Writes <evidence>/<cell>/load-extract.json.gz for every cell that has a
load.json, and prints the cell, its requests and its failed requests.
"""
import gzip
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
for cell in sorted(os.listdir(src)):
    path = os.path.join(src, cell, "load.json")
    if not os.path.isfile(path):
        continue
    load = json.load(open(path))
    del load["generated_texts"]
    data = json.dumps(load, separators=(",", ":")).encode() + b"\n"
    os.makedirs(os.path.join(dst, cell), exist_ok=True)
    with open(os.path.join(dst, cell, "load-extract.json.gz"), "wb") as f:
        f.write(gzip.compress(data, compresslevel=9, mtime=0))
    print(cell, load["completed"], load["failed"])
