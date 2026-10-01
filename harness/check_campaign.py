"""Read-only checks on a campaign results directory, before analyze.py.

Each check reads a field at the producer's end of a joint that analyze.py
crosses, or that the campaign's validity rests on without analyze.py
checking it. Usage: check_campaign.py <results dir> [campaign-seeds.json]
"""
import glob
import json
import os
import sys

res = sys.argv[1]
seeds = json.load(open(sys.argv[2]))["cells"] if len(sys.argv) > 2 else None
fails: dict[str, list[str]] = {}
n_cells = 0


def check(name, ok, tag, detail=""):
    fails.setdefault(name, [])
    if not ok:
        fails[name].append(f"{tag} {detail}".strip())


for mf in sorted(glob.glob(os.path.join(res, "*", "manifest.json"))):
    tag = os.path.basename(os.path.dirname(mf))
    if tag == "warmup":
        continue
    n_cells += 1
    m = json.load(open(mf))
    c, s, g = m["cell"], m.get("load_summary") or {}, m["gpu"]
    meas = json.load(open(os.path.join(os.path.dirname(mf), "measure.json")))
    check("energy_source = counter", g["energy_source"] == "counter", tag, str(g["energy_source"]))
    check("energy > 0", (g["energy_millijoules"] or 0) > 0, tag, str(g["energy_millijoules"]))
    check("measure.json gpu = manifest gpu",
          (meas.get("gpu") or {}).get("energy_millijoules") == g["energy_millijoules"], tag)
    check("measure duration_secs = window_secs", meas.get("duration_secs") == c["window_secs"],
          tag, f"{meas.get('duration_secs')} vs {c['window_secs']}")
    check("prefix hits = 0 (not None)", m["prefix_cache_hits_in_cell"] == 0, tag,
          str(m["prefix_cache_hits_in_cell"]))
    check("failed = 0", s.get("failed") == 0, tag, str(s.get("failed")))
    check("completed = num_prompts", s.get("completed") == c["num_prompts"], tag,
          f"{s.get('completed')} of {c['num_prompts']}")
    check("total_output_tokens = completed x output_len",
          s.get("total_output_tokens") == (s.get("completed") or 0) * c["output_len"], tag,
          f"{s.get('total_output_tokens')} vs {(s.get('completed') or 0) * c['output_len']}")
    pl = m["prompt_lengths"]
    check("prompts: recorded = num_prompts, off = 0",
          pl["recorded"] == c["num_prompts"] and pl["off_target"] == 0 and pl["exact"], tag, str(pl))
    check("tokenizer_mismatch false", m["tokenizer_mismatch"] is False, tag)
    check("contained, measure alive at load end",
          m["containment"]["load_inside_window"] and m["timing_s"]["measure_alive_at_load_end"], tag)
    check("exit codes 0", m["exit_codes"] == {"measure": 0, "load": 0}, tag, str(m["exit_codes"]))
    check("gpu required", g["required"] is True, tag)
    if seeds is not None:
        sd = seeds.get(tag)
        check("seed = main seed of campaign-seeds.json", sd is not None and c["seed"] == sd["seed"],
              tag, f"{c['seed']} vs {sd and sd['seed']}")
        check("list passed = list checked by seedcheck",
              sd is not None and c["lora_modules_as_passed"] == sd["adapters"], tag)

idle = sorted(glob.glob(os.path.join(res, "d8-idle*.json")))
for f in idle:
    d = json.load(open(f))
    g = d.get("gpu") or {}
    t = os.path.basename(f)
    check("idle: energy_source = counter", g.get("energy_source") == "counter", t, str(g.get("energy_source")))
    check("idle: energy > 0", (g.get("energy_millijoules") or 0) > 0, t)
    check("idle: duration_secs = 60", d.get("duration_secs") == 60, t, str(d.get("duration_secs")))

print(f"cells checked: {n_cells} | idle files: {len(idle)}")
for name, bad in fails.items():
    print(("OK      " if not bad else "FAILED  ") + name + ("" if not bad else f": {len(bad)} -> " + "; ".join(bad[:4])))
