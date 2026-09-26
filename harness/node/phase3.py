"""Fase 3 - misure della prova generale, con regole fissate prima del nodo.

Ordine: riscaldamento su tutti e otto gli adattatori (scartato); D8 potenza a
riposo; D2 concorrenza 64/32/16/8 su uniforme N=8; c* = concorrenza piu' bassa
con throughput >= 90% del massimo; D3/D7 tre ripetizioni alternate di N=1 e
N=8 a c*; R30 con due celle a seme ripetuto. Ogni cella passa per
run_cell.py con --require-gpu e un seme proprio. Scrive
<LORA_RUN>/results/summary.json.

Variabili solo per la prova sul simulatore (sul nodo restano ai default):
LORA_RUN, LORA_URL, LORA_NO_GPU=1, LORA_PROMPTS, LORA_IN, LORA_OUT,
LORA_PYTHON, LORA_BENCH, LORA_FIRST_WINDOW, LORA_SHORT_WINDOW, LORA_TOKENIZER.
Le finestre successive alla prima si ricavano dalla cella precedente:
preambolo misurato + 10 s + 2.2 x durata attiva x rapporto di concorrenza.

Una cella scartata si rilancia una volta con un seme nuovo: con finestra
doppia se il carico non era contenuto, con la stessa finestra altrimenti.
Ogni cella ha un tempo massimo (2 x finestra + 300 s); oltre, run_cell e i
suoi figli vengono terminati. I token di una cella sono le richieste
completate x lunghezza di uscita (--ignore-eos, max_tokens = lunghezza).
Il benchmark usa il tokenizer dello snapshot pinnato del modello base.
"""
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time

B = pathlib.Path(__file__).resolve().parent
R = pathlib.Path(os.environ.get("LORA_RUN", pathlib.Path.home() / "lora-run"))
RES = R / "results"
PY = os.environ.get("LORA_PYTHON", str(R / "venv" / "bin" / "python"))
BENCH = os.environ.get("LORA_BENCH", str(R / "venv" / "bin" / "vllm"))
URL = os.environ.get("LORA_URL", "http://127.0.0.1:8000")
NO_GPU = os.environ.get("LORA_NO_GPU") == "1"
N_PROMPTS = int(os.environ.get("LORA_PROMPTS", "168"))
IN_LEN = int(os.environ.get("LORA_IN", "256"))
OUT_LEN = int(os.environ.get("LORA_OUT", "256"))
FIRST_WINDOW = int(os.environ.get("LORA_FIRST_WINDOW", "180"))
SHORT_WINDOW = int(os.environ.get("LORA_SHORT_WINDOW", "120"))
ALL8 = [f"a{i}" for i in range(1, 9)]
BUDGET_S = 75 * 60
T0 = time.time()

session = json.loads((R / "session.json").read_text())
PID = session["pid"]
PINS = json.loads((R / "pins.json").read_text())
BASE = PINS["base"]["repo"]
TOK = os.environ.get("LORA_TOKENIZER") or PINS["base"]["path"]
seeds = iter(range(1001, 10**6))
cells: list[dict] = []
summary: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "no_gpu": NO_GPU,
                 "n_prompts": N_PROMPTS, "in_len": IN_LEN, "out_len": OUT_LEN}


def dump() -> None:
    summary["cells"] = cells
    summary["elapsed_s"] = round(time.time() - T0)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))


def run(tag, adapters, n, c, window, seed=None, weights=None):
    if time.time() - T0 > BUDGET_S:
        print(f"[{tag}] budget di fase esaurito: cella non eseguita", flush=True)
        return None
    seed = next(seeds) if seed is None else seed
    out = RES / tag
    cmd = [PY, str(B / "run_cell.py"), "--server-pid", str(PID),
           "--base-url", URL, "--metrics-url", URL + "/metrics", "--model", BASE,
           "--adapters", *adapters, "--num-prompts", str(n), "--max-concurrency", str(c),
           "--input-len", str(IN_LEN), "--output-len", str(OUT_LEN),
           "--window-secs", str(window), "--seed", str(seed), "--out-dir", str(out),
           "--inferscope", str(B / "inferscope"), "--bench-python", BENCH,
           "--tokenizer", TOK]
    if not NO_GPU:
        cmd.append("--require-gpu")
    if weights:
        cmd += ["--weights", *map(str, weights)]
    t = time.time()
    limit = 2 * window + 300
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        out_s, err_s = p.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out_s, err_s = p.communicate()
        print(f"[{tag}] oltre il tempo massimo di {limit} s: run_cell e figli terminati",
              flush=True)
    (RES / f"{tag}.stdout.txt").write_text(out_s)
    (RES / f"{tag}.stderr.txt").write_text(err_s)
    try:
        m = json.loads((out / "manifest.json").read_text())
    except OSError:
        print(f"[{tag}] nessun manifesto; rc {p.returncode}; {err_s[-300:]}", flush=True)
        return None
    s = m.get("load_summary") or {}
    tokens = (s.get("completed") or 0) * OUT_LEN
    e = m["gpu"]["energy_millijoules"]
    active = m["timing_s"]["active_reported_by_benchmark"]
    row = {
        "tag": tag, "distinct_adapters": len(set(adapters)), "c": c, "n": n,
        "window": window, "seed": seed, "verdict": "keep" if p.returncode == 0 else "discard",
        "contained": m["containment"]["load_inside_window"],
        "completed": s.get("completed"), "failed": s.get("failed"),
        "active_s": active, "preamble_s": m["timing_s"]["preamble"],
        "output_throughput": s.get("output_throughput"),
        "energy_mj": e, "energy_source": m["gpu"]["energy_source"],
        "tokens": tokens,
        "j_per_token": (e / 1000 / tokens) if e is not None and tokens else None,
        "s_per_token": (active / tokens) if active and tokens else None,
        "prefix_hits": m["prefix_cache_hits_in_cell"],
        "tokenizer_mismatch": m["tokenizer_mismatch"],
        "lora_observation": m.get("lora_observation"),
        "wall_s": round(time.time() - t, 1),
    }
    cells.append(row)
    dump()
    print(f"[{tag}] {row['verdict']} contained={row['contained']} active={active} "
          f"thr={row['output_throughput']} J/tok={row['j_per_token']} "
          f"hits={row['prefix_hits']} wall={row['wall_s']}s", flush=True)
    return row


def cell(tag, adapters, n, c, window):
    r = run(tag, adapters, n, c, window)
    if r is not None and r["verdict"] == "discard":
        if r["contained"]:
            r = run(tag + "-bis", adapters, n, c, window)
        else:
            r = run(tag + "-w2", adapters, n, c, window * 2)
    return r


def idle_power(secs=60):
    if NO_GPU:
        return {"secs": secs, "energy_mj": None, "watts": None, "power_mean_mw": None}
    p = subprocess.run(
        [str(B / "inferscope"), "--sample-only", "--pid", str(PID), "--include-descendants",
         "--duration-secs", str(secs), "--gpu", "--metrics-endpoint", URL + "/metrics",
         "--engine", "vllm", "--model", BASE, "--json"], capture_output=True, text=True)
    (RES / "d8-idle.json").write_text(p.stdout)
    g = (json.loads(p.stdout).get("gpu") or {}) if p.returncode == 0 and p.stdout.strip() else {}
    e = g.get("energy_millijoules")
    w = e / 1000 / secs if e is not None else None
    print(f"[d8] potenza a riposo {w} W (energia {e} mJ in {secs} s, "
          f"media {g.get('power_mean_milliwatts')} mW)", flush=True)
    return {"secs": secs, "energy_mj": e, "watts": w,
            "power_mean_mw": g.get("power_mean_milliwatts")}


def stats(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return {"n": 0, "mean": None, "band": None}
    mean = sum(v) / len(v)
    band = (max(v) - min(v)) / mean if len(v) >= 2 and mean else None
    return {"n": len(v), "mean": mean, "band": band}


RES.mkdir(parents=True, exist_ok=True)

run("warmup", ALL8, 24, 8, SHORT_WINDOW)
summary["d8_idle"] = idle_power(60)

sweep, prev = [], None
for c in (64, 32, 16, 8):
    window = FIRST_WINDOW if prev is None else math.ceil(
        (prev["preamble_s"] or 15) + 10 + 2.2 * prev["active_s"] * prev["c"] / c)
    r = cell(f"d2-c{c}", ALL8, N_PROMPTS, c, window)
    if r is None:
        break
    if r["verdict"] == "keep":
        sweep.append(r)
        prev = r
summary["d2"] = [r["tag"] for r in sweep]
if not sweep:
    dump()
    print("FASE3 FERMA: nessuna cella di concorrenza valida", flush=True)
    sys.exit(1)
best = max(r["output_throughput"] for r in sweep)
c_star = min(r["c"] for r in sweep if r["output_throughput"] >= 0.9 * best)
ref = next(r for r in sweep if r["c"] == c_star)
win = math.ceil((ref["preamble_s"] or 15) + 10 + 2.2 * ref["active_s"])
summary["c_star"] = c_star
summary["d3_window"] = win
print(f"[d2] throughput massimo {best}; c* = {c_star}; finestra D3 {win} s", flush=True)

reps = {"n1": [], "n8": []}
for i in range(1, 4):
    for key, ads in (("n1", ["a1"]), ("n8", ALL8)):
        r = cell(f"d3-{key}-r{i}", ads, N_PROMPTS, c_star, win)
        if r is not None and r["verdict"] == "keep":
            reps[key].append(r)

P = summary["d8_idle"]["watts"]
for r in cells:
    if P is not None and r["energy_mj"] is not None and r["active_s"] and r["tokens"]:
        e_active = r["energy_mj"] / 1000 - P * (r["window"] - r["active_s"])
        r["j_per_token_idle_corrected"] = e_active / r["tokens"]

d3 = {k: {m: stats(rows, m) for m in ("j_per_token", "j_per_token_idle_corrected", "s_per_token")}
      for k, rows in reps.items()}
summary["d3"] = d3
summary["d7_margin"] = {
    m: ((d3["n8"][m]["mean"] - d3["n1"][m]["mean"]) / d3["n1"][m]["mean"])
    if d3["n1"][m]["mean"] and d3["n8"][m]["mean"] else None
    for m in ("j_per_token", "j_per_token_idle_corrected", "s_per_token")}

a = run("r30-a", ALL8, 24, 8, SHORT_WINDOW, seed=777)
b = run("r30-b", ALL8, 24, 8, SHORT_WINDOW, seed=777)
summary["r30"] = {"first_hits": a and a["prefix_hits"], "second_hits": b and b["prefix_hits"],
                  "second_verdict": b and b["verdict"]}
dump()

print("\n== RIEPILOGO FASE 3")
print("D8 potenza a riposo (W):", P)
print("c*:", c_star)
for k in ("n1", "n8"):
    for m in ("j_per_token", "j_per_token_idle_corrected", "s_per_token"):
        st = d3[k][m]
        print(f"D3 {k} {m}: n={st['n']} media={st['mean']} banda={st['band']}")
print("D7 margine N=8 vs N=1:", summary["d7_margin"])
print("R30:", summary["r30"])
print("FASE3 COMPLETATA in", summary["elapsed_s"], "s", flush=True)
