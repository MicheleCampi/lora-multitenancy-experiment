"""Campagna - le 56 celle di campaign_plan.py, con regole fissate prima del nodo.

Si lancia al posto di phase3.py, dopo phase0, phase1 e phase2, in una sola
sessione del nodo. Le celle, le finestre e l'ordine vengono da campaign_plan.py;
i semi da campaign-seeds.json, scelti prima del nodo da seedcheck.py sulle
liste espanse che run_cell.py passa al benchmark. Ordine: riscaldamento su
tutti e otto gli adattatori (scartato); potenza a riposo (inizio); turni 1 e 2;
potenza a riposo (meta'); turni 3 e 4; potenza a riposo (fine). Ogni cella
passa per run_cell.py con --require-gpu. Scrive <LORA_RUN>/results/summary.json
dopo ogni cella. Prima del riscaldamento scrive <LORA_RUN>/results/gpu.csv con
nome, driver e memoria della GPU secondo nvidia-smi; se nvidia-smi non risponde,
la campagna si ferma (con LORA_NO_GPU=1 prosegue senza il file).

Le finestre sono fisse per concorrenza (campaign_plan.WINDOW). Una cella
scartata si rilancia una volta con il seme di riserva: con finestra doppia se
il carico non era contenuto, con la stessa finestra altrimenti. Ogni cella ha
un tempo massimo (2 x finestra + 300 s); oltre, run_cell e i suoi figli vengono
terminati. I token di una cella sono le richieste completate x lunghezza di
uscita (--ignore-eos, max_tokens = lunghezza). Il benchmark usa il tokenizer
dello snapshot pinnato del modello base.

La campagna si ferma se il processo del server non esiste piu' o se il tempo
di fase supera BUDGET_S; le celle non eseguite restano registrate come tali.
La correzione per il riposo usa la media delle misure di potenza a riposo
riuscite; le tre misure e la loro escursione sono nel riepilogo.

Variabili solo per la prova sul simulatore (sul nodo restano ai default):
LORA_RUN, LORA_URL, LORA_NO_GPU=1, LORA_PYTHON, LORA_BENCH, LORA_TOKENIZER,
LORA_WINDOW_SCALE (moltiplica le finestre, default 1), LORA_SHORT_WINDOW.
"""
import hashlib
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
SCALE = float(os.environ.get("LORA_WINDOW_SCALE", "1"))
SHORT_WINDOW = int(os.environ.get("LORA_SHORT_WINDOW", "120"))
IDLE_SECS = 60
BUDGET_S = 4 * 3600
T0 = time.time()

sys.path.insert(0, str(B))
import campaign_plan as plan  # noqa: E402

# I semi valgono per il piano e per le liste con cui sono stati verificati.
SEEDS = json.loads((B / "campaign-seeds.json").read_text())
if SEEDS.get("plan") != "campaign_plan":
    sys.exit(f"CAMPAGNA FERMA: campaign-seeds.json e' del piano {SEEDS.get('plan')}")
if SEEDS.get("plan_sha256") != hashlib.sha256((B / "campaign_plan.py").read_bytes()).hexdigest():
    sys.exit("CAMPAGNA FERMA: campaign-seeds.json e' di un'altra versione di campaign_plan.py")
SEEDS = SEEDS["cells"]
wrong = [x["tag"] for x in plan.RUNS
         if x["tag"] not in SEEDS
         or SEEDS[x["tag"]]["adapters"] != plan.adapter_list(x["adapters"], x["weights"])
         or SEEDS[x["tag"]]["num_prompts"] != plan.PROMPTS]
if wrong or "warmup" not in SEEDS:
    sys.exit(f"CAMPAGNA FERMA: semi mancanti o verificati su altre liste per "
             f"{wrong or ['warmup']}")

session = json.loads((R / "session.json").read_text())
PID = session["pid"]
PINS = json.loads((R / "pins.json").read_text())
BASE = PINS["base"]["repo"]
TOK = os.environ.get("LORA_TOKENIZER") or PINS["base"]["path"]
cells: list[dict] = []
summary: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "no_gpu": NO_GPU,
                 "prompts": plan.PROMPTS, "in_len": plan.IN_LEN, "out_len": plan.OUT_LEN,
                 "windows": {str(c): math.ceil(w * SCALE) for c, w in plan.WINDOW.items()},
                 "reps": plan.REPS, "rotation": plan.ROTATION, "not_run": [], "idle": {}}
METRICS = ("j_per_token", "j_per_token_idle_corrected", "s_per_token")


def dump() -> None:
    summary["cells"] = cells
    summary["elapsed_s"] = round(time.time() - T0)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))


def server_alive() -> bool:
    try:
        os.kill(PID, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run(tag, adapters, n, c, window, seed, weights=None, meta=None):
    out = RES / tag
    cmd = [PY, str(B / "run_cell.py"), "--server-pid", str(PID),
           "--base-url", URL, "--metrics-url", URL + "/metrics", "--model", BASE,
           "--adapters", *adapters, "--num-prompts", str(n), "--max-concurrency", str(c),
           "--input-len", str(plan.IN_LEN), "--output-len", str(plan.OUT_LEN),
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
    tokens = (s.get("completed") or 0) * plan.OUT_LEN
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
        **(meta or {}),
    }
    cells.append(row)
    dump()
    print(f"[{tag}] {row['verdict']} contained={row['contained']} active={active} "
          f"thr={row['output_throughput']} J/tok={row['j_per_token']} "
          f"hits={row['prefix_hits']} wall={row['wall_s']}s "
          f"fase={round(time.time() - T0)}s", flush=True)
    return row


def cell(x):
    s = SEEDS[x["tag"]]
    window = math.ceil(plan.WINDOW[x["c"]] * SCALE)
    meta = {"config": x["config"], "round": x["round"]}
    r = run(x["tag"], x["adapters"], plan.PROMPTS, x["c"], window, s["seed"],
            x["weights"], meta)
    if r is not None and r["verdict"] == "discard":
        if r["contained"]:
            r = run(x["tag"] + "-bis", x["adapters"], plan.PROMPTS, x["c"], window,
                    s["spare"], x["weights"], meta)
        else:
            r = run(x["tag"] + "-w2", x["adapters"], plan.PROMPTS, x["c"], window * 2,
                    s["spare"], x["weights"], meta)
    return r


def idle_power(when):
    if NO_GPU:
        res = {"secs": IDLE_SECS, "energy_mj": None, "watts": None, "power_mean_mw": None}
    else:
        p = subprocess.run(
            [str(B / "inferscope"), "--sample-only", "--pid", str(PID), "--include-descendants",
             "--duration-secs", str(IDLE_SECS), "--gpu", "--metrics-endpoint", URL + "/metrics",
             "--engine", "vllm", "--model", BASE, "--json"], capture_output=True, text=True)
        (RES / f"d8-idle-{when}.json").write_text(p.stdout)
        (RES / f"d8-idle-{when}.stderr.txt").write_text(p.stderr)
        ok = p.returncode == 0 and p.stdout.strip()
        g = (json.loads(p.stdout).get("gpu") or {}) if ok else {}
        e = g.get("energy_millijoules")
        res = {"secs": IDLE_SECS, "energy_mj": e,
               "watts": e / 1000 / IDLE_SECS if e is not None else None,
               "power_mean_mw": g.get("power_mean_milliwatts")}
    res["at_s"] = round(time.time() - T0)
    summary["idle"][when] = res
    dump()
    print(f"[d8-{when}] potenza a riposo {res['watts']} W (energia {res['energy_mj']} mJ "
          f"in {IDLE_SECS} s, media {res['power_mean_mw']} mW)", flush=True)
    return res


def stats(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return {"n": 0, "mean": None, "band": None}
    mean = sum(v) / len(v)
    band = (max(v) - min(v)) / mean if len(v) >= 2 and mean else None
    return {"n": len(v), "mean": mean, "band": band}


RES.mkdir(parents=True, exist_ok=True)

# La GPU della campagna, in results/ perche' entri nell'archivio con i risultati:
# gli stessi tre campi che phase0.sh legge, senza numero di serie ne' UUID.
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                        "--format=csv"], capture_output=True, text=True)
    why = f"codice {q.returncode}: {q.stderr.strip()[-200:]}"
except FileNotFoundError:
    q, why = None, "nvidia-smi non trovato"
if q is not None and q.returncode == 0:
    (RES / "gpu.csv").write_text(q.stdout)
    print("GPU:", " | ".join(q.stdout.strip().splitlines()), flush=True)
elif NO_GPU:
    print(f"GPU: gpu.csv non scritto ({why}; LORA_NO_GPU=1)", flush=True)
else:
    sys.exit(f"CAMPAGNA FERMA: la GPU non sarebbe registrata ({why})")

run("warmup", plan.ALL8, plan.WARMUP_PROMPTS, 8, SHORT_WINDOW, SEEDS["warmup"]["seed"])
idle_power("start")

kept: dict = {}
stopped = None
for i, x in enumerate(plan.RUNS):
    if stopped is None and time.time() - T0 > BUDGET_S:
        stopped = f"budget di fase di {BUDGET_S} s esaurito prima di {x['tag']}"
    if stopped is None and not server_alive():
        stopped = f"il server (pid {PID}) non esiste piu' prima di {x['tag']}"
    if stopped is not None:
        summary["not_run"].append(x["tag"])
        continue
    print(f"== cella {i + 1}/{len(plan.RUNS)}: {x['tag']}", flush=True)
    r = cell(x)
    if r is not None and r["verdict"] == "keep":
        kept.setdefault((x["config"], x["c"]), []).append(r)
    if x is plan.RUNS[len(plan.RUNS) // 2 - 1]:
        idle_power("middle")
if stopped is not None:
    summary["stopped"] = stopped
    print("CAMPAGNA INTERROTTA:", stopped, flush=True)
if server_alive():
    idle_power("end")

watts = [v["watts"] for v in summary["idle"].values() if v["watts"] is not None]
P = sum(watts) / len(watts) if watts else None
summary["idle_watts_used"] = P
summary["idle_spread"] = (max(watts) - min(watts)) / P if len(watts) >= 2 else None
for r in cells:
    if P is not None and r["energy_mj"] is not None and r["active_s"] and r["tokens"]:
        e_active = r["energy_mj"] / 1000 - P * (r["window"] - r["active_s"])
        r["j_per_token_idle_corrected"] = e_active / r["tokens"]

by_cell = {}
for (config, c), rows in sorted(kept.items(), key=lambda k: (-k[0][1], k[0][0])):
    by_cell[f"{config}-c{c}"] = {m: stats(rows, m) for m in METRICS}
summary["by_config"] = by_cell
margins = {}
for c in plan.CONCURRENCY:
    ref = by_cell.get(f"n1u-c{c}")
    for name, _, _ in plan.CONFIGS[1:]:
        cur = by_cell.get(f"{name}-c{c}")
        margins[f"{name}-c{c}"] = {
            m: ((cur[m]["mean"] - ref[m]["mean"]) / ref[m]["mean"])
            if ref and cur and ref[m]["mean"] and cur[m]["mean"] else None
            for m in METRICS}
summary["margin_vs_n1u"] = margins
dump()

print("\n== RIEPILOGO CAMPAGNA")
print("potenza a riposo (W):", {k: v["watts"] for k, v in summary["idle"].items()},
      "| usata:", P, "| escursione:", summary["idle_spread"])
for k, st in by_cell.items():
    print(k, " | ".join(f"{m}: n={st[m]['n']} media={st[m]['mean']} banda={st[m]['band']}"
                        for m in METRICS))
for k, mg in margins.items():
    print("margine", k, "vs n1u:", mg)
print("celle non eseguite:", summary["not_run"] or "nessuna")
print("CAMPAGNA", "INTERROTTA" if stopped else "COMPLETATA", "in", summary["elapsed_s"], "s",
      flush=True)
sys.exit(1 if stopped else 0)
