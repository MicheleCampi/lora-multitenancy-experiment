"""Fase 2 - avvia vllm serve con la riga del protocollo completata, e verifica.

Controlli bloccanti: il PID e' il processo vllm lanciato senza shell; i modelli
esposti sono esattamente il base e a1..a8; i figli visti da
/proc/<pid>/task/<pid>/children coincidono con ps --ppid; ogni processo che
tiene la GPU sta nell'insieme che inferscope misura; il contatore della cache
dei prefissi esiste; inferscope --gpu restituisce energia.
Il server resta acceso alla fine. Scrive ~/lora-run/session.json.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

B = pathlib.Path(__file__).resolve().parent
R = pathlib.Path.home() / "lora-run"
V = R / "venv" / "bin"
sys.path.insert(0, str(B))
from run_cell import counter_sum  # noqa: E402

PORT = 8000
URL = f"http://127.0.0.1:{PORT}"
METRICS = URL + "/metrics"
NAMES = [f"a{i}" for i in range(1, 9)]


def fail(msg: str) -> None:
    print("FASE2 FALLITA:", msg, flush=True)
    sys.exit(1)


def cmdline(pid: int) -> str:
    raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def ps_children(pid: int) -> list[int]:
    out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)],
                         capture_output=True, text=True).stdout
    return sorted(int(x) for x in out.split())


if any(k.startswith("VLLM_") for k in os.environ):
    fail("variabili VLLM_* nell'ambiente")

pins = json.loads((R / "pins.json").read_text())
base = pins["base"]
argv = [
    str(V / "vllm"), "serve", base["repo"],
    "--revision", base["revision"], "--tokenizer-revision", base["revision"],
    "--enable-lora", "--max-loras", "8", "--max-cpu-loras", "8",
    "--max-lora-rank", "16", "--lora-dtype", "auto", "--enforce-eager",
    "--data-parallel-size", "1", "--api-server-count", "1",
    "--gpu-memory-utilization", "0.92", "--max-model-len", "512",
    "--port", str(PORT),
    "--lora-modules", *[json.dumps({"name": n, "path": pins[n]["path"]}) for n in NAMES],
]
env = dict(os.environ, PATH=f"{V}:{os.environ['PATH']}")
log = open(R / "logs" / "serve.log", "w")
proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                        start_new_session=True)
pid = proc.pid
(R / "server.pid").write_text(str(pid))
print("vllm serve avviato, PID", pid, flush=True)

t0 = time.time()
models = None
while time.time() - t0 < 1800:
    if proc.poll() is not None:
        fail(f"vllm serve terminato con codice {proc.returncode}; log: {R / 'logs' / 'serve.log'}")
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=5) as r:
            models = json.load(r)
            break
    except OSError:
        time.sleep(2)
if models is None:
    fail("server non pronto in 1800 s")
ready_s = round(time.time() - t0)
ids = sorted(m["id"] for m in models["data"])
print(f"pronto in {ready_s} s; modelli: {ids}", flush=True)
if ids != sorted([base["repo"]] + NAMES):
    fail(f"modelli esposti diversi dall'atteso: {ids}")

cmd = cmdline(pid)
print("cmdline del PID:", cmd[:160], flush=True)
if "vllm" not in cmd:
    fail("il PID non e' un processo vllm")

try:
    kids = sorted(int(x) for x in
                  pathlib.Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
except OSError as e:
    fail(f"/proc/{pid}/task/{pid}/children illeggibile: {e}")
ps_kids = ps_children(pid)
print("figli (children):", kids, "| figli (ps --ppid):", ps_kids, flush=True)
if kids != ps_kids:
    fail("il file children e ps --ppid non coincidono: inferscope non vedrebbe tutti i figli")
if not kids:
    fail("il PID non ha figli: l'EngineCore non e' un figlio diretto")
for k in kids:
    print(f"  figlio {k}: {cmdline(k)[:100]} | nipoti: {ps_children(k)}", flush=True)

q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print("processi GPU:", q or "(nessuno)", flush=True)
gpu_pids = sorted({int(line.split(",")[0]) for line in q.splitlines() if line.strip()})
if not gpu_pids:
    fail("nvidia-smi non elenca processi GPU")
outside = sorted(set(gpu_pids) - {pid, *kids})
if outside:
    fail(f"processi GPU fuori dall'insieme misurato da inferscope: {outside}")

hits = counter_sum(METRICS, "vllm:prefix_cache_hits_total")
if hits is None:
    fail("vllm:prefix_cache_hits_total assente: il controllo di R30 non funzionerebbe")
print("prefix_cache_hits_total:", hits, flush=True)

smoke = subprocess.run(
    [str(B / "inferscope"), "--sample-only", "--pid", str(pid), "--include-descendants",
     "--duration-secs", "3", "--gpu", "--metrics-endpoint", METRICS, "--engine", "vllm",
     "--model", base["repo"], "--json"], capture_output=True, text=True)
if smoke.returncode != 0:
    fail(f"inferscope rc {smoke.returncode}: {smoke.stderr[-300:]}")
report = json.loads(smoke.stdout)
gpu = report.get("gpu") or {}
print("inferscope --gpu 3 s: energy_millijoules", gpu.get("energy_millijoules"),
      "| energy_source", gpu.get("energy_source"), flush=True)
if gpu.get("energy_millijoules") is None:
    fail("inferscope --gpu non restituisce energia")

text = (R / "logs" / "serve.log").read_text(errors="replace")
d1 = [line.strip() for line in text.splitlines()
      if "Available KV cache memory" in line or "Maximum concurrency for" in line]
mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("D1 dal log:", *(d1 or ["(nessuna riga)"]), sep="\n  ")
print("memoria GPU:", mem, flush=True)

session = {"pid": pid, "argv": argv, "models": ids, "ready_s": ready_s, "children": kids,
           "gpu_pids": gpu_pids, "gpu_apps": q, "d1_log": d1, "gpu_memory": mem,
           "smoke_energy_mj": gpu.get("energy_millijoules"),
           "smoke_energy_source": gpu.get("energy_source")}
(R / "session.json").write_text(json.dumps(session, indent=1))
print("FASE2 OK", flush=True)
