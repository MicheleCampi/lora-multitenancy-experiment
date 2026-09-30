"""Phase 2 - starts vllm serve on the completed protocol line, and checks it.

Blocking checks: the PID is the vllm process launched without a shell; the
models exposed are exactly the base and a1..a8; the children seen in
/proc/<pid>/task/<pid>/children match ps --ppid; every process that holds
the GPU is in the set inferscope measures; the prefix-cache counter exists;
inferscope --gpu returns energy.
The server is left running at the end. Writes ~/lora-run/session.json.

Before launching: if the port already answers, the phase stops without
touching anything, because the wait would accept another server's answer
(2026-09-26: "pronto in 0 s", ready in 0 s, from a server launched earlier).
session.json, server.pid and serve.log of a previous start are renamed with
a date and time, not overwritten.
"""
import json
import os
import pathlib
import socket
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
    print("PHASE2 FAILED:", msg, flush=True)
    sys.exit(1)


def cmdline(pid: int) -> str:
    raw = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def ps_children(pid: int) -> list[int]:
    out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)],
                         capture_output=True, text=True).stdout
    return sorted(int(x) for x in out.split())


if any(k.startswith("VLLM_") for k in os.environ):
    fail("VLLM_* variables in the environment")

if port_in_use(PORT):
    fail(f"port {PORT} already answers: a server is running "
         f"(ss -ltnp 'sport = :{PORT}'); nothing was launched or changed")

stamp = time.strftime("%Y%m%dT%H%M%S")
for old in (R / "session.json", R / "server.pid", R / "logs" / "serve.log"):
    if old.exists():
        kept = old.with_name(f"{old.name}.{stamp}")
        old.rename(kept)
        print("kept:", kept, flush=True)

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
print("vllm serve started, PID", pid, flush=True)

t0 = time.time()
models = None
while time.time() - t0 < 1800:
    if proc.poll() is not None:
        fail(f"vllm serve exited with code {proc.returncode}; log: {R / 'logs' / 'serve.log'}")
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=5) as r:
            models = json.load(r)
            break
    except OSError:
        time.sleep(2)
if models is None:
    fail("server not ready in 1800 s")
ready_s = round(time.time() - t0)
ids = sorted(m["id"] for m in models["data"])
print(f"ready in {ready_s} s; models: {ids}", flush=True)
if ids != sorted([base["repo"]] + NAMES):
    fail(f"models exposed differ from those expected: {ids}")

cmd = cmdline(pid)
print("PID cmdline:", cmd[:160], flush=True)
if "vllm" not in cmd:
    fail("the PID is not a vllm process")

try:
    kids = sorted(int(x) for x in
                  pathlib.Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
except OSError as e:
    fail(f"/proc/{pid}/task/{pid}/children unreadable: {e}")
ps_kids = ps_children(pid)
print("children (file):", kids, "| children (ps --ppid):", ps_kids, flush=True)
if kids != ps_kids:
    fail("the children file and ps --ppid differ: inferscope would not see every child")
if not kids:
    fail("the PID has no children: the EngineCore is not a direct child")
for k in kids:
    print(f"  child {k}: {cmdline(k)[:100]} | grandchildren: {ps_children(k)}", flush=True)

q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print("GPU processes:", q or "(none)", flush=True)
gpu_pids = sorted({int(line.split(",")[0]) for line in q.splitlines() if line.strip()})
if not gpu_pids:
    fail("nvidia-smi lists no GPU process")
outside = sorted(set(gpu_pids) - {pid, *kids})
if outside:
    fail(f"GPU processes outside the set inferscope measures: {outside}")

hits = counter_sum(METRICS, "vllm:prefix_cache_hits_total")
if hits is None:
    fail("vllm:prefix_cache_hits_total missing: the R30 check would not work")
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
    fail("inferscope --gpu returns no energy")

text = (R / "logs" / "serve.log").read_text(errors="replace")
d1 = [line.strip() for line in text.splitlines()
      if "Available KV cache memory" in line or "Maximum concurrency for" in line]
mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("D1 from the log:", *(d1 or ["(no line)"]), sep="\n  ")
print("GPU memory:", mem, flush=True)

session = {"pid": pid, "argv": argv, "models": ids, "ready_s": ready_s, "children": kids,
           "gpu_pids": gpu_pids, "gpu_apps": q, "d1_log": d1, "gpu_memory": mem,
           "smoke_energy_mj": gpu.get("energy_millijoules"),
           "smoke_energy_source": gpu.get("energy_source")}
(R / "session.json").write_text(json.dumps(session, indent=1))
print("PHASE2 OK", flush=True)
