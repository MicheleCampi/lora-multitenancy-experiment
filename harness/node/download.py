"""Download pinnati: modello base, otto adattatori e due riserve.

Il modello base si scarica per intero alla sua revisione; degli adattatori
solo i due file della radice (PROTOCOL.md 339-342), perche' alcuni repository
pubblicano anche checkpoint-*, gguf/ e merged/. Ogni percorso restituito deve
terminare con snapshots/<sha>: e' la prova che il pin scaricato e' quello
dichiarato. Scrive ~/lora-run/pins.json (nome -> repository, revisione, percorso).
"""
import json
import pathlib

from huggingface_hub import snapshot_download

BASE = ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")
ADAPTERS = [
    ("a1", "codewithdark/mlpr-qwen2.5-7b-instruct-50ep-adaptive", "3d55449ee2e2c886138f181f905f37afe2672d28"),
    ("a2", "namanadep/Qwen2.5-7B-Manus-Distill", "097382c877a5669ff4d720725759ccfd0edd82bb"),
    ("a3", "Manikanta23/qwen2.5-7b-rtl-vlsi-lora", "af13b834cc11a2972356570bf0eb0190dea017ea"),
    ("a4", "MirzaJunaid/Amna-AI", "873d03b5843df7e214957ffc6630ae6302956ab5"),
    ("a5", "Tamir39/qwen2_5-7b-vietnam-tax-lora", "4b46d018068369d0a2361b7b8eed19cc22b9beaf"),
    ("a6", "millat/Qwen2.5-7B-BDLAW-LoRA", "73dc7a0cc673d4f1e9d9c1116d435ec376c8d9e5"),
    ("a7", "keer2004ks/ade-lora-adapter", "195ae4711d262f8d08a397f7c476773cc2f85f69"),
    ("a8", "ritam-05/qwen2.5-7b-sql-specialist", "0505bdf624ae32d3d7a571135a68100437a0d8ea"),
    ("r1", "diegogs1451/qwen2.5-7B-Instruct-dUO-finetuned-20260706-3epochs", "dfc855dfad125ad64f6d7012458b6d18a55b18d5"),
    ("r2", "vidyaganga/slytherin-loyalty-organism-v21", "1074dad1bd05b3d5ebdbe27fb4fc8aaad48e6c43"),
]
ROOT_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def pinned(path: str, sha: str) -> pathlib.Path:
    p = pathlib.Path(path)
    assert p.parts[-2:] == ("snapshots", sha), f"percorso non pinnato: {p}"
    return p


pins = {}
repo, sha = BASE
base = pinned(snapshot_download(repo, revision=sha), sha)
assert any(base.glob("*.safetensors")), "nessun safetensors nel modello base"
pins["base"] = {"repo": repo, "revision": sha, "path": str(base)}
print("base", repo, sha, base, flush=True)

for name, repo, sha in ADAPTERS:
    p = pinned(snapshot_download(repo, revision=sha, allow_patterns=ROOT_FILES), sha)
    for f in ROOT_FILES:
        assert (p / f).is_file(), f"{name}: manca {f}"
    cfg = json.loads((p / "adapter_config.json").read_text())
    assert cfg.get("r") == 16, (name, "r", cfg.get("r"))
    assert not cfg.get("modules_to_save"), (name, "modules_to_save")
    assert not cfg.get("use_dora"), (name, "use_dora")
    pins[name] = {"repo": repo, "revision": sha, "path": str(p)}
    print(name, repo, sha, p, flush=True)

out = pathlib.Path.home() / "lora-run" / "pins.json"
out.write_text(json.dumps(pins, indent=1))
print("DOWNLOAD OK", out, flush=True)
