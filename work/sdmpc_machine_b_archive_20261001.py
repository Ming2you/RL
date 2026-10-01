"""Split-ZIP archive of the machine-B results (selected raw files), with inventory and manifest.

Usage (repository root):
    python -m work.sdmpc_machine_b_archive_20261001 build      # create artifacts/sdmpc_machine_b_20261001
    python -m work.sdmpc_machine_b_archive_20261001 verify     # check parts, full ZIP and every member hash
    python -m work.sdmpc_machine_b_archive_20261001 restore    # extract missing files only; refuse differing ones
Standard library only. It never runs RL, never unpickles anything and never overwrites a differing file.
Excluded on purpose (large, redundant, reproducible): per-interval evaluator checkpoints (*/checkpoints/*),
runner locks, and the machine-B center checkpoint.pt files. Everything else under the machine-B
result root and the probe checkpoint cache on D: is archived with its relative path and SHA-256.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts/sdmpc_machine_b_20261001"
SOURCES = [("results/sdmpc_rl_machine_b_20260930", REPO / "results/sdmpc_rl_machine_b_20260930"),
           ("probe_checkpoint_cache", Path("D:/RL_data/sdmpc_rl_machine_b_20260930/ckpt"))]
PART = 40 * 1024 * 1024


def sha(path_or_bytes):
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def excluded(rel):
    parts = rel.split("/")
    return ("checkpoints" in parts or rel.endswith("runner.lock") or rel.endswith(".tmp") or
            (rel.endswith("checkpoint.pt") and "center_repro_v1" in rel))


def files():
    rows = []
    for prefix, root in SOURCES:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rel = f"{prefix}/{p.relative_to(root).as_posix()}"
                if not excluded(rel):
                    rows.append((rel, p))
    return rows


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.glob("research.zip.part*")):
        raise SystemExit("Archive parts already exist; refusing to overwrite")
    inventory, buffer = [], io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel, p in files():
            data = p.read_bytes()
            inventory.append(dict(path=rel, bytes=len(data), sha256=sha(data)))
            z.writestr(rel, data)
    blob = buffer.getvalue()
    parts = []
    for i in range(0, len(blob), PART):
        name = f"research.zip.part{i // PART + 1:03d}"
        (OUT / name).write_bytes(blob[i:i + PART])
        parts.append(dict(name=name, bytes=len(blob[i:i + PART]), sha256=sha(blob[i:i + PART])))
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=1), encoding="utf-8")
    manifest = dict(zip_sha256=sha(blob), zip_bytes=len(blob), parts=parts, files=len(inventory),
                    original_bytes=sum(r["bytes"] for r in inventory), inventory_sha256=sha(OUT / "inventory.json"),
                    excluded="per-interval evaluator checkpoints, runner locks, machine-B center checkpoint.pt")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "parts"} | dict(parts=len(parts)), indent=1))


def load():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    blob = b""
    for part in manifest["parts"]:
        data = (OUT / part["name"]).read_bytes()
        if sha(data) != part["sha256"]:
            raise SystemExit("Part hash differs: " + part["name"])
        blob += data
    if sha(blob) != manifest["zip_sha256"]:
        raise SystemExit("Full ZIP hash differs")
    inventory = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))
    if sha(OUT / "inventory.json") != manifest["inventory_sha256"]:
        raise SystemExit("Inventory hash differs")
    return zipfile.ZipFile(io.BytesIO(blob)), inventory


def verify():
    z, inventory = load()
    for row in inventory:
        if sha(z.read(row["path"])) != row["sha256"]:
            raise SystemExit("Member hash differs: " + row["path"])
    print(json.dumps(dict(verified_files=len(inventory), mode="verify", written_files=0)))


def restore():
    z, inventory = load()
    targets = {"results/sdmpc_rl_machine_b_20260930": REPO / "results/sdmpc_rl_machine_b_20260930",
               "probe_checkpoint_cache": REPO / "results/sdmpc_rl_machine_b_20260930/probe_checkpoint_cache"}
    plan = []
    for row in inventory:
        prefix, rest = row["path"].split("/", 1)
        dest = (targets[prefix] / rest).resolve()
        if not dest.is_relative_to(REPO.resolve()):
            raise SystemExit("Path escapes the repository: " + row["path"])
        if dest.exists() and sha(dest) != row["sha256"]:
            raise SystemExit("Existing file differs, refusing: " + str(dest))
        plan.append((row, dest))
    written = 0
    for row, dest in plan:
        if not dest.exists():
            data = z.read(row["path"])
            if sha(data) != row["sha256"]:
                raise SystemExit("Member hash differs: " + row["path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            written += 1
    print(json.dumps(dict(verified_files=len(inventory), mode="restore", written_files=written)))


if __name__ == "__main__":
    {"build": build, "verify": verify, "restore": restore}[sys.argv[1]]()
