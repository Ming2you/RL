"""Preserve P9-P17 results and models as a verified split ZIP; standard library only.

Run from the repository root with build, verify or restore. Restore writes missing
files only and refuses changed existing files. Nothing is unpickled or executed.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

REPO=Path(__file__).resolve().parents[1]
OUT=REPO / "artifacts/sdmpc_continuation_20261003"
ROOTS=["sdmpc_rl_p9_20261001","sdmpc_rl_p10_20261001","sdmpc_rl_p11_20261002",
       "sdmpc_rl_p12_20261002","sdmpc_rl_p13_20261002","sdmpc_rl_p14_20261002",
       "sdmpc_rl_p15_20261002","sdmpc_rl_p16_20261003","sdmpc_rl_p17_20261003",
       "sdmpc_rl_p17_v2_20261003"]
PART=40*1024*1024


def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))


def target_for(name):
    rel=PurePosixPath(name)
    if (rel.is_absolute() or len(rel.parts)<3 or rel.parts[0]!="results"
            or rel.parts[1] not in ROOTS or any(p in (".","..") or ":" in p or "\\" in p for p in rel.parts)):
        raise ValueError("Unexpected archive path: "+name)
    dest=(REPO / Path(*rel.parts)).resolve()
    if not dest.is_relative_to(REPO.resolve()):raise ValueError("Path escapes repository")
    return dest


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT / "manifest.json").exists() or list(OUT.glob("research.zip.part*")):
        raise FileExistsError("Preserve existing archive")
    inventory=[];buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for root_name in ROOTS:
            root=REPO / "results" / root_name
            if not root.is_dir():raise FileNotFoundError(root)
            for p in sorted(root.rglob("*")):
                if not p.is_file():continue
                if p.suffix in (".lock",".tmp",".pyc") or "__pycache__" in p.parts:continue
                if p.is_symlink():raise ValueError("Do not archive symlinks")
                name=p.relative_to(REPO).as_posix();target_for(name)
                data=p.read_bytes()
                inventory.append(dict(path=name,bytes=len(data),sha256=sha(data)))
                z.writestr(name,data)
                if sha(p.read_bytes())!=sha(data):raise ValueError("Source changed while archiving: "+name)
    blob=buffer.getvalue();parts=[]
    for i in range(0,len(blob),PART):
        name=f"research.zip.part{i//PART+1:03d}";chunk=blob[i:i+PART]
        with (OUT / name).open("xb") as f:f.write(chunk)
        parts.append(dict(name=name,bytes=len(chunk),sha256=sha(chunk)))
    with (OUT / "inventory.json").open("x",encoding="utf-8") as f:json.dump(inventory,f,indent=1)
    manifest=dict(format="sdmpc-continuation-split-zip-v1",roots=ROOTS,files=len(inventory),
        original_bytes=sum(r["bytes"] for r in inventory),zip_bytes=len(blob),zip_sha256=sha(blob),
        inventory_sha256=sha((OUT / "inventory.json").read_bytes()),parts=parts,
        excluded="queue locks, temporary files and Python bytecode only",
        scope="All local P9-P17 result roots, including models, experiences, caches and incomplete-wave evidence")
    with (OUT / "manifest.json").open("x",encoding="utf-8") as f:json.dump(manifest,f,indent=1)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("roots","parts")} | dict(parts=len(parts)),indent=2))


def load():
    manifest=read(OUT / "manifest.json")
    if manifest["format"]!="sdmpc-continuation-split-zip-v1" or manifest["roots"]!=ROOTS:raise ValueError("Unexpected manifest")
    blocks=[]
    for i,part in enumerate(manifest["parts"],1):
        if part["name"]!=f"research.zip.part{i:03d}":raise ValueError("Unexpected part name")
        data=(OUT / part["name"]).read_bytes()
        if len(data)!=part["bytes"] or sha(data)!=part["sha256"]:raise ValueError("Part differs")
        blocks.append(data)
    blob=b"".join(blocks)
    if len(blob)!=manifest["zip_bytes"] or sha(blob)!=manifest["zip_sha256"]:raise ValueError("ZIP differs")
    raw=(OUT / "inventory.json").read_bytes()
    if sha(raw)!=manifest["inventory_sha256"]:raise ValueError("Inventory differs")
    inventory=json.loads(raw);names=[r["path"] for r in inventory]
    z=zipfile.ZipFile(io.BytesIO(blob))
    if len(names)!=manifest["files"] or len(set(names))!=len(names) or z.namelist()!=names:raise ValueError("Member set differs")
    for row in inventory:
        target_for(row["path"])
        data=z.read(row["path"])
        if len(data)!=row["bytes"] or sha(data)!=row["sha256"]:raise ValueError("Member differs: "+row["path"])
    return z,inventory


def verify():
    z,inventory=load();z.close()
    print(json.dumps(dict(verified_files=len(inventory),written_files=0,mode="verify")))


def restore():
    z,inventory=load();missing=[]
    for row in inventory:
        dest=target_for(row["path"])
        if dest.exists():
            if not dest.is_file() or sha(dest.read_bytes())!=row["sha256"]:
                raise ValueError("Existing file differs; refusing: "+str(dest))
        else:missing.append((row,dest))
    for row,dest in missing:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("xb") as f:f.write(z.read(row["path"]))
    z.close()
    print(json.dumps(dict(verified_files=len(inventory),written_files=len(missing),mode="restore")))


if __name__=="__main__":
    {"build":build,"verify":verify,"restore":restore}[sys.argv[1]]()
