"""Read-only admission evidence and exact parent review binding."""
import argparse
import json
import subprocess
import support as s
from policy import authenticate, parity


def centers():
    result = subprocess.run([s.sys.executable, "-B", str(s.HERE / "baselines.py")],
        cwd=s.REPO, env=dict(s.os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), capture_output=True, text=True,
        encoding="utf-8")
    if result.returncode:
        raise ValueError("Isolated center authentication failed:\n" + result.stderr)
    return json.loads(result.stdout)


def build():
    before = s.preservation()
    source = s.sources()
    actor, auth = authenticate()
    parity_result = parity(actor)
    baseline = centers()
    s.import_boundary()
    after = s.preservation()
    if after != before or source != s.sources():
        raise ValueError("Authentication changed preserved inputs or source")
    return dict(format=s.FORMAT+"-preflight", status="authenticated", spec=s.SPEC, spec_sha256=s.SPEC_SHA,
        sources=source, sources_sha256=s.digest(source), authentication=auth, parity=parity_result,
        centers=baseline, input_sha256=before, input_count=len(before), before_after_equal=True,
        actual_physical_steps=0, optimizer_updates=0, actual_evaluation_outputs_absent=not s.ROOT.exists())


def live(preflight):
    s.import_boundary()
    if (preflight["format"] != s.FORMAT+"-preflight" or preflight["status"] != "authenticated" or
            preflight["spec"] != s.SPEC or preflight["spec_sha256"] != s.SPEC_SHA or
            preflight["sources"] != s.sources() or preflight["sources_sha256"] != s.digest(s.sources()) or
            preflight["parity"]["observations"] != 375 or preflight["parity"]["bitexact"] is not True):
        raise ValueError("Preflight source/spec/parity identity differs")
    s.check_hash(s.MODEL / "completion.json", s.COMPLETION_SHA)
    s.check_hash(s.MODEL / "model_final.pt", s.MODEL_SHA)
    s.check_hash(s.GATE, s.GATE_SHA)
    s.check_hash(s.MODEL / "settings.json", s.MODEL_DONE["outputs_sha256"]["settings.json"])
    s.verify_files({name: sha for name, sha in s.DEPENDENCIES.items() if name.endswith(".py")})
    s.verify_files(preflight["centers"]["files"])
    s.old.verify_pins(s.DEFAULT_SNAPSHOT, preflight["authentication"]["contract"]["source_pins"])
    if s.old.runtime_versions() != preflight["authentication"]["contract"]["runtime_versions"]:
        raise ValueError("Runtime changed")


def admission(path, receipt_path):
    preflight, receipt = s.read(path), s.read(receipt_path)
    live(preflight)
    s.verify_files(preflight["input_sha256"])
    required = dict(format=s.FORMAT+"-parent-review", decision="approved_for_physical_evaluation",
        source_sha256=s.digest(s.sources()), spec_sha256=s.SPEC_SHA, preflight_sha256=s.file_hash(path),
        model_sha256=s.MODEL_SHA, physical_contract_sha256=s.CONTRACT_SHA)
    if (any(receipt.get(k) != v for k, v in required.items()) or
            not isinstance(receipt.get("reviewer"), str) or not receipt["reviewer"].strip() or
            not isinstance(receipt.get("parent_admission"), str) or not receipt["parent_admission"].strip()):
        raise ValueError("Exact source/spec/preflight parent review receipt required")
    return preflight, dict(path=str(s.Path(receipt_path).resolve()), sha256=s.file_hash(receipt_path), value=receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=s.Path)
    args = parser.parse_args()
    result = build()
    if args.output:
        s.save_once(args.output, result)
        print(json.dumps(dict(status=result["status"], input_count=result["input_count"],
            observations=result["parity"]["observations"], source_sha256=result["sources_sha256"],
            spec_sha256=s.SPEC_SHA, output=str(args.output), output_sha256=s.file_hash(args.output))))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

