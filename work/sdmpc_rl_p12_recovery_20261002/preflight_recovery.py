"""Check the actual failed UTF-8 metadata path without another physical rollout."""
import hashlib
import json
from pathlib import Path
import p12_worker as worker

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results/sdmpc_rl_p12_20261002"
OLD = REPO / "work/sdmpc_rl_p12_20261002"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / "preflight_recovery.json"
    if target.exists():
        raise FileExistsError("Preserve repair preflight")
    previous = json.loads((ROOT / "preflight.json").read_text(encoding="utf-8"))
    assert previous["status"] == "passed"
    assert sha(HERE / "bounded_policy.py") == sha(OLD / "bounded_policy.py")
    for path in (HERE / "specs").glob("*.json"):
        assert sha(path) == sha(OLD / "specs" / path.name)
    before = (OLD / "p12_worker.py").read_text(encoding="utf-8")
    after = (HERE / "p12_worker.py").read_text(encoding="utf-8")
    assert after == before.replace('(collector.OUT / "carry.json").read_text()',
                                   '(collector.OUT / "carry.json").read_text(encoding="utf-8")')
    for path in HERE.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    original = worker.collector.run_to_end
    samples = []
    try:
        for path in sorted((ROOT / "wave1").glob("*/carry.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if "reused_from" not in data:
                continue
            try:
                path.read_bytes().decode("cp949")
            except UnicodeDecodeError:
                pass
            else:
                raise AssertionError("Regression fixture must reproduce original decode error")
            worker.collector.OUT = path.parent
            sentinel = object()
            def capture(*args):
                assert args == (None, None, None, [], 123., 75)
                assert worker.collector.EXPECTED_TTT == data["ttt"]
                return sentinel
            worker.collector.run_to_end = capture
            assert worker.run_to_end(None, None, None, [], 123., 75) is sentinel
            samples.append(dict(path=str(path.relative_to(REPO)), sha256=sha(path), ttt=data["ttt"]))
    finally:
        worker.collector.run_to_end = original
    assert len(samples) == 4
    sources = {str(p.relative_to(REPO)): sha(p) for p in list(HERE.glob("*.py")) + list((HERE / "specs").glob("*.json"))}
    report = dict(status="passed", scope="UTF-8 carry metadata repair; original physical preflight reused",
                  original_preflight_sha256=sha(ROOT / "preflight.json"), policy_and_specs_byte_identical=True,
                  regression_samples=samples, sources=sources)
    with target.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
