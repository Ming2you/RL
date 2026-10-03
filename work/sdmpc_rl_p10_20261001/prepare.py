"""Create the predeclared P10 specs; run only before source freezing/dispatch."""
import json
from pathlib import Path
from retention_actor import spec_sha256

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    definitions = [("b1_both_w25", 25, False), ("long_both_w30", 30, False),
                   ("retain_nuf_w30", 30, True)]
    base = json.loads((REPO / "work/sdmpc_rl_p9_20261001/specs/long_both_return.json").read_text())
    (HERE / "specs").mkdir(exist_ok=True)
    options = []
    for tag, end, retain in definitions:
        recovery = json.loads(json.dumps(base))
        recovery["base"]["params"]["bind_end"] = end
        spec = dict(format="sdmpc-retention-p10-v1", recovery=recovery, retain_nuf_during_bind=retain)
        path = HERE / "specs" / f"{tag}.json"
        if path.exists():
            raise RuntimeError("Refusing to overwrite an existing P10 spec")
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        options.append(dict(name="p10_actor", tag=tag, at=16, spec_path=path.relative_to(REPO).as_posix(),
                            spec_sha256=spec_sha256(spec)))
    (HERE / "options.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
    print(json.dumps(options, indent=2))


if __name__ == "__main__":
    main()
