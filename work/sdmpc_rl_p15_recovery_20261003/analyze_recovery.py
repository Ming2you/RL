"""Apply the unchanged P15 full analyzer after checking recovery provenance."""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
WAVE = REPO / "results/sdmpc_rl_p15_20261002/wave2_recovery"


def main():
    plan = json.loads((WAVE / "plan.json").read_text(encoding="utf-8"))
    old = Path(plan["recovery_from"])
    for rel,digest in plan["recovered_files"].items():
        for wave in (WAVE,old):
            assert hashlib.sha256((wave / rel).read_bytes()).hexdigest() == digest
    sys.path.insert(0,str(REPO / "work/sdmpc_rl_p15_20261002"))
    import analyze_wave
    analyze_wave.WAVE = WAVE
    analyze_wave.main()


if __name__ == "__main__":
    main()
