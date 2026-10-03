"""P11 neural pilot through the unchanged P10 collector and frozen simulator."""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "work/sdmpc_rl_p10_20261001"))
import p10_worker as collector


class NeuralOption(collector.probe.Option):
    def act(self, env, obs=None):
        if self.name != "p11_actor":
            return super().act(env, obs)
        if not hasattr(self, "actor"):
            from neural_policy import NeuralPolicy
            spec_path = REPO / self.spec["spec_path"]
            if hashlib.sha256(spec_path.read_bytes()).hexdigest() != self.spec["spec_sha256"]:
                raise ValueError("P11 policy spec changed")
            self.actor = NeuralPolicy.load(spec_path, env.observer.names)
        return self.actor.act(obs)


def run_to_end(env, obs, option, rows, deadline, max_step=75):
    # probe.main has already finished or loaded the full carry at this point.
    collector.EXPECTED_TTT = json.loads((collector.OUT / "carry.json").read_text())["ttt"]
    return collector.run_to_end(env, obs, option, rows, deadline, max_step)


original_compact = collector.probe.compact


def checked_compact(row):
    # The original carry-generation loop does not call run_to_end.
    if any((root / "STOP").exists() for root in collector.STOP_ROOTS):
        raise RuntimeError("STOP requested during physical interval; completed files preserved")
    return original_compact(row)


if __name__ == "__main__":
    collector.OUT = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    collector.SCENARIO = sys.argv[sys.argv.index("--scenario") + 1]
    collector.STOP_ROOTS = (REPO, REPO / "results/sdmpc_rl_balanced_goal_20260930",
        REPO / "results/sdmpc_rl_machine_b_20260930", collector.OUT,
        collector.OUT.parent, collector.OUT.parent.parent)
    collector.probe.Option = NeuralOption
    collector.probe.run_to_end = run_to_end
    collector.probe.compact = checked_compact
    collector.probe.main()
