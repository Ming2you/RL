"""Training-only probe adapter using unchanged P8 physics and checkpoint restore."""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "work/sdmpc_rl_probe_b_20260930"))
import probe


class RecoveryOption(probe.Option):
    def act(self, env, obs=None):
        if self.name != "p9_actor":
            return super().act(env, obs)
        if not hasattr(self, "actor"):
            # boot() must set numerical thread limits before importing NumPy.
            from actor import RecoveryActor
            self.actor = RecoveryActor.load(REPO / self.spec["spec_path"], env.observer.names)
            if self.actor.sha256 != self.spec["spec_sha256"]:
                raise ValueError("P9 spec hash mismatch")
        return self.actor.act(obs)


def run_to_end(env, obs, option, rows, deadline, max_step=75):
    terminal = False
    while not terminal and env.k - 5 < max_step:
        if any((p / "STOP").exists() for p in STOP_ROOTS):
            raise RuntimeError("STOP requested; durable completed branches preserved")
        if time.time() > deadline:
            raise TimeoutError("P9 wall-clock limit")
        action = option.act(env, obs)
        obs, _, terminal, row = env.step(action, "rl", 0., actor_cpu_seconds=0.)
        rows.append(probe.compact(row))
        if (env.k - 5) % 5 == 0:
            print(json.dumps(dict(event="progress", option=option.spec.get("tag", option.name),
                                  decision=env.k - 5, ttt=row["total_ttt"])), flush=True)
    if max_step == 75:
        if not terminal or env.k != 80 or len(rows) != 60:
            raise RuntimeError("P9 branch did not finish all 60 remaining intervals")
        if option.name == "carry" and rows[-1]["total_ttt"] != EXPECTED_TTT:
            raise RuntimeError("Restored carry does not exactly reproduce cache; aborting candidates")
    return obs


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    cache = Path(sys.argv[sys.argv.index("--checkpoint-dir") + 1]).resolve()
    EXPECTED_TTT = json.loads((cache / "carry.json").read_text(encoding="utf-8"))["ttt"]
    STOP_ROOTS = (REPO, REPO / "results/sdmpc_rl_balanced_goal_20260930",
                  REPO / "results/sdmpc_rl_machine_b_20260930", out.parent, out)
    probe.Option = RecoveryOption
    probe.run_to_end = run_to_end
    probe.main()
