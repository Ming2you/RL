"""P10 training branches with durable observation/action/reward records."""
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "work/sdmpc_rl_probe_b_20260930"))
import probe


class P10Option(probe.Option):
    def act(self, env, obs=None):
        if self.name != "p10_actor":
            return super().act(env, obs)
        if not hasattr(self, "actor"):
            # Delay NumPy import until boot has set all numerical thread limits.
            from retention_actor import RetentionActor
            self.actor = RetentionActor.load(REPO / self.spec["spec_path"], env.observer.names)
            if self.actor.sha256 != self.spec["spec_sha256"]:
                raise ValueError("P10 spec identity mismatch")
        return self.actor.act(obs)


def run_to_end(env, obs, option, rows, deadline, max_step=75):
    import numpy as np
    if max_step != 75:
        raise ValueError("P10 only accepts true-terminal full continuations")
    sequence = []
    terminal = False
    while not terminal:
        if any((p / "STOP").exists() for p in STOP_ROOTS):
            raise RuntimeError("STOP requested; completed branches preserved")
        if time.time() > deadline:
            raise TimeoutError("P10 wall-clock limit")
        before = obs.copy()
        memory = option.actor.memory() if hasattr(option, "actor") else np.zeros(4, np.float32)
        action = option.act(env, obs)
        memory_next = option.actor.memory() if hasattr(option, "actor") else np.zeros(4, np.float32)
        obs, reward, terminal, row = env.step(action, "rl", 0., actor_cpu_seconds=0.)
        rows.append(probe.compact(row))
        sequence.append((before, action.copy(), reward, obs.copy(), terminal, memory, memory_next))
        if (env.k - 5) % 5 == 0:
            print(json.dumps(dict(event="progress", option=option.spec.get("tag", option.name),
                                  decision=env.k - 5, ttt=row["total_ttt"])), flush=True)
    if env.k != 80 or len(rows) != 60 or env.sim.state.time_sec != 14400.:
        raise RuntimeError("P10 branch is not a complete 60-interval continuation")
    if option.name == "carry" and rows[-1]["total_ttt"] != EXPECTED_TTT:
        raise RuntimeError("Carry cache reproduction failed; candidate dispatch aborted")
    tag = option.spec.get("tag", option.name)
    folder = OUT / "experience"
    folder.mkdir(exist_ok=True)
    target = folder / f"{tag}.npz"
    if target.exists():
        raise RuntimeError("Experience already exists; inspect before resuming")
    names = ("obs", "action", "reward", "next_obs", "terminal", "memory", "next_memory")
    arrays = {name: np.asarray([r[i] for r in sequence]) for i, name in enumerate(names)}
    temporary = target.with_suffix(".tmp")
    with open(temporary, "xb") as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(target)
    meta = dict(format="sdmpc-training-continuation-p10-v1", scenario=SCENARIO,
        training_seed=env.training_seed, profile_sha256=env.profile_hash, option=option.spec,
        source_kind="training_only", first_decision=16, terminal_decision=75, transitions=60,
        reward_divisor=env.reward_scale, gamma=1., ttt=rows[-1]["total_ttt"],
        observation_names=list(env.observer.names), memory_names=["has_anchor", "NP_anchor/1000", "NUF_anchor/10000", "aborted"],
        file_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    probe.save(folder / f"{tag}.json", meta)
    return obs


if __name__ == "__main__":
    OUT = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    SCENARIO = sys.argv[sys.argv.index("--scenario") + 1]
    cache = Path(sys.argv[sys.argv.index("--checkpoint-dir") + 1]).resolve()
    EXPECTED_TTT = json.loads((cache / "carry.json").read_text())["ttt"]
    STOP_ROOTS = (REPO, REPO / "results/sdmpc_rl_balanced_goal_20260930",
                  REPO / "results/sdmpc_rl_machine_b_20260930", OUT.parent, OUT)
    probe.Option = P10Option
    probe.run_to_end = run_to_end
    probe.main()
