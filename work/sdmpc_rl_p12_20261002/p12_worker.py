"""P12 branch adapter; unchanged P10 collector and frozen physical runtime."""
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO / "work/sdmpc_rl_p10_20261001"))
import p10_worker as collector


class BoundedOption(collector.probe.Option):
    def act(self,env,obs=None):
        if self.name!="p12_actor":
            return super().act(env,obs)
        if not hasattr(self,"actor"):
            from bounded_policy import BoundedPolicy
            path=REPO / self.spec["spec_path"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.spec["spec_sha256"]:
                raise ValueError("P12 spec changed")
            self.actor=BoundedPolicy.load(path,env.observer.names)
        return self.actor.act(obs)


def run_to_end(env,obs,option,rows,deadline,max_step=75):
    collector.EXPECTED_TTT=json.loads((collector.OUT / "carry.json").read_text())["ttt"]
    return collector.run_to_end(env,obs,option,rows,deadline,max_step)


if __name__=="__main__":
    collector.OUT=Path(sys.argv[sys.argv.index("--output")+1]).resolve()
    collector.SCENARIO=sys.argv[sys.argv.index("--scenario")+1]
    cache=Path(sys.argv[sys.argv.index("--checkpoint-dir")+1]).resolve()
    if not all((cache / f).is_file() for f in ("carry.json","k16.pt")):
        raise ValueError("P12 requires authenticated existing carry/checkpoint")
    collector.STOP_ROOTS=(REPO,REPO / "results/sdmpc_rl_balanced_goal_20260930",
        REPO / "results/sdmpc_rl_machine_b_20260930",collector.OUT,collector.OUT.parent,collector.OUT.parent.parent)
    collector.probe.Option=BoundedOption
    collector.probe.run_to_end=run_to_end
    collector.probe.main()
