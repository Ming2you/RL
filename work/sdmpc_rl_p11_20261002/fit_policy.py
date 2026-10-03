"""Fixed-budget fitting on the predeclared two P10 training seeds per scenario."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import torch
from neural_policy import FORMAT, network, NeuralPolicy

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results/sdmpc_rl_p10_20261001/wave2_recovery"
OUT = REPO / "results/sdmpc_rl_p11_20261002/fit_v1"
FIT = {"sweet_155_w": [8701, 8501], "sweet_170_w": [8702, 8502],
       "sweet_170_incident_w": [8703, 8503], "sweet_170_skew15_w": [8804, 8504],
       "sweet_190_w": [8705, 8505]}
CONFIG = dict(seed=11001, updates=2000, per_scenario_batch=20, temperature=.03,
    weight_clip=[.1, 10.], hidden=[64, 64], activation="tanh", lr=.0003,
    weight_decay=.001, gradient_clip=1., std_floor=.05, input_clip=10., window=[16, 30],
    after="return_both", algorithm="measured_return_weighted_behavioral_cloning")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stopped():
    return any((p / "STOP").exists() for p in (REPO, OUT, OUT.parent,
        REPO / "results/sdmpc_rl_balanced_goal_20260930", REPO / "results/sdmpc_rl_machine_b_20260930", DATA))


def main():
    if OUT.exists():
        raise FileExistsError("Preserve existing fit; do not overwrite or tune this run")
    if stopped():
        raise RuntimeError("STOP present")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG["seed"])
    rng = np.random.default_rng(CONFIG["seed"])
    report = read(Path(__file__).resolve().parent / "fit_manifest.json")
    if (report["status"] != "authenticated_fitting_split" or len(report["rows"]) != 40
            or report["analysis_sha256"] != sha(DATA / "analysis.json")
            or {(r["scenario"], r["seed"]) for r in report["rows"]}
               != {(s,n) for s, seeds in FIT.items() for n in seeds}):
        raise ValueError("Authenticated fitting-only manifest required")
    plan = read(DATA / "plan.json")
    for rel, digest in plan["sources"].items():
        if sha(REPO / rel) != digest:
            raise ValueError("P10 source identity changed")
    sys.path.insert(0, str(REPO / "work/sdmpc_rl_p10_20261001"))
    from p10_analysis import validate_experience
    provenance, groups, names = [], {}, None
    for scenario, seeds in FIT.items():
        features, labels, weights = [], [], []
        for seed in seeds:
            slot = DATA / f"{scenario}_s{seed}"
            carry = read(slot / "carry.json")
            for row in report["rows"]:
                if row["scenario"] != scenario or row["seed"] != seed:
                    continue
                tag = row["policy"]
                xp = slot / "experience" / f"{tag}.npz"
                meta = read(xp.with_suffix(".json"))
                branch = slot / "branches" / ("k16_carry.json" if tag == "carry" else f"k16_p10_actor_{tag}.json")
                if sha(xp) != row["experience_sha256"] or sha(branch) != row["sha256"]:
                    raise ValueError("Fitting input changed after authentication")
                if names is None:
                    names = meta["observation_names"]
                if names != meta["observation_names"]:
                    raise ValueError("Mixed observation schemas")
                with np.load(xp, allow_pickle=False) as a:
                    validate_experience(a, meta, read(branch), carry)
                    obs = a["obs"][:15].copy()
                    anchor = obs[0, [names.index("memory/action_anchor/0"), names.index("memory/action_anchor/1")]]
                    features.append(np.concatenate([obs, np.tile(anchor, (15, 1))], axis=1))
                    labels.append(a["action"][:15].copy())
                gain = (carry["ttt"] - row["ttt"]) / carry["ttt"]
                weight = float(np.exp(np.clip(gain / CONFIG["temperature"], np.log(.1), np.log(10.))))
                weights.extend([weight] * 15)
                provenance.append(dict(scenario=scenario, seed=seed, policy=tag,
                    experience_sha256=sha(xp), branch_sha256=sha(branch),
                    metadata_sha256=sha(xp.with_suffix(".json")), weight=weight))
        if len(features) != 8:
            raise ValueError("Eight fitting trajectories per scenario required")
        w = np.asarray(weights, np.float32); w /= w.mean()
        groups[scenario] = (np.concatenate(features), np.concatenate(labels), w)
    if len(provenance) != 40:
        raise ValueError("Unexpected fitting membership")
    all_x = np.concatenate([g[0] for g in groups.values()])
    mean = torch.from_numpy(all_x.mean(axis=0))
    scale = torch.from_numpy(np.maximum(all_x.std(axis=0), .05))
    data = {s: (((torch.from_numpy(x)-mean)/scale).clamp(-10., 10.),
                torch.from_numpy(y), torch.from_numpy(w)) for s, (x,y,w) in groups.items()}
    OUT.mkdir(parents=True)
    source_files = list(Path(__file__).resolve().parent.glob("*.py"))
    source_files += [Path(__file__).resolve().parent / "fit_manifest.json"]
    training_plan = dict(config=CONFIG, fitting_seeds=FIT, provenance=provenance,
        analysis_sha256=sha(DATA / "analysis.json"), source_pins={str(p.relative_to(REPO)): sha(p) for p in source_files},
        protocol_sha256=sha(REPO / "docs/rl_continuation_p11_20261002.md"),
        labels=600, complete_trajectories=40, holdout_loaded=False, started=time.time())
    (OUT / "plan.json").write_text(json.dumps(training_plan, indent=2), encoding="utf-8")
    net = network()
    opt = torch.optim.AdamW(net.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    counts = dict.fromkeys(FIT, 0)
    def full_loss():
        with torch.no_grad():
            return float(torch.stack([(((net(x)-y)**2).mean(1)*w).mean() for x,y,w in data.values()]).mean())
    initial = full_loss()
    for update in range(1, CONFIG["updates"] + 1):
        if stopped():
            raise RuntimeError("STOP during fit; no policy released")
        losses = []
        for s, (x,y,w) in data.items():
            idx = rng.integers(0, len(x), CONFIG["per_scenario_batch"])
            losses.append((((net(x[idx])-y[idx])**2).mean(1)*w[idx]).mean())
            counts[s] += len(idx)
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
        opt.step()
        if update % 250 == 0:
            print(json.dumps(dict(update=update, fitting_loss=full_loss())), flush=True)
    final = full_loss()
    payload = dict(format=FORMAT, observation_names=names, mean=mean, scale=scale,
        state_dict=net.state_dict(), config=CONFIG, provenance=provenance)
    target = OUT / "model.pt"
    torch.save(payload, target)
    spec = dict(format=FORMAT, model_path=str(target.relative_to(REPO)).replace('\\','/'),
                model_sha256=sha(target), window=[16,30], after="return_both")
    (OUT / "policy.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    loaded = NeuralPolicy.load(OUT / "policy.json", names)
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, loaded.net.state_dict()[key], rtol=0, atol=0)
    test = torch.from_numpy(all_x[:15])
    with torch.no_grad():
        torch.testing.assert_close(net(((test-mean)/scale).clamp(-10.,10.)),
            loaded.net(((test-loaded.mean)/loaded.scale).clamp(-10.,10.)), rtol=0, atol=0)
    passed = bool(np.isfinite(final) and final < .8 * initial)
    receipt = dict(status="fit_passed" if passed else "fit_failed", optimizer_updates=CONFIG["updates"],
        initial_zero_action_mse=initial, final_weighted_mse=final, sampled_per_scenario=counts,
        model_sha256=sha(target), model_changed=any(torch.count_nonzero(v).item() for v in list(net[-2].parameters())),
        holdout_loaded=False, canonical_loaded=False, torch_threads=torch.get_num_threads(),
        labels=600, full_training_trajectories=40, ended=time.time(), gate="fitting only; rollout performance untested")
    (OUT / "completion.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
