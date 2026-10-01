"""Isolated read-only legacy center authentication; never boot or unpickle physics."""
import json
import support as s
import checks
from compare_runs import load_completed_run


def validate_center(result, settings, traces, schema, scenario, contract, physical):
    if (settings["scenario"] != scenario or settings["mode"] != "center" or settings["seeds"] != [None] or
            settings["source_pins"] != contract["source_pins"] or
            settings["runtime_versions"] != contract["runtime_versions"] or
            settings["environment_contract"] != contract["environment_contract"] or
            schema != contract["observation_schema"] or settings["model_sha256"] is not None):
        raise ValueError("Matched center source/runtime/environment identity differs")
    episode, trace = result["episodes"][0], traces[0]
    s.same_cost(episode["ttt"], s.BASE_TTT[scenario], "canonical baseline TTT")
    canonical = next(r for r in contract["environment_contract"]["scenarios"] if r["scenario"] == scenario)
    if episode["profile_sha256"] != canonical["profile_sha256"]:
        raise ValueError("Matched center canonical profile differs")
    settings = dict(settings, contract=contract, capacity=6000., warmup_ttt=episode["warmup_ttt"],
                    profile_sha256=episode["profile_sha256"], **physical)
    checks.physical_rows(trace, settings)
    for i, row in enumerate(trace):
        s.np.testing.assert_array_equal(row["action_requested"], [0., 0.])
        if i:
            s.np.testing.assert_array_equal(row["action_anchor"], trace[i-1]["B_executed"])
        raw, request = s.old.residual_budget(s.np.zeros(2, dtype=s.np.float32), row["action_anchor"], 6000.)
        s.np.testing.assert_array_equal(raw, row["B_raw"])
        s.np.testing.assert_array_equal(request, row["B_requested"][0])
    summary = checks.summarize(trace, settings, dict(elapsed_wall_seconds=result["elapsed_wall_seconds"],
        scope="legacy center completion elapsed, includes reset/recovery/PFO/reference/actor/lower/guard"))
    summary.update(model_sha256=None, behavior="carry_center")
    return summary


def main():
    import budget_runtime
    import budget_env
    import run_budget

    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only center authentication forbids physical calls and tensor loads")

    s.old.boot = budget_runtime.boot = run_budget.boot = forbidden
    s.torch.load = forbidden
    for name in ("__init__", "reset", "step", "restore"):
        setattr(budget_env.BudgetEnv, name, forbidden)
    s.torch.optim.Adam.__init__ = forbidden
    contract = s.read(s.GATE)["settings"]["contract"]
    if s.digest(contract) != s.CONTRACT_SHA:
        raise ValueError("Physical contract changed")
    previous = s.read(s.WAVE / s.SCENARIOS[0] / "settings.json")
    physical = {k: previous[k] for k in ("physical_config", "physical_options")}
    checks.helpers()  # Initialize inference-safe primitives before legacy td3 import.
    rows, files = [], {}
    for scenario in s.SCENARIOS:
        folder = s.CENTER / scenario
        before = {p.relative_to(s.REPO).as_posix(): s.file_hash(p) for p in folder.iterdir() if p.is_file()}
        values = load_completed_run(folder, "center", [None])
        rows.append(validate_center(*values, scenario, contract, physical))
        s.verify_files(before)
        files.update(before)
    print(json.dumps(dict(rows=rows, files=files, environment_calls=0, optimizer_updates=0,
                         physical_checkpoint_loads=0), allow_nan=False))


if __name__ == "__main__":
    main()

