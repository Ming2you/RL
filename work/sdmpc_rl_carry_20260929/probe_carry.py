"""Matched saved states: one actual interval, never a Q target or full-run claim."""
import argparse
import copy
import importlib.util
from pathlib import Path
import time
from budget_runtime import boot, DEFAULT_SNAPSHOT, REPO, read, save, plain, protocol
from run_budget import exclusive_run, pins, verify_pins, file_hash


def fixture(rt, step):
    import numpy as np
    folder = rt["root"] / "outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/sweet_170_incident_w/upper"
    row = read(folder / f"plant_{step-1:03d}.json")
    return (rt["restore_state"](rt["rc"], row["state"]), rt["rc"].ControlAction(**row["control"]),
            np.array(row["persistent_dual"]), None if row["last_budget"] is None else np.array(row["last_budget"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=int, choices=(5, 30, 50), required=True)
    p.add_argument("--cpu-mask", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    rt = boot(DEFAULT_SNAPSHOT, args.cpu_mask)
    import numpy as np
    from budget_controller import BudgetController, InvalidReference
    old_path = REPO / "work/sdmpc_rl_budget_20260929/budget_controller.py"
    spec = importlib.util.spec_from_file_location("v1_probe_controller", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    source_pins = pins(DEFAULT_SNAPSHOT)
    actions = {"zero": [0., 0.], "np_plus": [.25, 0.], "np_minus": [-.25, 0.],
               "nuf_plus": [0., .25], "nuf_minus": [0., -.25]}
    settings = dict(step=args.step, actions=actions, source_pins=source_pins,
                    v1_controller_sha256=file_hash(old_path), cpu_mask=args.cpu_mask,
                    modes=["v1_pfo_h3", "carry_physical"], scope="matched_one_interval_diagnostic")
    with exclusive_run(args.output):
        if (args.output / "completion.json").exists():
            raise RuntimeError("Probe already complete")
        if args.resume and read(args.output / "settings.json") != settings:
            raise RuntimeError("Probe resume contract differs")
        save(args.output / "settings.json", settings)
        state, previous, dual, prior_budget = fixture(rt, args.step)
        _, _, profile = protocol(rt)
        forecast = profile.horizon(state.time_sec, 3)
        records = []
        for mode in settings["modes"]:
            for label, action in actions.items():
                if (args.output / "STOP").exists() or (args.output.parent / "STOP").exists():
                    save(args.output / "status.json", dict(status="paused", completed=len(records)))
                    return
                verify_pins(DEFAULT_SNAPSHOT, source_pins)
                path = args.output / f"{mode}_{label}.json"
                if path.exists():
                    saved = read(path)
                    if saved["settings"] != settings or saved["mode"] != mode or saved["action"] != action:
                        raise RuntimeError("Probe record provenance differs")
                    records.append(saved)
                    continue
                controller = old.BudgetController(rt) if mode == "v1_pfo_h3" else BudgetController(rt)
                controller.lower.dual = dual.copy()
                controller.lower.last_budget = copy.deepcopy(prior_budget)
                tick = time.perf_counter()
                warm, pfo_seconds, pfo_calls, source = None, 0., 0, "previous"
                try:
                    if mode == "carry_physical" and args.step > 5:
                        try:
                            controller.prepare_reference(state, forecast, previous, previous, reference_source="previous")
                        except InvalidReference:
                            source = "pfo_recovery"
                    else:
                        source = "pfo_current" if mode == "v1_pfo_h3" else "pfo_initial"
                    if source != "previous":
                        start = time.perf_counter()
                        warm = rt["rc"].make_controller("WU-FAITHFUL-FOLLOWER", rt["cfg"])
                        control = warm.solve(state.copy(), None, forecast, previous).control
                        pfo_seconds, pfo_calls = time.perf_counter() - start, 1
                        if mode == "v1_pfo_h3":
                            controller.prepare_reference(state, forecast, previous, control)
                        else:
                            controller.prepare_reference(state, forecast, previous, control, reference_source=source)
                    prepare_seconds = time.perf_counter() - tick
                    selected, audit = controller.evaluate_and_commit(action)
                    sim = rt["rc"].MixedTrafficSimulator(rt["cfg"])
                    sim.state = state.copy()
                    before = sim.total_ttt
                    log = sim.step(selected["control"], forecast[0], args.step)
                    interval_ttt = sim.total_ttt - before
                    if not np.isfinite(interval_ttt) or interval_ttt < 0 or abs(interval_ttt-log.freeway_ttt-log.urban_ttt) > 1e-8:
                        raise RuntimeError("Probe actual interval accounting failed")
                    if not audit["execution_check"]["physical_control_valid"] or not audit["execution_check"]["budget_feasible"]:
                        raise RuntimeError("Probe executed invalid control")
                    record = dict(settings=settings, mode=mode, candidate=label, action=action,
                        source=source, pfo_calls=pfo_calls, pfo_seconds=pfo_seconds,
                        prepare_seconds=prepare_seconds,
                        decision_seconds=prepare_seconds+audit["lower_wall_seconds"]+audit["guard_wall_seconds"],
                        interval_ttt=interval_ttt, full_run=False, original_state_seconds=state.time_sec,
                        reached_state=plain(sim.state), physical_point=controller.lower.coords.encode(selected["control"]),
                        audit=audit)
                    save(path, record)
                    records.append(plain(record))
                    save(args.output / "status.json", dict(status="running", completed=len(records)))
                    print("PROBE", args.step, mode, label, interval_ttt, audit["selection_source"], flush=True)
                finally:
                    if warm is not None and hasattr(warm, "close"):
                        warm.close()
        summary_rows = []
        for row in records:
            zero = next(r for r in records if r["mode"] == row["mode"] and r["candidate"] == "zero")
            audit = row["audit"]
            summary_rows.append(dict(mode=row["mode"], candidate=row["candidate"], action=row["action"],
                changed_physical_control=bool(np.max(abs(np.array(row["physical_point"])-zero["physical_point"])) > 1e-9),
                interval_ttt=row["interval_ttt"], delta_vs_zero=row["interval_ttt"]-zero["interval_ttt"],
                requested=audit["B_requested"][0], executed=audit["B_executed"], source=row["source"],
                selection_source=audit["selection_source"], fallback_reasons=audit["fallback_reasons"],
                h3_would_reject=audit.get("h3_guard_would_reject"), pfo_calls=row["pfo_calls"],
                decision_seconds=row["decision_seconds"], counts=audit["counts"]))
        verify_pins(DEFAULT_SNAPSHOT, source_pins)
        result = dict(status="completed", settings=settings, step=args.step, records=summary_rows,
            all_executions_valid=True, carry_influence=any(r["changed_physical_control"] for r in summary_rows if r["mode"] == "carry_physical"),
            no_ordinary_pfo=all(r["pfo_calls"] == 0 for r in summary_rows if r["mode"] == "carry_physical") if args.step > 5 else None,
            scope="one_interval_matched_state_not_Q_labels_or_full_run")
        save(args.output / "completion.json", result)
        save(args.output / "status.json", dict(status="completed"))


if __name__ == "__main__":
    main()
