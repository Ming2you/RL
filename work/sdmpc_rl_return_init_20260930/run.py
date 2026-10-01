"""One reviewed, single-thread offline job. No training until this CLI is dispatched."""
import argparse
import os
from pathlib import Path
import sys
import time
from runtime import (HERE, OUTPUT, SPEC, torch, np, sources, verify, read, save, save_once,
                     sha, digest, finite, locked, checkpoint, restore, process_identity,
                     check_stop, Stopped)
from data import authenticate, load_data
from learner import Learner


def review_receipt(path, source):
    receipt = read(path)
    if (receipt.get("status") != "approved" or not receipt.get("reviewer") or
            receipt.get("source_sha256") != source or receipt.get("spec_sha256") != digest(SPEC)):
        raise ValueError("Independent review must approve this exact source/specification")
    return dict(path=str(Path(path).resolve()), sha256=sha(path), review=receipt)


def execute(output, data, settings, resume=False):
    output = Path(output)
    check_stop(output)
    with locked(output):
        if (output / "completion.json").exists():
            raise FileExistsError("Completed run cannot be overwritten or resumed")
        existing = {p.name for p in output.iterdir()} - {"runner.lock"}
        if existing and not resume:
            raise FileExistsError("Partial run exists; resume in place")
        if resume and not (output / "settings.json").exists():
            raise ValueError("No frozen settings to resume")
        if resume and read(output / "settings.json") != settings:
            raise ValueError("Resume source/data/specification/review mismatch")
        verify(settings["sources"])
        verify(settings["data"]["files"])
        save_once(output / "settings.json", settings)
        identity = process_identity()
        session = output / "sessions" / (identity["session_id"] + ".json")
        started = time.perf_counter()
        save_once(session, dict(**identity, event="started"))
        learner = Learner(data)

        def persist():
            verify(settings["sources"])
            verify(settings["data"]["files"])
            return checkpoint(output, dict(settings=settings, learner=learner.state(),
                process=identity, critic_continuation="carry", candidate=learner.phase == "done"))

        if (output / "latest.json").exists():
            state = restore(output)
            if state["settings"] != settings:
                raise ValueError("Checkpoint settings mismatch")
            learner.load(state["learner"])
        else:
            # A crash after frozen settings but before the initial checkpoint is restartable.
            persist()
        outcome = "failed"
        try:
            while learner.phase not in ("done", "gate_failed"):
                check_stop(output)
                verify(settings["sources"])
                before = learner.phase
                learner.update()
                if (before != learner.phase or sum(learner.counts[k] for k in ("phi", "critic", "actor")) % 25 == 0):
                    persist()
                    save(output / "status.json", dict(status="running", phase=learner.phase, counters=learner.counts))
                    print(learner.phase, learner.counts, flush=True)
            check_stop(output)
            verify(settings["sources"])
            verify(settings["data"]["files"])
            learner.validate()
            save_once(output / "metrics.json", learner.metrics)
            latest = read(output / "latest.json")
            outputs = {"metrics.json": sha(output / "metrics.json"), "latest.json": sha(output / "latest.json"),
                       latest["path"]: latest["sha256"]}
            outcome = "completed" if learner.phase == "done" else "numerical_fit_failed"
            if learner.phase == "done":
                candidate = output / "model_final.pt"
                if candidate.exists():
                    if sha(candidate) != latest["sha256"]:
                        raise FileExistsError("Existing final candidate differs")
                else:
                    os.link(output / latest["path"], candidate)
                outputs["model_final.pt"] = sha(candidate)
            check_stop(output)
            save_once(output / "completion.json", dict(status=outcome, settings=settings,
                counters=learner.counts, outputs_sha256=outputs, critic_continuation="carry",
                candidate_admitted=learner.phase == "done", traffic_improvement_claim=False,
                canonical_evaluation=False, intermediate_selection_allowed=False))
            save(output / "status.json", dict(status=outcome, phase=learner.phase, counters=learner.counts))
            return outcome
        except Stopped:
            # The durable done checkpoint may already be the immutable candidate.
            if learner.phase != "done":
                persist()
            outcome = "stopped"
            save(output / "status.json", dict(status=outcome, phase=learner.phase, counters=learner.counts))
            return outcome
        except BaseException as exc:
            # Keep the last finite, authenticated checkpoint and every partial file.
            save(output / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}",
                phase=learner.phase, counters=learner.counts, resume_from="latest.json"))
            raise
        finally:
            save_once(session.with_name(session.stem + ".end.json"), dict(**identity, event=outcome,
                elapsed_wall_seconds=time.perf_counter()-started, counters=learner.counts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output.resolve() != OUTPUT.resolve():
        raise ValueError("Use only the declared return_init_v1 output root")
    try:
        check_stop(args.output)
        if (args.output / "completion.json").exists():
            raise FileExistsError("Completed run cannot be overwritten")
        source = sources()
        review = review_receipt(args.review, source)
        authenticated = authenticate()
        check_stop(args.output)
        data = load_data(authenticated["manifest"])
        settings = dict(spec=SPEC, sources=source, review=review, authentication=authenticated,
            data=authenticated["manifest"], runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__),
            scope="offline_training_only_no_environment", critic_continuation="carry")
        outcome = execute(args.output, data, settings, args.resume)
        print(outcome, flush=True)
        return 0 if outcome == "completed" else 3 if outcome == "stopped" else 2
    except Stopped:
        print("stopped_before_update", flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
