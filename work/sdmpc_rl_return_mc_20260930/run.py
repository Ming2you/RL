"""One single-thread offline MC job; parent owns review and global worker admission."""
import argparse
import os
from pathlib import Path
import sys
import time
from mc_common import (OUTPUT, SPEC, MODEL_SHA, torch, np, read, save, save_once, sha, digest,
    sources, verify, check_stop, Stopped, locked, process_identity, checkpoint, restore)
from mc_data import authenticate, load_data, load_parent
from mc_learner import Learner, final_metrics


def review_receipt(path, source):
    receipt = read(path)
    if (receipt.get("status") != "approved" or not receipt.get("reviewer") or
            receipt.get("source_sha256") != source or receipt.get("spec_sha256") != digest(SPEC)):
        raise ValueError("Independent review must approve this exact source/specification")
    return dict(path=str(Path(path).resolve()), sha256=sha(path), review=receipt)


def execute(output, data, parent, settings, resume=False):
    output = Path(output)
    check_stop(output)
    with locked(output):
        check_stop(output)
        if (output / "completion.json").exists():
            raise FileExistsError("Completed MC run cannot be repeated or resumed")
        existing = {p.name for p in output.iterdir()} - {"runner.lock", "stdout.log", "stderr.log", "launcher.json"}
        if existing and not resume:
            raise FileExistsError("Partial/orphan run exists; preserve and resume")
        if resume and (not (output / "settings.json").exists() or read(output / "settings.json") != settings):
            raise ValueError("Resume settings/source/data/spec/review mismatch")
        verify(settings["sources"])
        verify(settings["data"]["files"])
        save_once(output / "settings.json", settings)
        identity = process_identity()
        identity["command"] = list(sys.orig_argv)
        save(output / "process.json", dict(**identity, lock_held=True, source_sha256=settings["sources"]))
        session = output / "sessions" / (identity["session_id"] + ".start.json")
        save_once(session, dict(process=identity, lock_held=True, settings_sha256=digest(settings)))
        started, outcome, learner = time.perf_counter(), "failed", None
        try:
            learner = Learner(data, parent)
            if (output / "latest.json").exists():
                state = restore(output)
                if (state["settings"] != settings or state["format"] != SPEC["format"] or
                        state["target_policy_parent_model_sha256"] != MODEL_SHA):
                    raise ValueError("Checkpoint identity mismatch")
                learner.load(state["learner"])
            else:
                checkpoint(output, dict(format=SPEC["format"], settings=settings, learner=learner.state(),
                    process=identity, target_policy_parent_model_sha256=MODEL_SHA))
            while learner.phase != "done":
                check_stop(output)
                verify(settings["sources"])
                learner.update()
                if learner.counts["mc_critic"] % 25 == 0:
                    verify(settings["data"]["files"])
                checkpoint(output, dict(format=SPEC["format"], settings=settings, learner=learner.state(),
                    process=identity, target_policy_parent_model_sha256=MODEL_SHA))
                save(output / "status.json", dict(status="running", counters=learner.counts))
                if learner.counts["mc_critic"] % 25 == 0:
                    print("MC", learner.counts["mc_critic"], flush=True)
            # Reuse the durable done payload through publication, including STOP/resume.
            # Never serialize it again with a new process identity after exporting it.
            check_stop(output)
            verify(settings["sources"])
            verify(settings["data"]["files"])
            save_once(output / "metrics.json", final_metrics(learner))
            latest = read(output / "latest.json")
            if latest["phase"] != "done" or latest["counters"] != dict(mc_critic=250):
                raise ValueError("Final checkpoint is not the durable 250-update boundary")
            candidate = output / "model_final.pt"
            if candidate.exists():
                if sha(candidate) != latest["sha256"]:
                    raise FileExistsError("Existing final model differs; preserve it")
            else:
                os.link(output / latest["path"], candidate)
            outputs = {name: sha(output / name) for name in
                ("settings.json", "metrics.json", "latest.json", "model_final.pt", latest["path"])}
            check_stop(output)
            verify(settings["sources"])
            verify(settings["data"]["files"])
            save_once(output / "completion.json", dict(format=SPEC["format"], status="completed",
                settings=settings, outputs_sha256=outputs, counters=learner.counts,
                ancestor_counts=parent["counts"], critic_continuation=SPEC["critic_continuation"],
                target_policy_actor_sha256=learner.initial_hashes["actor"],
                target_policy_parent_model_sha256=MODEL_SHA, continued_critic_optimizer=True,
                training_fit_only=True, heldout=False, actor_changed=False, phi_changed=False,
                physical_policy_changed=False, traffic_improvement_claim=False, canonical_evaluation=False,
                next_action="Parent reviews actual fit; no automatic policy proposal or dispatch"))
            outcome = "completed"
        except Stopped:
            outcome = "stopped"
        except BaseException as exc:
            save(output / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}",
                resume_from="latest.json", preserve_orphans=True))
            raise
        finally:
            save_once(session.with_name(identity["session_id"] + ".end.json"), dict(process=identity,
                start_sha256=sha(session), event=outcome, elapsed_wall_seconds=time.perf_counter()-started))
        save(output / "status.json", dict(status=outcome, counters=learner.counts if learner else None))
        return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output.resolve() != OUTPUT.resolve():
        raise ValueError("Only the fixed return_mc_v1 output is allowed")
    try:
        check_stop(args.output)
        if (args.output / "completion.json").exists():
            raise FileExistsError("Completed MC run cannot be repeated")
        source = sources()
        review = review_receipt(args.review, source)
        authentication = authenticate()
        check_stop(args.output)
        data, parent = load_data(authentication["manifest"]), load_parent(authentication["manifest"])
        settings = dict(format=SPEC["format"], spec=SPEC, spec_sha256=digest(SPEC), sources=source,
            review=review, authentication=authentication, data=authentication["manifest"],
            parent_model_sha256=MODEL_SHA, runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__),
            scope="offline_on_policy_MC_critic_fit_only", critic_continuation=SPEC["critic_continuation"])
        outcome = execute(args.output, data, parent, settings, args.resume)
        print(outcome, flush=True)
        return 0 if outcome == "completed" else 3
    except Stopped:
        print("stopped_before_update", flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
