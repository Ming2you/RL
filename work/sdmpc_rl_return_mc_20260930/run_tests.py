"""Scoped synthetic tests and optional read-only actual authentication; never actual fit."""
import os
import sys
import uuid
import xml.etree.ElementTree as ET
from mc_common import (HERE, OUTPUT, SCENARIOS, SPEC, ANCESTOR_COUNTS, np, torch, sources, read, save,
    sha, digest, tensor_hash, verify)
from mc_data import manifest, authenticate, load_data, load_parent
from mc_learner import state_hash


def readonly_actual():
    receipt = authenticate()
    data = load_data(receipt["manifest"])
    parent = load_parent(receipt["manifest"])
    optimizer = parent["optimizers"]["critics"]
    assert parent["counts"] == ANCESTOR_COUNTS and parent["phase"] == "done"
    assert parent["critic_continuation"] == "carry"
    assert len(optimizer["param_groups"]) == 1 and optimizer["param_groups"][0]["lr"] == 3e-4
    assert {float(state["step"]) for state in optimizer["state"].values()} == {250.}
    rows = []
    for i, scenario in enumerate(SCENARIOS):
        mask = data["scenario"] == i
        actual = receipt["readout"]["scenarios"][i]
        intervals = np.asarray(actual["interval_ttt"], dtype=np.float64)
        expected = -np.cumsum(intervals[::-1]/100.)[::-1].copy()
        np.testing.assert_allclose(data["returns"][mask].numpy(), expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(data["returns"][mask][-1].numpy(), data["reward"][mask][-1].numpy())
        np.testing.assert_allclose(float(data["returns"][mask][0]), -(actual["ttt"]-actual["warmup_ttt"])/100., rtol=0, atol=1e-10)
        assert int(mask.sum()) == 75 and int(data["terminal"][mask].sum()) == 1
        rows.append(dict(scenario=scenario, rows=75, G0=float(data["returns"][mask][0]),
            terminal_reward=float(data["reward"][mask][-1]), warmup_excluded=actual["warmup_ttt"],
            nuf_cap_requests=int((data["request"][mask, 1] == 6000.).sum())))
    assert data["anchor"].dtype == data["request"].dtype == torch.float64
    assert data["obs"].shape == (375, 2367)
    return dict(authentication=receipt, data_arrays_sha256=tensor_hash(data), rows=rows,
        parent_model_state_sha256={k: tensor_hash(v) for k, v in parent["models"].items()},
        parent_critic_optimizer_sha256=state_hash(parent["optimizers"]["critics"]),
        parent_critic_optimizer_steps=250, parent_critic_optimizer_lr=3e-4,
        ancestor_counts=parent["counts"], actual_optimizer_updates=0, physical_calls=0,
        note="Actual arrays never passed to a Learner; optimizer tests use generated synthetic arrays only.")


def main():
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    import pytest
    args = list(sys.argv[1:])
    synthetic_only = "--synthetic-only" in args
    if synthetic_only:
        args.remove("--synthetic-only")
    evidence = HERE / "test-evidence" / uuid.uuid4().hex[:8]
    evidence.mkdir(parents=True)
    before, source = manifest(), sources()
    output_existed = OUTPUT.exists()
    if not synthetic_only:
        save(evidence / "readonly-actual.json", readonly_actual())
    code = pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", "--confcutdir="+str(HERE),
        "--junitxml="+str(evidence / "tests.xml"), str(HERE / "test_mc.py"), *args])
    verify(before["files"])
    if before != manifest() or source != sources() or OUTPUT.exists() != output_existed:
        raise ValueError("Frozen files/source or actual output existence changed")
    suite = ET.parse(evidence / "tests.xml").getroot().find("testsuite")
    save(evidence / "evidence.json", dict(exit_code=code, command=[sys.executable, "-B", str(HERE / "run_tests.py"), *sys.argv[1:]],
        source_sha256=source, spec=SPEC, spec_sha256=digest(SPEC), test_summary=suite.attrib,
        tests_xml_sha256=sha(evidence / "tests.xml"),
        readonly_actual_sha256=None if synthetic_only else sha(evidence / "readonly-actual.json"),
        preserved_before_after_sha256=before["files"], preserved_file_count=len(before["files"]),
        actual_rows_validated=0 if synthetic_only else 375, actual_optimizer_updates=0, physical_calls=0,
        actual_output_existed_before_and_after=output_existed,
        runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__)))
    print("EVIDENCE", evidence, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
