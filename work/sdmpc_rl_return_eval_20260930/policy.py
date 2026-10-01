"""Authenticate the MC archive; construct only the strict meta-loaded actor."""
import support as s

s.reuse("actor.py", ("FrozenActor",), dict_out := dict(w=s.old))
FrozenActor = dict_out["FrozenActor"]
COUNTS = dict(mc_critic=250)
ANCESTOR = dict(phi=1000, critic=250, actor=10, polyak=125)


def tensor_hash(state):
    h = s.hashlib.sha256()
    for name, item in sorted(state.items()):
        array = item.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str((array.dtype.str, array.shape)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def validate_model(payload, settings, done):
    s.finite_tree(payload)
    learner = payload["learner"]
    if (payload["format"] != "sdmpc-on-policy-return-mc-v1" or
            learner["format"] != payload["format"] or settings["format"] != payload["format"] or
            done["format"] != payload["format"] or payload["settings"] != settings or
            done["settings"] != settings or learner["spec"] != settings["spec"] or
            s.digest(settings["spec"]) != s.MODEL_SPEC_SHA or settings["spec_sha256"] != s.MODEL_SPEC_SHA or
            learner["phase"] != "done" or learner["counts"] != COUNTS or done["counters"] != COUNTS or
            learner["ancestor_counts"] != ANCESTOR or done["ancestor_counts"] != ANCESTOR or
            done["status"] != "completed" or len(learner["losses"]) != 250 or
            any(x["critic_continuation"] != s.CONTINUATION for x in (learner, settings, done)) or
            any(done[k] is not False for k in ("actor_changed", "phi_changed", "physical_policy_changed",
                                               "traffic_improvement_claim", "canonical_evaluation")) or
            done["training_fit_only"] is not True or
            learner["target_policy_actor_sha256"] != s.ACTOR_SHA or
            done["target_policy_actor_sha256"] != s.ACTOR_SHA or
            tensor_hash(learner["models"]["actor"]) != s.ACTOR_SHA):
        raise ValueError("MC format/specification/phase/continuation/actor identity differs")
    return FrozenActor(learner["models"]["actor"])


def authenticate():
    s.import_boundary()
    s.check_hash(s.MODEL / "completion.json", s.COMPLETION_SHA)
    done, settings = s.read(s.MODEL / "completion.json"), s.read(s.MODEL / "settings.json")
    s.verify_files(settings["sources"])
    s.verify_files(settings["data"]["files"])
    for name, expected in done["outputs_sha256"].items():
        path = (s.MODEL / name).resolve()
        if not path.is_relative_to(s.MODEL.resolve()):
            raise ValueError("Model output path escaped")
        s.check_hash(path, expected)
    s.check_hash(s.MODEL / "model_final.pt", s.MODEL_SHA)
    s.check_hash(s.GATE, s.GATE_SHA)
    gate = s.read(s.GATE)
    contract = gate["settings"]["contract"]
    if gate["status"] != "completed" or s.digest(contract) != s.CONTRACT_SHA:
        raise ValueError("Frozen physical contract differs")
    s.verify_files(gate["settings"]["source_sha256"])
    s.old.verify_pins(s.DEFAULT_SNAPSHOT, contract["source_pins"])
    if s.old.runtime_versions() != contract["runtime_versions"]:
        raise ValueError("Runtime contract differs")
    payload = s.torch.load(s.MODEL / "model_final.pt", map_location="cpu", weights_only=False)
    actor = validate_model(payload, settings, done)
    profiles = {r["scenario"]: r["profile_sha256"] for r in contract["environment_contract"]["scenarios"]}
    for scenario in s.SCENARIOS:
        path = s.DEFAULT_SNAPSHOT / "outputs/sdmpc_budget_exception_all_20260922/protocols_0" / scenario / "forecast.json"
        if s.digest(s.read(path)) != profiles[scenario]:
            raise ValueError("Canonical profile changed")
    snapshot = s.read(s.DEFAULT_SNAPSHOT.parent / "manifest.json")
    auth = dict(model_sha256=s.MODEL_SHA, actor_sha256=s.ACTOR_SHA, model_spec_sha256=s.MODEL_SPEC_SHA,
        model_completion_sha256=s.COMPLETION_SHA, model_settings_digest=s.digest(settings),
        contract=contract, profiles=profiles, critic_continuation=s.CONTINUATION,
        physical_snapshot=dict(manifest_sha256=contract["source_pins"]["snapshot_manifest_sha256"],
            file_count=snapshot["file_count"], total_bytes=snapshot["total_bytes"]))
    s.import_boundary()
    return actor, auth


def parity(actor):
    rows = []
    for scenario in s.SCENARIOS:
        path = s.WAVE / scenario / "experience.pt"
        expected = s.MODEL_DONE["settings"]["data"]["files"][path.relative_to(s.REPO).as_posix()]
        s.check_hash(path, expected)
        payload = s.torch.load(path, map_location="cpu", weights_only=False)
        if len(payload["transitions"]) != 75 or payload["settings"]["scenario"] != scenario:
            raise ValueError("Retained Task 2 trajectory differs")
        for obs, action, _, _, _ in payload["transitions"]:
            actual = actor.act(obs)
            if action.dtype != s.np.float32 or actual.tobytes() != action.tobytes():
                raise ValueError("Recorded action is not bitexact")
        rows.append(dict(scenario=scenario, observations=75, bitexact=True, experience_sha256=expected))
    s.import_boundary()
    return dict(observations=375, bitexact=True, scenarios=rows, actor_sha256=s.ACTOR_SHA)

