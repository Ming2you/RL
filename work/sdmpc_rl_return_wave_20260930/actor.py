"""Strict admitted actor authentication and inference; never import a learner."""
import wave_support as w

COUNTS = dict(phi=1000, critic=250, actor=10, polyak=125)


class FrozenActor(w.torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        shapes = {"net.0.weight": (64, 2367), "net.0.bias": (64,),
                  "net.2.weight": (64, 64), "net.2.bias": (64,),
                  "net.4.weight": (2, 64), "net.4.bias": (2,), "bounds": (2,)}
        if set(state) != set(shapes):
            raise ValueError("Incomplete/extra actor weights")
        for key, shape in shapes.items():
            t = state[key]
            if not isinstance(t, w.torch.Tensor) or tuple(t.shape) != shape or t.dtype != w.torch.float32:
                raise ValueError("Actor shape/dtype differs: " + key)
        w.finite_tree(state)
        if not w.torch.equal(state["bounds"], w.torch.tensor([.2, .1], dtype=w.torch.float32)):
            raise ValueError("Actor bounds differ")
        # Meta construction allocates no random replacement weights or RNG draws.
        with w.torch.device("meta"):
            self.net = w.torch.nn.Sequential(w.torch.nn.Linear(2367, 64), w.torch.nn.ReLU(),
                w.torch.nn.Linear(64, 64), w.torch.nn.ReLU(), w.torch.nn.Linear(64, 2))
            self.register_buffer("bounds", w.torch.empty(2, dtype=w.torch.float32))
        self.load_state_dict(state, strict=True, assign=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, obs):
        return self.bounds * w.torch.tanh(self.net(obs))

    @w.torch.no_grad()
    def act(self, obs):
        if not isinstance(obs, w.np.ndarray) or obs.dtype != w.np.float32 or obs.shape != (2367,):
            raise ValueError("Native float32 observation required")
        w.finite_tree(obs)
        action = self(w.torch.from_numpy(obs)).numpy().copy()
        w.finite_tree(action)
        if action.dtype != w.np.float32 or action.shape != (2,):
            raise ValueError("Native float32 action required")
        return action


def validate_candidate(payload, settings, completion):
    w.finite_tree(payload)
    learner = payload["learner"]
    if (payload["settings"] != settings or w.digest(settings["spec"]) != w.SPEC_SHA or
            learner["spec"] != settings["spec"] or learner["phase"] != "done" or
            learner["counts"] != COUNTS or completion["counters"] != COUNTS or
            payload["candidate"] is not True or payload["critic_continuation"] != "carry" or
            learner["critic_continuation"] != "carry" or settings["critic_continuation"] != "carry" or
            completion["status"] != "completed" or completion["candidate_admitted"] is not True or
            completion["critic_continuation"] != "carry" or completion["canonical_evaluation"] is not False or
            completion["traffic_improvement_claim"] is not False):
        raise ValueError("Admitted model specification/phase/counters differ")
    gate = learner["metrics"]["gate"]
    initial, final = gate["initial"], gate["final"]
    checks = dict(pooled_halved=final["pooled"] <= .5 * initial["pooled"],
        each_scenario_decreased=all(final["per_scenario"][s] < initial["per_scenario"][s] for s in w.SCENARIOS))
    if gate["passed"] is not True or gate["checks"] != checks or not all(checks.values()):
        raise ValueError("Admitted numerical fit gate differs")
    for key in ("phi", "critic", "actor"):
        if len(learner["losses"][key]) != COUNTS[key]:
            raise ValueError("Admitted loss/counter mismatch")
    return FrozenActor(learner["models"]["actor"])


def authenticate():
    w.import_boundary()
    w.check_hash(w.PREDECESSOR / "completion.json", w.COMPLETION_SHA)
    done = w.read(w.PREDECESSOR / "completion.json")
    settings = w.read(w.PREDECESSOR / "settings.json")
    if settings != done["settings"] or w.digest(settings["spec"]) != w.SPEC_SHA:
        raise ValueError("Predecessor settings/specification differ")
    w.verify_files(settings["sources"])
    w.verify_files(settings["data"]["files"])
    for name, expected in done["outputs_sha256"].items():
        path = (w.PREDECESSOR / name).resolve()
        if not path.is_relative_to(w.PREDECESSOR.resolve()):
            raise ValueError("Predecessor output path escaped")
        w.check_hash(path, expected)
    w.check_hash(w.PREDECESSOR / "model_final.pt", w.MODEL_SHA)
    w.check_hash(w.GATE, w.GATE_SHA)
    gate = w.read(w.GATE)
    contract = gate["settings"]["contract"]
    identity = settings["data"]["frozen_identity"]
    if (gate["status"] != "completed" or w.digest(contract) != identity["contract_sha256"] or
            contract["source_pins"] != identity["physical"] or
            contract["runtime_versions"] != identity["runtime"] or identity["gate_sha256"] != w.GATE_SHA):
        raise ValueError("Physical predecessor contract differs")
    w.verify_files(gate["settings"]["source_sha256"])
    w.verify_pins(w.DEFAULT_SNAPSHOT, contract["source_pins"])
    if w.runtime_versions() != contract["runtime_versions"]:
        raise ValueError("Frozen runtime versions differ")
    payload = w.torch.load(w.PREDECESSOR / "model_final.pt", map_location="cpu", weights_only=False)
    actor = validate_candidate(payload, settings, done)
    if payload["learner"]["metrics"] != w.read(w.PREDECESSOR / "metrics.json"):
        raise ValueError("Admitted model metrics differ")
    w.import_boundary()
    snapshot = w.read(w.DEFAULT_SNAPSHOT.parent / "manifest.json")
    return actor, dict(model_sha256=w.MODEL_SHA, predecessor_completion_sha256=w.COMPLETION_SHA,
        predecessor_settings_digest=w.digest(settings), spec_sha256=w.SPEC_SHA, contract=contract,
        physical_snapshot=dict(manifest_sha256=contract["source_pins"]["snapshot_manifest_sha256"],
                               file_count=snapshot["file_count"], total_bytes=snapshot["total_bytes"]),
        helper_sources=identity["local_sources"], profiles={s: w.expected_profile(s) for s in w.SCENARIOS},
        critic_continuation="carry")


def verify_live(authentication, sources):
    w.import_boundary()
    if w.sources() != sources:
        raise ValueError("Worker source changed")
    w.check_hash(w.PREDECESSOR / "completion.json", w.COMPLETION_SHA)
    w.check_hash(w.PREDECESSOR / "model_final.pt", authentication["model_sha256"])
    w.check_hash(w.GATE, w.GATE_SHA)
    settings = w.read(w.PREDECESSOR / "settings.json")
    if w.digest(settings) != authentication["predecessor_settings_digest"]:
        raise ValueError("Predecessor settings changed")
    w.verify_files(settings["sources"])
    w.verify_files(authentication["helper_sources"])
    w.verify_pins(w.DEFAULT_SNAPSHOT, authentication["contract"]["source_pins"])
    if w.runtime_versions() != authentication["contract"]["runtime_versions"]:
        raise ValueError("Runtime changed")
