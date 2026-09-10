"""Immutable semantic contract shared by native PStack and RL execution arms."""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping

from src.controllers.pstack_factory import (
    CANONICAL_PSTACK_OPTIONS,
    PStackControllerOptions,
)
from src.models.demand import ScenarioConfig
from src.models.state import ExperimentConfig


EXPERIMENT_CONTRACT_VERSION = 1
EXPERIMENT_PROFILE_ID = "pstack_b13_wang_phase0_v1"


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"experiment contract contains unsupported value {type(value)!r}")


def canonical_contract_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _plain(payload),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def experiment_contract_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = canonical_contract_json(payload).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonicalize_experiment_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Remove behaviorally accidental mapping-order differences from a config."""
    return ExperimentConfig.from_dict(_plain(cfg))


@dataclass(frozen=True)
class ExperimentContract:
    """Canonical experiment semantics, independent of the policy execution arm."""

    canonical_json: str
    sha256: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExperimentContract":
        if int(payload.get("experiment_contract_version", -1)) != EXPERIMENT_CONTRACT_VERSION:
            raise ValueError("unsupported experiment contract version")
        if str(payload.get("profile_id", "")) != EXPERIMENT_PROFILE_ID:
            raise ValueError("unsupported experiment contract profile")
        canonical = canonical_contract_json(payload)
        return cls(
            canonical_json=canonical,
            sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_artifact(
        cls,
        payload: Mapping[str, Any],
        expected_sha256: str,
    ) -> "ExperimentContract":
        contract = cls.from_payload(payload)
        if not hmac.compare_digest(contract.sha256, str(expected_sha256)):
            raise ValueError(
                "experiment contract fingerprint mismatch: "
                f"expected {expected_sha256}, resolved {contract.sha256}"
            )
        return contract

    @classmethod
    def from_env(cls, env) -> "ExperimentContract":
        options = getattr(env, "pstack_options", CANONICAL_PSTACK_OPTIONS)
        scenario_name = str(getattr(env.scenario, "name", env.scenario_name))
        payload = {
            "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
            "profile_id": EXPERIMENT_PROFILE_ID,
            "scenario_name": scenario_name,
            "scenario": _plain(env.scenario),
            "resolved_config": _plain(env.cfg),
            "pstack_options": _plain(options),
            "warmup": {
                "control": "uncontrolled",
                "steps": int(env.warmup),
                "far_updates": False,
                "included_in_total_ttt": True,
            },
            "supervisor": {
                "mode": "link_pfo" if env.pfo_supervisor_enabled else "none",
            },
            "dynamic_far": {
                "mode": "incident_or_capacity_drop",
                "activation_ratio": 0.95,
            },
            "simulation": {
                "T_total_sec": float(env.T_total),
                "control_interval_sec": float(env.dt),
                "stochastic_seed": 0,
            },
        }
        return cls.from_payload(payload)

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)

    def artifact_fields(self) -> dict[str, Any]:
        return {
            "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
            "experiment_contract_sha256": self.sha256,
            "experiment_contract": self.payload,
        }

    def materialize(
        self,
    ) -> tuple[ExperimentConfig, ScenarioConfig, PStackControllerOptions]:
        payload = self.payload
        cfg = ExperimentConfig.from_dict(copy.deepcopy(payload["resolved_config"]))
        scenario_raw = copy.deepcopy(payload["scenario"])
        scenario_raw["name"] = str(payload["scenario_name"])
        scenario = ScenarioConfig(**scenario_raw)
        options = PStackControllerOptions.from_mapping(payload["pstack_options"])
        return cfg, scenario, options


def experiment_contract_payload(env) -> dict[str, Any]:
    return ExperimentContract.from_env(env).payload


def verify_contract_map(
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, ExperimentContract]:
    verified = {}
    for sha256, payload in contracts.items():
        verified[str(sha256)] = ExperimentContract.from_artifact(payload, str(sha256))
    return verified
