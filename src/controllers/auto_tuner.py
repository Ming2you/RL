from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.models.state import ExperimentConfig


@dataclass
class TuningAttempt:
    attempt_id: int
    config: ExperimentConfig
    changes: List[str] = field(default_factory=list)


class AutoTuner:
    """Rule-based deterministic tuner from the implementation spec."""

    def __init__(self, base_config: ExperimentConfig):
        self.base = base_config

    def attempt_config(self, attempt_id: int, dominant_failure_mode: str = "none") -> TuningAttempt:
        cfg = self.base
        changes: List[str] = []
        if attempt_id == 0:
            return TuningAttempt(attempt_id, cfg, ["default config"])
        updates: Dict[str, object] = {}
        if attempt_id == 1:
            updates = {
                "leader": {
                    "N_P_star_range": [
                        max(0.0, cfg.leader.N_P_star_range[0] * 0.8),
                        cfg.leader.N_P_star_range[1] * 1.25,
                    ],
                    "N_UF_star_range": [
                        max(0.0, cfg.leader.N_UF_star_range[0] * 0.8),
                        cfg.leader.N_UF_star_range[1] * 1.15,
                    ],
                    "w_L": cfg.leader.w_L * 0.75,
                }
            }
            changes.append("expanded leader grid and reduced leader smoothness")
        elif attempt_id == 2:
            updates = {
                "freeway_follower": {
                    "density_penalty": cfg.freeway_follower.density_penalty * 1.35,
                    "ramp_queue_penalty": cfg.freeway_follower.ramp_queue_penalty * 1.25,
                }
            }
            changes.append("increased density and ramp queue penalties")
        elif attempt_id == 3:
            updates = {
                "mpc": {
                    "max_nash_iter": cfg.mpc.max_nash_iter + 4,
                    "horizon_steps": cfg.mpc.horizon_steps + 1,
                    "nash_relaxation_alpha": max(0.5, cfg.mpc.nash_relaxation_alpha - 0.1),
                }
            }
            changes.append("increased horizon and Nash iterations with stronger relaxation")
        elif attempt_id == 4:
            updates = {
                "urban_follower": {
                    "boundary_balance_weight": cfg.urban_follower.boundary_balance_weight * 1.5,
                    "receiving_space_rule": "equal_split"
                    if cfg.urban_follower.receiving_space_rule == "proportional"
                    else "proportional",
                }
            }
            changes.append("strengthened boundary balancing and switched storage allocation rule")
        else:
            updates = {
                "freeway_follower": {"metering_smoothness_weight": cfg.freeway_follower.metering_smoothness_weight * 0.8},
                "urban_follower": {"green_smoothness_weight": cfg.urban_follower.green_smoothness_weight * 0.8},
            }
            changes.append("final best-known reduced smoothness pass")
        tuned = cfg.with_updates(updates)
        return TuningAttempt(attempt_id, tuned, changes)
