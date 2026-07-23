import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.models.demand import ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, segment_vsl
from src.experiments.rl_collect_replay_dataset import collect_replay_dataset
from src.experiments.rl_stackelberg_smoke import (
    run_smoke_experiment,
    run_smoke_rollout,
    write_smoke_outputs,
)
from src.rl import StackelbergRLEnvironment, export_rollout_records, random_safe_rollout
from src.rl.nash_probe import probe_follower_nash_residual
from src.rl.replay import ReplayDatasetWriter, RLTransition, transitions_from_env_step


def short_config():
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {"simulation": {"T_total": 360.0}},
    )


class StackelbergRLEnvironmentTest(unittest.TestCase):
    def test_no_e_control_actor(self):
        env = StackelbergRLEnvironment(short_config(), ScenarioConfig("test"), seed=7)

        self.assertNotIn("urban_E", env.agents)
        self.assertFalse(any(agent.signal == "E" for agent in env.agents.values()))
        self.assertFalse(env.centralized_action_space["emits_e_control"])
        self.assertFalse(any(agent_id.endswith("_E") for agent_id in env.follower_action_spaces))

    def test_follower_observations_are_local(self):
        cfg = short_config()
        env = StackelbergRLEnvironment(cfg, ScenarioConfig("test"), seed=7)
        observations = env.follower_observations_for_leader(
            env.leader_action_space.neutral_index()
        )

        self.assertTrue(observations)
        for obs in observations.values():
            names = " ".join(obs.feature_names)
            self.assertNotIn("global", names)
            self.assertNotIn("all_freeway", names)
            self.assertNotIn("all_urban", names)
            if obs.family == "freeway_segment":
                self.assertLessEqual(len(obs.owned_links), 1)
                self.assertLess(len(obs.features), len(cfg.network.freeway_links) * cfg.network.freeway_segments_per_link * 3)
            if obs.family == "urban_intersection":
                self.assertEqual(len(obs.owned_signals), 1)
                self.assertLess(len(obs.owned_movements), len(cfg.network.urban_movements))

    def test_action_mappings_respect_physical_bounds(self):
        cfg = short_config()
        env = StackelbergRLEnvironment(cfg, ScenarioConfig("test"), seed=7)
        n_p_low, n_p_high = cfg.leader.N_P_star_range
        n_uf_low, n_uf_high = cfg.leader.N_UF_star_range

        for action in env.leader_action_space.actions:
            self.assertGreaterEqual(action.N_P_star, n_p_low)
            self.assertLessEqual(action.N_P_star, n_p_high)
            self.assertGreaterEqual(action.N_UF_star, n_uf_low)
            self.assertLessEqual(action.N_UF_star, n_uf_high)

        for agent_id, action_space in env.follower_action_spaces.items():
            agent = env.agents[agent_id]
            for index in range(action_space.size):
                if agent.is_freeway:
                    action = action_space.map_index(index)
                    self.assertGreaterEqual(action.vsl_km_h, cfg.freeway_follower.vsl_min_km_h)
                    self.assertLessEqual(action.vsl_km_h, cfg.freeway_follower.vsl_max_km_h)
                    for ramp, value in action.ramp_metering.items():
                        cap = cfg.network.ramp_capacity_veh_h[ramp]
                        self.assertGreaterEqual(value, cfg.freeway_follower.ramp_metering_rate_min * cap)
                        self.assertLessEqual(value, cfg.freeway_follower.ramp_metering_rate_max * cap)
                else:
                    action = action_space.map_index(index)
                    self.assertGreaterEqual(action.green_p1_sec, cfg.network.green_min)
                    self.assertGreaterEqual(action.green_p2_sec, cfg.network.green_min)
                    self.assertLessEqual(action.green_p1_sec, cfg.network.green_max)
                    self.assertLessEqual(action.green_p2_sec, cfg.network.green_max)
                    self.assertAlmostEqual(
                        action.green_p1_sec + action.green_p2_sec,
                        cfg.network.effective_green_total,
                    )
                    self.assertTrue(all(0.0 <= value < cfg.network.cycle_length for value in action.offsets.values()))

    def test_leader_np_grid_uses_compact_ddqn_vocabulary(self):
        cfg = short_config()
        env = StackelbergRLEnvironment(cfg, ScenarioConfig("test"), seed=7)
        n_p_values = sorted({action.N_P_star for action in env.leader_action_space.actions})

        self.assertEqual(n_p_values, [-100.0, 175.0, 450.0, 725.0, 1000.0])
        self.assertGreaterEqual(min(n_p_values), cfg.leader.N_P_star_range[0])
        self.assertLessEqual(max(n_p_values), cfg.leader.N_P_star_range[1])
        self.assertNotIn(cfg.leader.N_P_star_range[0], n_p_values)
        self.assertNotIn(cfg.leader.N_P_star_range[1], n_p_values)

    def test_extreme_vsl_action_projects_to_dynamic_step_bound(self):
        cfg = short_config()
        env = StackelbergRLEnvironment(cfg, ScenarioConfig("test"), seed=7)
        freeway_agent_id = next(
            agent_id for agent_id, agent in env.agents.items() if agent.is_freeway
        )
        agent = env.agents[freeway_agent_id]
        action_space = env.follower_action_spaces[freeway_agent_id]
        extreme_index = min(
            range(action_space.size),
            key=lambda index: action_space.map_index(index).vsl_km_h,
        )
        leader_index, follower_indices = env.scripted_safe_action_indices()
        follower_indices[freeway_agent_id] = extreme_index
        previous = ControlAction.fixed(cfg)
        previous_vsl = segment_vsl(previous, agent.link, agent.segment_index, cfg)
        requested_vsl = action_space.map_index(extreme_index).vsl_km_h

        step = env.step(leader_index, follower_indices)
        applied_vsl = segment_vsl(
            step.record.control,
            agent.link,
            agent.segment_index,
            cfg,
        )
        action_details = step.record.physical_follower_actions[freeway_agent_id]
        diagnostics = step.record.control.diagnostics

        self.assertGreater(abs(requested_vsl - previous_vsl), cfg.freeway_follower.max_vsl_step)
        self.assertAlmostEqual(
            abs(applied_vsl - previous_vsl),
            cfg.freeway_follower.max_vsl_step,
        )
        self.assertGreaterEqual(applied_vsl, cfg.freeway_follower.vsl_min_km_h)
        self.assertLessEqual(applied_vsl, cfg.freeway_follower.vsl_max_km_h)
        self.assertEqual(diagnostics["rl_action_projection_applied"], 1.0)
        self.assertGreaterEqual(diagnostics["rl_projected_vsl_action_count"], 1.0)
        self.assertGreater(diagnostics["rl_max_requested_vsl_delta"], cfg.freeway_follower.max_vsl_step)
        self.assertLessEqual(diagnostics["rl_max_applied_vsl_delta"], cfg.freeway_follower.max_vsl_step)
        self.assertEqual(diagnostics["rl_action_fallback_used"], 0.0)
        self.assertEqual(action_details["requested_vsl_km_h"], requested_vsl)
        self.assertEqual(action_details["applied_vsl_km_h"], applied_vsl)
        self.assertEqual(action_details["vsl_projected"], 1.0)

    def test_short_safe_rollout_completes(self):
        records = random_safe_rollout(
            short_config(),
            ScenarioConfig("test"),
            max_steps=2,
            seed=7,
            policy="random",
        )

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.global_step_ttt >= 0.0 for record in records))
        self.assertTrue(all(record.control.diagnostics["rl_emits_e_control"] == 0.0 for record in records))
        self.assertTrue(all("global_step_ttt" in record.leader_reward_terms for record in records))
        for record in records:
            self.assertTrue(record.follower_rewards)
            self.assertTrue(all(value <= 0.0 for value in record.follower_rewards.values()))
            self.assertTrue(all(terms for terms in record.follower_reward_terms.values()))

    def test_rollout_export_preserves_reconstruction_keys(self):
        records = random_safe_rollout(
            short_config(),
            ScenarioConfig("test"),
            max_steps=1,
            seed=7,
            policy="scripted",
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = export_rollout_records(records, tmp)

            jsonl_rows = [
                json.loads(line)
                for line in paths["jsonl"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            with paths["csv"].open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(jsonl_rows), 1)
        self.assertEqual(len(csv_rows), 1)
        self.assertIn("leader_action", jsonl_rows[0])
        self.assertIn("physical_follower_actions", jsonl_rows[0])
        self.assertIn("follower_observations", jsonl_rows[0])
        self.assertIn("control", jsonl_rows[0])
        self.assertIn("diagnostics", jsonl_rows[0])
        self.assertEqual(csv_rows[0]["step_index"], "0")
        self.assertIn("leader_N_P_star", csv_rows[0])
        self.assertIn("control_rl_emits_e_control", csv_rows[0])

    def test_smoke_experiment_writes_required_output_files(self):
        records = run_smoke_rollout(
            short_config(),
            ScenarioConfig("test"),
            steps=1,
            seed=7,
            policy="scripted",
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_smoke_outputs(
                records,
                tmp,
                {"scenario": "test", "policy": "scripted"},
            )
            expected = {
                "rl_step_records.csv",
                "rl_rewards.csv",
                "rl_actions.csv",
                "rl_observations_summary.csv",
                "metadata.json",
            }
            written = {path.name for path in Path(tmp).iterdir()}
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            with paths["observations_summary"].open(newline="", encoding="utf-8") as handle:
                observation_rows = list(csv.DictReader(handle))

        self.assertTrue(expected.issubset(written))
        self.assertEqual(metadata["steps_recorded"], 1)
        self.assertEqual(metadata["scenario"], "test")
        self.assertTrue(all("features" in row for row in observation_rows))
        self.assertTrue(all(isinstance(json.loads(row["features"]), list) for row in observation_rows))

    def test_smoke_experiment_can_stream_replay_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "smoke"
            replay_output = Path(tmp) / "replay"
            paths = run_smoke_experiment(
                scenario_name="medium_demand",
                steps=1,
                policy="scripted",
                output_dir=output,
                replay_output_dir=replay_output,
                seed=7,
                t_total=360.0,
            )
            transitions = [
                json.loads(line)
                for line in paths["replay_transitions"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            metadata = json.loads(paths["replay_metadata"].read_text(encoding="utf-8"))

        expected_players = 1 + 5 + 2 * short_config().network.freeway_segments_per_link
        self.assertEqual(len(transitions), expected_players)
        self.assertEqual(metadata["transition_count"], expected_players)
        self.assertEqual(metadata["source"], "rl_stackelberg_smoke")

    def test_replay_transition_schema_accepts_leader_and_follower(self):
        leader_transition = RLTransition(
            episode=0,
            step=0,
            agent_id="leader",
            agent_family="leader",
            observation={"features": [0.0]},
            action_index=2,
            reward=-1.0,
            next_observation={"features": [0.1]},
            done=False,
            info={"target": "N_P_star"},
        )
        follower_transition = RLTransition(
            episode=0,
            step=0,
            agent_id="freeway_FW_E_seg0",
            agent_family="freeway_segment",
            observation={"features": [0.2]},
            action_index=1,
            reward=-0.5,
            next_observation={"features": [0.3]},
            done=True,
            info={"local": True},
        )

        self.assertEqual(leader_transition.as_dict()["agent_family"], "leader")
        self.assertEqual(follower_transition.as_dict()["agent_family"], "freeway_segment")
        self.assertIsInstance(leader_transition.action_index, int)
        self.assertIsInstance(follower_transition.action_index, int)

    def test_replay_writer_streams_step_transitions(self):
        env = StackelbergRLEnvironment(short_config(), ScenarioConfig("test"), seed=7)
        env.reset()
        leader_index, follower_indices = env.scripted_safe_action_indices()
        step_result = env.step(leader_index, follower_indices)
        transitions = transitions_from_env_step(episode=0, step_result=step_result)

        self.assertEqual(len(transitions), 1 + len(env.follower_action_spaces))
        self.assertEqual(transitions[0].agent_id, "leader")
        self.assertIn("features", transitions[0].observation)
        self.assertIn("features", transitions[0].next_observation)

        with tempfile.TemporaryDirectory() as tmp:
            with ReplayDatasetWriter(tmp, metadata={"scenario": "test"}) as writer:
                writer.append_many(transitions)
            transition_rows = [
                json.loads(line)
                for line in (Path(tmp) / "rl_transitions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            with (Path(tmp) / "rl_transition_index.csv").open(newline="", encoding="utf-8") as handle:
                index_rows = list(csv.DictReader(handle))
            metadata = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(len(transition_rows), len(transitions))
        self.assertEqual(len(index_rows), len(transitions))
        self.assertEqual(metadata["transition_count"], len(transitions))

    def test_collect_replay_dataset_writes_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = collect_replay_dataset(
                scenario_name="medium_demand",
                episodes=1,
                steps=1,
                policy="scripted",
                output_dir=tmp,
                seed=7,
                t_total=360.0,
            )
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            transitions = [
                json.loads(line)
                for line in paths["transitions"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(summary["episodes_recorded"], 1)
        self.assertEqual(summary["environment_steps_recorded"], 1)
        expected_players = 1 + 5 + 2 * short_config().network.freeway_segments_per_link
        self.assertEqual(summary["transitions_recorded"], expected_players)
        self.assertEqual(len(transitions), expected_players)

    def test_nash_probe_returns_nonnegative_epsilon(self):
        env = StackelbergRLEnvironment(short_config(), ScenarioConfig("test"), seed=7)
        env.reset()
        leader_index, follower_indices = env.scripted_safe_action_indices()

        result = probe_follower_nash_residual(env, leader_index, follower_indices)

        self.assertGreaterEqual(result.epsilon_Nash_hat, 0.0)
        self.assertIsNotNone(result.best_deviating_agent_id)
        self.assertIsNotNone(result.best_unilateral_action_index)
        self.assertEqual(env.step_index, 0)
        self.assertEqual(len(env.records), 0)

    def test_nash_probe_preserves_simulator_accumulators_after_prior_step(self):
        env = StackelbergRLEnvironment(short_config(), ScenarioConfig("test"), seed=7)
        env.reset()
        leader_index, follower_indices = env.scripted_safe_action_indices()
        env.step(leader_index, follower_indices)
        total_ttt_before = env.sim.total_ttt
        logs_before = len(env.sim.logs)
        step_index_before = env.step_index
        records_before = len(env.records)

        probe_follower_nash_residual(env, leader_index, follower_indices)

        self.assertEqual(env.step_index, step_index_before)
        self.assertEqual(len(env.records), records_before)
        self.assertEqual(len(env.sim.logs), logs_before)
        self.assertAlmostEqual(env.sim.total_ttt, total_ttt_before)


if __name__ == "__main__":
    unittest.main()
