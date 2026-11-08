from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from development_worlds import catalog
from training import DiscoveryEnv, MAX_INTERACTION_ROUNDS
from training.generate import generate


ROOT = Path(__file__).resolve().parents[1]
CLAIMS = {"dynamics_order": 2, "hidden_matter": False, "time_dependence": False}


def request(method, **params):
    return {"method": method, "params": params}


class TrainingEnvironmentTests(unittest.TestCase):
    def make_env(self, world="2"):
        env = DiscoveryEnv([world], clock_mode="frozen")
        env.reset(seed=13)
        return env

    def launch_request(self, n=1):
        launch = copy.deepcopy(catalog.first_light("2"))
        launch["probes"] = [copy.deepcopy(launch["probes"][0]) for _ in range(n)]
        launch["readout"] = [0.0, .005, .01]
        return request("launch", **launch)

    def submit_request(self):
        # Interface demo law; scientific quality is not assumed by these tests.
        return request("submit", law_path=str(ROOT / "examples" / "law.py"), claims=CLAIMS)

    @staticmethod
    def fake_score(world, law_path, claims, energy, calibration=None):
        # Tests score/energy plumbing, not a fake performance claim.
        return {"prediction": 80., "law": 15., "saving": 5. * (1 - energy / 10),
                "total": 100. - .5 * energy, "energy_used": energy,
                "claims_truth": {"secret": "never expose"}, "truths": [[123.]]}

    def test_seed_repeats_observations_and_transitions(self):
        env = self.make_env()
        first, info = env.reset(seed=789)
        after = env.step([request("observe", times=[0., .01]), self.launch_request()])
        again, info_again = env.reset(seed=789)
        repeated = env.step([request("observe", times=[0., .01]), self.launch_request()])
        self.assertEqual(first, again)
        self.assertEqual(info, info_again)
        self.assertEqual(after, repeated)
        different, _ = env.reset(seed=790)
        self.assertNotEqual(first, different)

    def test_frozen_budget_and_free_first_light(self):
        env = self.make_env()
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs["turns_remaining"], 3)
        self.assertEqual(obs["interaction_round"], 0)
        self.assertEqual(obs["interaction_rounds_remaining"], MAX_INTERACTION_ROUNDS)
        self.assertEqual(obs["budget"]["energy_used"], 0)
        self.assertEqual(obs["budget"]["wall_clock_remaining_seconds"], 300.)
        self.assertEqual(len(obs["results"][0]["result"]["first_light"]["positions"]), 8)

    def test_energy_budget_counts_probes_and_rejects_overdraft(self):
        env = self.make_env()
        obs, reward, term, trunc, _ = env.step([self.launch_request(8), self.launch_request(2),
                                               self.launch_request(1), request("archive")])
        self.assertEqual([row["ok"] for row in obs["results"]], [True, True, False, True])
        self.assertEqual(obs["budget"]["energy_used"], 10)
        self.assertEqual(obs["budget"]["energy_remaining"], 0)
        self.assertEqual(reward, 0.)
        self.assertFalse(term or trunc)

    def test_observe_quota_and_cache_do_not_resample(self):
        env = self.make_env()
        obs, *_ = env.step([request("observe", times=[0.]) for _ in range(21)])
        self.assertEqual(sum(row["ok"] for row in obs["results"]), 20)
        self.assertEqual(obs["budget"]["observe_used"], 20)
        archived = env.step([request("archive"), request("archive")])[0]
        self.assertEqual(archived["results"][0]["result"], archived["results"][1]["result"])
        self.assertEqual(archived["budget"]["energy_used"], 0)

    def test_no_submission_terminates_on_turn_three_without_bootstrap(self):
        env = self.make_env()
        for i in range(3):
            _, reward, term, trunc, info = env.step(request("archive"))
            self.assertEqual(reward, 0.)
            self.assertEqual(term, i == 2)
            self.assertFalse(trunc)
            self.assertEqual(info["bootstrap_allowed"], i != 2)
        self.assertEqual(info["reason"], "turn_limit")
        with self.assertRaises(RuntimeError):
            env.step(request("archive"))

    def test_errors_consume_turns_and_do_not_spend_energy(self):
        env = self.make_env()
        for action in ([], {"method": "launch", "params": {}}, request("unknown")):
            observation, reward, terminated, truncated, _ = env.step(action)
            self.assertFalse(observation["results"][0]["ok"])
            self.assertEqual(observation["budget"]["energy_used"], 0)
            self.assertEqual(reward, 0.)
        self.assertFalse(truncated)
        self.assertTrue(terminated)

    def test_submit_on_last_turn_uses_actual_energy_and_only_scalar_score(self):
        env = self.make_env()
        env.step(self.launch_request(2))
        env.step(request("archive"))
        with mock.patch("training.env.evaluation.calibrate_world", return_value={"reference": "test"}), \
             mock.patch("training.env.evaluation.evaluate_law", side_effect=self.fake_score) as evaluate:
            obs, reward, terminated, truncated, info = env.step(
                [request("archive"), self.submit_request(), self.launch_request(8)])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, .99)
        self.assertFalse(info["bootstrap_allowed"])
        self.assertEqual(evaluate.call_args.args[3], 2)
        self.assertEqual(len(obs["results"]), 2)  # Ignore operations after submission.
        self.assertEqual(obs["budget"]["energy_used"], 2)
        self.assertEqual(set(info["score"]), {"prediction", "law", "saving", "total", "energy_used"})
        self.assertNotIn("never expose", json.dumps((obs, info)))
        with self.assertRaises(RuntimeError):
            env.step(self.submit_request())

    def test_bad_submission_can_be_repaired_within_same_batch(self):
        env = self.make_env()
        invalid = self.submit_request()
        invalid["params"]["claims"] = {"dynamics_order": 99}
        with mock.patch.object(env, "_score_submission", return_value=self.fake_score("2", None, CLAIMS, 0)):
            obs, reward, term, trunc, _ = env.step([invalid, self.submit_request()])
        self.assertEqual([row["ok"] for row in obs["results"]], [False, True])
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(reward, 1.)

    def test_reference_cache_invalidates_when_sources_change(self):
        env = self.make_env()
        with mock.patch("training.env.evaluation._source_hashes", side_effect=[{"x": "a"}, {"x": "a"}, {"x": "b"}]), \
             mock.patch("training.env.evaluation.calibrate_world", side_effect=[{"version": "a"}, {"version": "b"}]) as calibrate:
            self.assertEqual(env._reference(), {"version": "a"})
            self.assertEqual(env._reference(), {"version": "a"})
            self.assertEqual(env._reference(), {"version": "b"})
        self.assertEqual(calibrate.call_count, 2)

    def test_scoring_failure_propagates_instead_of_zero_reward(self):
        env = self.make_env()
        with mock.patch.object(env, "_score_submission", side_effect=RuntimeError("calibration failed")):
            with self.assertRaisesRegex(RuntimeError, "calibration failed"):
                env.step(self.submit_request())
        with self.assertRaises(RuntimeError):
            env.step(request("archive"))

    def test_truth_labels_and_world_ids_not_in_policy_observations(self):
        env = self.make_env()
        obs, info = env.reset(seed=3)
        self.assertEqual(obs["results"][0]["result"]["world_id"], "training")
        archived = env.step(request("archive"))
        payload = json.dumps((obs, info, archived))
        for name in ("claims_truth", "dynamics_order", "time_dependence", "hidden_matter",
                     "PARAMETERS", "reference_sha256", "global_V", "ballistic_baseline"):
            self.assertNotIn(name, payload)
        self.assertEqual(env.trainer_metadata["world_id"], "2")
        self.assertEqual(archived[0]["results"][0]["result"]["world_id"], "training")

    def test_real_time_deadline_gates_submission(self):
        now = [100.]
        with mock.patch("training.env.time.monotonic", side_effect=lambda: now[0]):
            env = DiscoveryEnv(["2"])
            env.reset(seed=1)
            now[0] += 301.
            _, reward, term, trunc, info = env.step(self.submit_request())
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(reward, 0.)
        self.assertEqual(info["reason"], "deadline")
        self.assertFalse(info["bootstrap_allowed"])

    def test_seeded_task_selection_repeats(self):
        def draw():
            env = DiscoveryEnv(["1", "2", "3"], clock_mode="frozen")
            env.reset(seed=11)
            ids = [env.trainer_metadata["world_id"]]
            for _ in range(3):
                env.reset()
                ids.append(env.trainer_metadata["world_id"])
            return ids
        self.assertEqual(draw(), draw())

    def test_generator_streams_noisy_data_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "one.jsonl"
            second = Path(directory) / "two.jsonl"
            for path in (first, second):
                generate(path, episodes=1, seed=4, worlds=["2"])
            self.assertEqual(first.read_bytes(), second.read_bytes())
            record = json.loads(first.read_text())
            self.assertEqual(len(record["transitions"]), 3)
            self.assertEqual([row["observation"]["interaction_round"] for row in record["transitions"]], [1, 2, 3])
            self.assertEqual([action["method"] for action in record["transitions"][0]["action"]],
                             ["observe", "insert_probe", "archive"])
            self.assertTrue(record["transitions"][-1]["terminated"])
            self.assertFalse(record["transitions"][-1]["truncated"])
            self.assertFalse(record["transitions"][-1]["info"]["bootstrap_allowed"])
            self.assertNotIn("claims_truth", first.read_text())
            manifest = json.loads(Path(str(first) + ".manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["episodes"][0]["world_id"], "2")
            with self.assertRaises(FileExistsError):
                generate(first, episodes=1, seed=4, worlds=["2"])

    def test_insert_probe_is_exact_launch_alias_in_schema_noise_and_cost(self):
        launch_env = self.make_env()
        insert_env = self.make_env()
        launch_action = self.launch_request(2)
        insert_action = copy.deepcopy(launch_action)
        insert_action["method"] = "insert_probe"
        launch_output = launch_env.step(launch_action)
        insert_output = insert_env.step(insert_action)
        self.assertEqual(launch_output[0]["results"][0]["result"],
                         insert_output[0]["results"][0]["result"])
        self.assertEqual(launch_output[0]["budget"], insert_output[0]["budget"])
        self.assertEqual(insert_output[0]["budget"]["energy_used"], 2)
        self.assertEqual(launch_output[1:], insert_output[1:])
        for env, method in ((launch_env, "launch"), (insert_env, "insert_probe")):
            obs, *_ = env.step(request(method, probes=[]))
            self.assertFalse(obs["results"][0]["ok"])
            self.assertEqual(obs["budget"]["energy_used"], 2)

    def test_three_interaction_batches_allow_multiple_tools_and_final_submit(self):
        env = self.make_env()
        insert = self.launch_request(2)
        insert["method"] = "insert_probe"
        obs, reward, term, trunc, _ = env.step([
            request("observe", times=[0., .01]), insert, request("archive")])
        self.assertEqual(len(obs["results"]), 3)
        self.assertTrue(all(row["ok"] for row in obs["results"]))
        self.assertEqual((obs["interaction_round"], obs["interaction_rounds_remaining"]), (1, 2))
        self.assertFalse(term or trunc)
        # Any number of local reasoning/fitting calls does not call env.step.
        for _ in range(10):
            self.assertEqual(env._turn, 1)
        obs, *_ = env.step(request("batch", actions=[
            request("observe", times=[0.]), request("archive")]))
        self.assertEqual((obs["interaction_round"], obs["turn"]), (2, 2))
        self.assertEqual((obs["interaction_rounds_remaining"], obs["turns_remaining"]), (1, 1))
        last_insert = self.launch_request(1)
        last_insert["method"] = "insert_probe"
        with mock.patch.object(env, "_score_submission", side_effect=lambda path, claims, energy:
                               self.fake_score("2", path, claims, energy)) as scorer:
            obs, reward, term, trunc, info = env.step(request("batch", actions=[
                request("observe", times=[0.]), last_insert, self.submit_request(), request("archive")]))
        self.assertEqual(len(obs["results"]), 3)
        self.assertEqual(obs["interaction_round"], 3)
        self.assertEqual(obs["interaction_rounds_remaining"], 0)
        self.assertEqual(obs["budget"]["energy_used"], 3)
        self.assertEqual(scorer.call_args.args[2], 3)
        self.assertEqual(reward, .985)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertFalse(info["bootstrap_allowed"])
        self.assertEqual(info["max_interaction_rounds"], MAX_INTERACTION_ROUNDS)

    def test_batch_wrapper_rejects_malformed_and_nested_batches(self):
        env = self.make_env()
        obs, *_ = env.step(request("batch", actions=request("archive")))
        self.assertFalse(obs["results"][0]["ok"])
        self.assertEqual(obs["interaction_round"], 1)
        obs, *_ = env.step(request("batch", actions=[request("batch", actions=[request("archive")])]))
        self.assertFalse(obs["results"][0]["ok"])
        self.assertEqual(obs["interaction_round"], 2)
        self.assertEqual(obs["budget"]["energy_used"], 0)

    def test_generator_shuffles_each_complete_coverage_block(self):
        worlds = ["1", "2", "3"]
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"episodes-{i}.jsonl" for i in range(2)]
            for path in paths:
                generate(path, episodes=6, seed=42, worlds=worlds)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            manifest = json.loads(Path(str(paths[0]) + ".manifest.json").read_text())
            sequence = [episode["world_id"] for episode in manifest["episodes"]]
            self.assertEqual(set(sequence[:3]), set(worlds))
            self.assertEqual(set(sequence[3:]), set(worlds))
            self.assertNotEqual(sequence, worlds * 2)
            self.assertEqual(manifest["world_sampling"], "seeded permutation per coverage block")
            for row in paths[0].read_text().splitlines():
                episode = json.loads(row)
                self.assertEqual(episode["initial_observation"]["results"][0]["result"]["world_id"], "training")


if __name__ == "__main__":
    unittest.main()
