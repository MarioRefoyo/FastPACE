from types import SimpleNamespace
import unittest

import numpy as np

from methods.FastPACE.algorithms import CEMSequencePlanningNN, HierarchicalCEMNN
from methods.FastPACE.env import JointTimeChannelDiscreteCFEnv


class FixedActionRng:
    def __init__(self, action):
        self.action = action

    def choice(self, _n_actions, size, p):
        return np.full(size, self.action, dtype=np.int32)


class PlannerEnv:
    def __init__(self, action_value, planning_credit=False):
        self.action_space = SimpleNamespace(n=3)
        self.end_action = 2
        self.action_value = action_value
        self.evaluation_batches = []
        self.planning_credit = planning_credit

    def renew_mask_vec(self, masks, actions):
        updated = masks.copy()
        updated[actions == 0] = self.action_value
        return updated

    def evaluate_counterfactual_batch(self, masks):
        self.evaluation_batches.append(masks.copy())
        return masks.reshape(masks.shape[0], -1).mean(axis=1)


def make_planner(action_value, planning_credit=False):
    env = PlannerEnv(action_value, planning_credit)
    planner = CEMSequencePlanningNN(
        elite_frac=0.5,
        alpha=0.1,
        tabu_mode=False,
        use_conditional_reuse=False,
        planning_credit=planning_credit,
        env=env,
    )
    planner.rng = FixedActionRng(action=0)
    return planner, env


class CEMValidityTests(unittest.TestCase):
    def test_cem_keeps_stop_fallback_for_worse_or_tied_samples(self):
        for action_value in (0.0, 1.0):
            with self.subTest(action_value=action_value):
                planner, _ = make_planner(action_value)

                _, best_seq, best_fit, *_ = planner.run_cem(
                    mask=np.ones((1, 2), dtype=np.float32),
                    logits=np.zeros(3),
                    iters=1,
                    num_sims=4,
                    planning_steps=2,
                    temp=1.0,
                    tabu_mode=False,
                    target="basic",
                )

                np.testing.assert_array_equal(best_seq, [planner.stop_action, planner.stop_action])
                self.assertAlmostEqual(best_fit, 1.0)

    def test_cem_strictly_better_sample_replaces_stop_fallback(self):
        planner, _ = make_planner(action_value=1.0)

        _, best_seq, best_fit, *_ = planner.run_cem(
            mask=np.zeros((1, 2), dtype=np.float32),
            logits=np.zeros(3),
            iters=1,
            num_sims=4,
            planning_steps=2,
            temp=1.0,
            tabu_mode=False,
            target="basic",
        )

        np.testing.assert_array_equal(best_seq, [0, 0])
        self.assertAlmostEqual(best_fit, 1.0)

    def test_planning_credit_reuses_initial_mask_evaluation(self):
        planner, env = make_planner(action_value=0.0, planning_credit=True)

        planner.run_cem(
            mask=np.ones((1, 2), dtype=np.float32),
            logits=np.zeros(3),
            iters=1,
            num_sims=4,
            planning_steps=2,
            temp=1.0,
            tabu_mode=False,
            target="basic",
        )

        single_mask_evaluations = [batch for batch in env.evaluation_batches if batch.shape[0] == 1]
        self.assertEqual(len(single_mask_evaluations), 1)


class ThresholdModel:
    def to(self, _device):
        return self

    def predict(self, samples):
        samples = np.asarray(samples)
        if samples.ndim == 2:
            samples = samples[np.newaxis, ...]
        desired_probability = (samples.mean(axis=(1, 2)) >= 0.5).astype(float)
        return np.column_stack((1.0 - desired_probability, desired_probability))

    def predict_class(self, samples):
        return np.argmax(self.predict(samples), axis=1)


class ZeroOutlierCalculator:
    def get_outlier_scores(self, samples):
        samples = np.asarray(samples)
        batch_size = 1 if samples.ndim == 2 else samples.shape[0]
        return np.zeros(batch_size)


def make_env(mask_init="zeros"):
    return JointTimeChannelDiscreteCFEnv(
        X_train=np.zeros((1, 1, 2), dtype=np.float32),
        nuns=np.ones((1, 1, 1, 2), dtype=np.float32),
        model_wrapper=ThresholdModel(),
        latent_ts_block_len=1,
        reward_type="final",
        non_valid_penalization=-0.1,
        channel_clusters=[[0]],
        outlier_calculator=ZeroOutlierCalculator(),
        weights_losses=[0, 1, 0, 0],
        mask_init=mask_init,
        max_steps=3,
        include_repetition_end=False,
        include_end_action=True,
        device="cpu",
    )


class EpisodeValidityTests(unittest.TestCase):
    def test_zero_search_start_retains_valid_nun_fallback(self):
        env = make_env(mask_init="zeros")
        x_orig = np.zeros((1, 2), dtype=np.float32)
        nun = np.ones((1, 1, 2), dtype=np.float32)

        state, _ = env.reset(x_orig=x_orig, nun=nun)
        result = env.get_cfe()

        np.testing.assert_array_equal(state["mask"], np.zeros((1, 2)))
        np.testing.assert_array_equal(result["best_mask"], np.ones((1, 2)))
        np.testing.assert_array_equal(result["cfs"], nun[0])
        self.assertEqual(result["valid"], 1)
        self.assertAlmostEqual(result["fitness"], env.nun_fitness)
        self.assertEqual(result["step"], 0)

    def test_episode_retains_earlier_valid_mask_across_level_and_budget_end(self):
        base_env = make_env(mask_init="zeros")
        level_envs = [make_env(mask_init="zeros"), make_env(mask_init="zeros")]
        level_cemnns = {
            index: SimpleNamespace(
                env=level_env,
                stop_action=level_env.end_action,
                nA=level_env.action_space.n,
                level_idx=index,
            )
            for index, level_env in enumerate(level_envs)
        }

        algorithm = HierarchicalCEMNN.__new__(HierarchicalCEMNN)
        algorithm.env = base_env
        algorithm.level_CEMNNs = level_cemnns
        algorithm.block_non_adj_mask = False
        algorithm.inf_plan_every = 1
        algorithm.inf_num_simulations_ratio = 1
        algorithm.inf_planning_steps = 1
        algorithm.inf_cem_iters = 1
        algorithm.temp = 1.0
        algorithm.pruning_ch_block_pct = None
        algorithm.pruning_cemnn = None
        algorithm._warm_start_logits = {
            algorithm._make_level_key(level_cemnns[0]): np.zeros(level_cemnns[0].nA)
        }
        algorithm._inherit_warm_start_logits = lambda *_args, **_kwargs: None

        planned_actions = iter([0, level_cemnns[0].stop_action, 0])

        def plan(*_args, **_kwargs):
            action = next(planned_actions)
            return np.zeros(3), np.zeros(3), 0.0, np.array([action], dtype=np.int32)

        algorithm.plan = plan
        x_orig = np.zeros((1, 2), dtype=np.float32)
        nun = np.ones((1, 1, 2), dtype=np.float32)

        algorithm.collect_episode(
            levels=[0, 1],
            max_plan_calls=3,
            x_orig=x_orig,
            nun=nun,
            record_teacher_targets=False,
        )
        result = base_env.get_cfe()

        np.testing.assert_array_equal(base_env.mask, np.zeros((1, 2)))
        self.assertEqual(base_env.steps, 3)
        np.testing.assert_array_equal(result["best_mask"], np.array([[1.0, 0.0]]))
        np.testing.assert_array_equal(result["cfs"], np.array([[1.0, 0.0]]))
        self.assertEqual(result["valid"], 1)
        self.assertAlmostEqual(result["fitness"], -0.5)
        self.assertEqual(result["step"], 1)
        self.assertAlmostEqual(result["improvement_over_nun"], 0.5)


if __name__ == "__main__":
    unittest.main()
