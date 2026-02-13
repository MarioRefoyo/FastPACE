import itertools
import os
import time

import numpy as np
from copy import copy, deepcopy
import tempfile

from .counterfactual_common import CounterfactualMethod
from methods.RL.env import SingleTSDiscreteMaskCFEnv

from methods.RL.algorithms import HierarchicalCEMNN
from methods.RL.hierachical_clustering import ChannelHierarchy, NaiveOrder, ChannelGreedyGroups, ChannelHierarchyFixedK

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


class FastPACECF(CounterfactualMethod):

    def __init__(
        self,
        model_wrapper, outlier_calculator, X_train, nuns,
        latent_ts_block_pcts, latent_ts_block_train_pct,
        ch_block_pcts, ch_block_train_pct,
        pruning_ch_block_pct,
        channel_groups, standardize, ch_similarity,
        reward_type, non_valid_penalization, weight_losses, mask_init,
        max_steps,
        algorithm_params,
        device, tensorboard_path=None,
        cf_max_plan_calls: int = 50,
    ):
        super().__init__(model_wrapper)
        self.cf_max_plan_calls = cf_max_plan_calls
        self.n_channels = X_train.shape[1]
        self.ts_length = X_train.shape[2]

        if len(latent_ts_block_pcts) != len(ch_block_pcts):
            raise ValueError("latent_ts_block_pcts and ch_block_pcts must have the same length and hierarchy.")
        self.latent_ts_block_pcts = sorted(latent_ts_block_pcts, reverse=True)
        latent_ts_block_lens = [int(max(1, np.floor(pct * self.ts_length))) for pct in self.latent_ts_block_pcts]
        self.latent_ts_block_lens = latent_ts_block_lens
        self.ch_block_pcts = sorted(ch_block_pcts, reverse=True)

        if latent_ts_block_train_pct is None or ch_block_train_pct is None:
            self.train_index = None
        else:
            try:
                ts_matches = np.where(np.isclose(latent_ts_block_pcts, latent_ts_block_train_pct))[0]
                ch_matches = np.where(np.isclose(ch_block_pcts, ch_block_train_pct))[0]
                common = np.intersect1d(ts_matches, ch_matches)
                if common.size == 0:
                    raise IndexError("The ts and ch block pcts do not align to the same hierarchy index.")
                self.train_index = int(common[-1])

                if int(max(1, np.floor(latent_ts_block_train_pct * self.ts_length))) != self.latent_ts_block_lens[self.train_index]:
                    raise ValueError
                print(f"Using ts block pct {latent_ts_block_train_pct} for training: {self.latent_ts_block_lens[self.train_index]}-"
                      f"floor({latent_ts_block_train_pct}*{self.ts_length}={latent_ts_block_train_pct*self.ts_length})")
            except ValueError:
                raise f"{latent_ts_block_train_pct} not in latent_ts_block_pcts"
            except IndexError:
                raise "The ts or ch blocks pct do not match"

        self.pruning_ch_block_pct = pruning_ch_block_pct
        self.base_latent_ts_block_len = self.latent_ts_block_lens[-1]
        self.base_ch_block_pct = self.ch_block_pcts[-1]

        # --- Choose between channel grouper ---
        if self.n_channels == 1:
            channel_groups = "NaiveOrder"
        self.ch_similarity = ch_similarity
        if channel_groups == "NaiveOrder":
            ch_cluster = NaiveOrder(self.n_channels)
        elif channel_groups == "HierarchicalClustering":
            ch_cluster = ChannelHierarchy(
                self.n_channels, similarity=self.ch_similarity, corr_power=0.75, linkage_method="ward",
                standardize=standardize, pct_margin=1.0)
            ch_cluster.fit(X_train)
        elif channel_groups == "HierarchicalClusteringFixedK":
            ch_cluster = ChannelHierarchyFixedK(
                self.n_channels, similarity=self.ch_similarity, corr_power=0.75, standardize=standardize)
            ch_cluster.fit(X_train)
        elif channel_groups == "FixedClosestClustering":
            ch_cluster = ChannelGreedyGroups(
                self.n_channels, similarity=self.ch_similarity, corr_power=0.75, standardize=standardize)
            ch_cluster.fit(X_train)
        else:
            raise ValueError("Not valid channel grouping")

        # --- Create base environment ---
        self.env_params = dict(
            X_train=X_train,
            nuns=nuns,
            model_wrapper=model_wrapper,
            outlier_calculator=outlier_calculator,
            reward_type=reward_type,
            non_valid_penalization=non_valid_penalization,
            weights_losses=weight_losses,
            mask_init=mask_init,
            max_steps=max_steps,
            include_repetition_end=False,
            include_end_action=True,
            device=device
        )
        verbose = 0
        
        # Create base env
        self.env_base = SingleTSDiscreteMaskCFEnv
        self.env_params["latent_ts_block_len"] = 1
        self.env = self.get_env(self.env_base, self.env_params, 1, "dummy")

        # --- Define agent ---
        cemnn_class = HierarchicalCEMNN
        self.cemnn_params = dict(
            elite_frac=algorithm_params["elite_frac"],
            alpha=algorithm_params["alpha"],
            tabu_mode=algorithm_params["tabu_mode"],
            use_conditional_reuse=algorithm_params["use_conditional_reuse"],
            planning_credit=algorithm_params["planning_credit"]
        )
        self.agent = cemnn_class(
            latent_ts_block_lens=self.latent_ts_block_lens,
            ch_block_pcts=self.ch_block_pcts,
            train_index=self.train_index,
            channel_hierarchy=ch_cluster,
            pruning_ch_block_pct=self.pruning_ch_block_pct,
            env_params=self.env_params,
            cemnn_params=self.cemnn_params,
            block_non_adj_mask=algorithm_params["block_non_adj_mask"],
            warm_alpha=algorithm_params["warm_alpha"],
            hierarchy_warm_alpha=algorithm_params["hierarchy_warm_alpha"],
            train_num_simulations_ratio=algorithm_params["train_num_simulations_ratio"],
            inf_num_simulations_ratio=algorithm_params["inf_num_simulations_ratio"],
            train_planning_steps=algorithm_params["train_planning_steps"],
            inf_planning_steps=algorithm_params["inf_planning_steps"],
            train_cem_iters=algorithm_params["train_cem_iters"],
            inf_cem_iters=algorithm_params["inf_cem_iters"],
            train_plan_every=algorithm_params["train_plan_every"],
            inf_plan_every=algorithm_params["inf_plan_every"],
            search_target=algorithm_params["search_target"],
            train_target=algorithm_params["train_target"],
            train_target_type=algorithm_params["train_target_type"],
            train_pisoft_func=algorithm_params["train_pisoft_func"],
            temp=algorithm_params["temp"],
            train_add_episodes_each_steps=None,
            training_episodes=None,
            val_pct=None,
            early_patience=None,
            algorithm_policy_loss=None,
            env=self.env,
            device=device,
            tensorboard_log=tensorboard_path,
            verbose=verbose,
        )

    def fit(self, total_steps):
        raise NotImplementedError()

    def generate_counterfactual_specific(
        self,
        x_orig,
        desired_target=None,
        nun_example=None,
    ):
        if nun_example is None:
            raise ValueError("nun_example must be provided to generate a counterfactual.")
        if not hasattr(self.agent, "collect_episode"):
            raise AttributeError("HCEMNNCF requires an agent with a collect_episode method.")

        levels = getattr(self.agent, "train_levels", None)
        if not levels:
            levels = list(self.agent.level_CEMNNs.keys())
        max_plan_calls = self.cf_max_plan_calls

        _, _, _, _, _, _, ebuf_e_len = self.agent.collect_episode(
            levels,
            max_plan_calls,
            x_orig=x_orig,
            nun=nun_example,
            use_inf_params=True,
            record_teacher_targets=True,
        )
        
        result = self.env.get_cfe()
        result["steps"] = ebuf_e_len[0]

        return result

    @staticmethod
    def get_env(env_base, env_params, parallel_envs, parallel_env_type):
        if parallel_envs == 1:
            env = env_base(**env_params)
        elif parallel_envs > 1:
            if parallel_env_type == "subproc":
                vec_env_cls = SubprocVecEnv
            elif parallel_env_type == "dummy":
                vec_env_cls = DummyVecEnv
            else:
                raise ValueError("Not valid parallel_env_type")
            env = make_vec_env(env_base, n_envs=parallel_envs, env_kwargs=env_params, vec_env_cls=vec_env_cls)
        else:
            raise ValueError("Not valid number of parallel_envs")

        return env
    
