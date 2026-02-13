import hashlib
from abc import ABC, abstractmethod
import time
from copy import copy, deepcopy

import numpy as np
from stable_baselines3.common.logger import configure, Logger
import torch as th

from methods.FastPACE.env import JointTimeChannelDiscreteCFEnv, SubsequenceChannelDiscreteCFEnv


def resolve_inference_params(
    train_num_simulations,
    train_planning_steps,
    train_cem_iters,
    inf_num_simulations_ratio,
    inf_planning_steps_ratio,
    inf_cem_iters_ratio,
    inf_num_simulations=None,
    inf_planning_steps=None,
    inf_cem_iters=None,

):
    """Resolve explicit inference hyperparameters, falling back to ratios."""
    resolved_inf_num_sim = (
        inf_num_simulations
        if inf_num_simulations is not None
        else int(round(train_num_simulations * inf_num_simulations_ratio))
    )
    resolved_inf_steps = (
        inf_planning_steps
        if inf_planning_steps is not None
        else int(round(train_planning_steps * inf_planning_steps_ratio))
    )
    resolved_inf_cem_iters = (
        inf_cem_iters
        if inf_cem_iters is not None
        else int(round(train_cem_iters * inf_cem_iters_ratio))
    )
    return {
        "inf_num_simulations": int(resolved_inf_num_sim),
        "inf_planning_steps": int(resolved_inf_steps),
        "inf_cem_iters": int(resolved_inf_cem_iters),
    }


def _mix_logits_probabilities(logits_a, logits_b, weight, temp, eps=1e-12):
    """
    Blend two logit vectors by averaging their induced probability distributions.
    The returned logits are scaled so that softmax(., temp) recovers the mixed probs.
    """

    def _softmax_no_floor(x):
        z = (x - np.max(x)) / temp
        e = np.exp(z, dtype=np.float64)
        return e / np.sum(e)

    probs_a = _softmax_no_floor(logits_a)
    probs_b = _softmax_no_floor(logits_b)
    probs_mix = (1.0 - weight) * probs_a + weight * probs_b
    probs_mix = np.maximum(probs_mix, eps)
    probs_mix /= np.sum(probs_mix)

    return temp * np.log(probs_mix)


class RLAlgorithm(ABC):
    def __init__(self, env, gamma, device, tensorboard_log, verbose):
        self.env = env
        self.gamma = gamma
        self.device = device
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose

        if tensorboard_log is not None:
            self.logger: Logger = configure(folder=tensorboard_log, format_strings=["tensorboard"])

    @abstractmethod
    def reset_agent(self):
        pass

    def change_tensorboard_log(self, tensorboard_log):
        self.logger: Logger = configure(folder=tensorboard_log, format_strings=["tensorboard"])

    def change_env(self, env):
        self.env = env


class HierarchicalCEMNN(RLAlgorithm):
    def __init__(
            self,
            latent_ts_block_lens,
            ch_block_pcts,
            train_index,
            channel_hierarchy,
            pruning_ch_block_pct,
            env_params,
            cemnn_params,
            block_non_adj_mask,
            warm_alpha, hierarchy_warm_alpha,
            train_num_simulations_ratio, inf_num_simulations_ratio,
            train_planning_steps, inf_planning_steps,
            train_cem_iters, inf_cem_iters,
            train_plan_every, inf_plan_every,
            search_target,
            train_target,
            train_target_type,
            train_pisoft_func,
            temp,
            train_add_episodes_each_steps,
            training_episodes,
            val_pct,
            early_patience,
            algorithm_policy_loss,
            env, device=None, tensorboard_log=None, verbose=0,
    ):
        super().__init__(env, 1, device, tensorboard_log, verbose)
        self.latent_ts_block_lens = latent_ts_block_lens
        self.ch_block_pcts = ch_block_pcts
        self.train_index = train_index

        self.env_params = env_params
        self.cemnn_params = cemnn_params
        self.block_non_adj_mask = block_non_adj_mask
        self.algorithm_policy_loss = algorithm_policy_loss

        self.warm_alpha = warm_alpha
        self.hierarchy_warm_alpha = hierarchy_warm_alpha

        # Set train - inference parameters
        self.train_num_simulations_ratio = train_num_simulations_ratio
        self.train_planning_steps = train_planning_steps
        self.train_cem_iters = train_cem_iters
        self.inf_num_simulations_ratio = inf_num_simulations_ratio
        self.inf_planning_steps = inf_planning_steps
        self.inf_cem_iters = inf_cem_iters
        self.train_plan_every = self._set_plan_every(train_plan_every, self.train_planning_steps)
        self.inf_plan_every = self._set_plan_every(inf_plan_every, self.inf_planning_steps)
        self.search_target = search_target
        self.train_target = train_target
        self.train_target_type = train_target_type
        self.train_pisoft_func = train_pisoft_func
        self.temp = temp
        self.train_add_episodes_each_steps = train_add_episodes_each_steps
        self.training_episodes = training_episodes
        self.val_pct = val_pct
        self.early_patience = early_patience

        # Create low level CEMNNs (do not include pruning)
        level_CEMNNs = {}
        for idx in range(len(latent_ts_block_lens)):
            # Create env for this CEMNN
            latent_ts_block_len = latent_ts_block_lens[idx]
            ch_block_pct = ch_block_pcts[idx]
            channel_groups = channel_hierarchy.cut(channel_pct=ch_block_pct)
            env_params["latent_ts_block_len"] = latent_ts_block_len
            env_params["channel_clusters"] = channel_groups
            env = JointTimeChannelDiscreteCFEnv(**env_params)
            current_cemnn_params = {"env": env, **cemnn_params}
            # Define CEMNN.
            cemnn = CEMSequencePlanningNN(**current_cemnn_params)
            cemnn.level_idx = idx
            level_CEMNNs[idx] = cemnn
        self.level_CEMNNs = level_CEMNNs
        # Set pruning_cemnn
        self.channel_hierarchy = channel_hierarchy
        self.pruning_ch_block_pct = pruning_ch_block_pct
        self.pruning_cemnn = None

        self.trained_decision_model = False
        self.train_metrics = {}
        # Init episodes buffer
        if self.train_index is not None:
            self.train_levels = [self.train_index]
        else:
            self.train_levels = list(self.level_CEMNNs.keys())

        self.current_steps = 0
        self.best_action_seq = None
        # Cache per-level logits to warm start CEM runs within a level
        self._warm_start_logits = {}

    @staticmethod
    def _set_plan_every(plan_every, planning_steps):
        if isinstance(plan_every, int):
            return min(plan_every, planning_steps)
        elif isinstance(plan_every, str):
            if plan_every == "horizon":
                return planning_steps
            elif plan_every == "half_horizon":
                return max(round(planning_steps / 2), 1)
            else:
                raise ValueError("Not valid plan_every string")
        else:
            raise ValueError("Not valid plan_every type")


    def reset_agent(self):
        # Set current cemnn
        self.current_cemnn_idx = 0
        self.current_cemnn = self.level_CEMNNs[self.current_cemnn_idx]
        # Reset logits of each cemn
        for cemnn in self.level_CEMNNs.values():
            # Reset agent
            cemnn.reset_agent()
            # Reset internal (imaginary) environment
            cemnn.env.reset(self.env.x_orig, np.expand_dims(self.env.nun, axis=0))

        # Rest status for planning
        self.current_steps = 0
        self.best_action_seq = None

        # Reset pruning_cemnn
        self.pruning_cemnn = None
        # Reset warm-start cache
        self._warm_start_logits = {}
        self._last_level_logits = {}

    @staticmethod
    def _ensure_batch3(x):
        """Ensure (N, C, T) for x_orig/x_nun/x_cf; allow (C, T) input."""
        if x.ndim == 2:
            return x[np.newaxis, ...]
        return x

    @staticmethod
    def _make_level_key(cemnn):
        """Unique-ish key for warm start that is safe to extend across hierarchies later."""
        return (getattr(cemnn, "level_idx", None), cemnn.nA, cemnn.stop_action)

    def _set_cemnn_action_mask(self, cemnn, mask_curr):
        if not isinstance(cemnn.env, JointTimeChannelDiscreteCFEnv):
            cemnn.allowed_action_mask = None
            return

        if mask_curr.ndim == 3:
            mask_curr = mask_curr[0]
        active_time = np.any(mask_curr > 0, axis=0)

        n_time_blocks = cemnn.env.ts_action_dim
        block_len = cemnn.env.latent_ts_block_len
        ts_len = cemnn.env.ts_length

        active_blocks = []
        for block_idx in range(n_time_blocks):
            start = block_idx * block_len
            end = min(start + block_len, ts_len)
            if active_time[start:end].any():
                active_blocks.append(block_idx)

        neighbor_blocks = set()
        for block_idx in active_blocks:
            if block_idx - 1 >= 0:
                neighbor_blocks.add(block_idx - 1)
            if block_idx + 1 < n_time_blocks:
                neighbor_blocks.add(block_idx + 1)

        allowed_mask = np.zeros(cemnn.nA, dtype=bool)
        for block_idx in neighbor_blocks:
            start = block_idx * cemnn.env.ch_action_dim
            end = start + cemnn.env.ch_action_dim
            allowed_mask[start:end] = True
        if cemnn.env.end_action >= 0:
            allowed_mask[cemnn.env.end_action] = True
        cemnn.allowed_action_mask = allowed_mask

    def _get_warm_start_logits(self, cemnn, temp, weight):
        """
        Retrieve cached logits for this level/action-space and blend them into a fresh start.
        If shapes differ, ignore the cache.
        """
        key = self._make_level_key(cemnn)
        base = np.zeros(cemnn.nA, dtype=np.float64)
        cached = self._warm_start_logits.get(key)
        if cached is not None and cached.shape[0] == cemnn.nA:
            logits = _mix_logits_probabilities(
                base,
                cached,
                weight=weight,
                temp=temp
            )
        else:
            logits = base
        return logits, key

    def _update_warm_start_logits(self, key, logits):
        self._warm_start_logits[key] = np.array(logits, copy=True)

    @staticmethod
    def _remap_probs_from_maps(
        parent_probs,
        parent_block_map,
        child_block_map,
        parent_stop_idx,
        child_stop_idx,
        child_nA,
        eps=1e-12
    ):
        """
        Remap parent distribution onto child action space using block maps.
        If a child block overlaps multiple parent blocks, distribute that
        parent's probability mass proportionally to the overlap.
        """
        if parent_block_map.shape != child_block_map.shape:
            raise ValueError("Parent/child block maps must have the same shape.")

        parent_probs = np.array(parent_probs, dtype=np.float64, copy=False)
        parent_probs = np.maximum(parent_probs, eps)
        parent_probs /= parent_probs.sum()

        child_probs = np.zeros(child_nA, dtype=np.float64)
        parent_stop_idx = int(parent_stop_idx)
        child_stop_idx = int(child_stop_idx)

        stop_prior = 1.0 / float(child_nA)
        assigned_stop = stop_prior
        remaining = 1.0 - assigned_stop
        if remaining <= 0.0:
            raise ValueError("stop_prior leaves no probability mass for other actions.")
        effective_parent_probs = parent_probs.copy()
        effective_parent_probs[parent_stop_idx] = 0.0
        effective_parent_probs /= effective_parent_probs.sum()
        effective_parent_probs *= remaining

        # Spread each parent's mass across the child blocks it covers, weighted by overlap size.
        for pid in range(effective_parent_probs.shape[0]):
            if pid == parent_stop_idx:
                continue
            mask = parent_block_map == pid
            if not np.any(mask):
                continue
            child_counts = np.bincount(
                child_block_map[mask].astype(int),
                minlength=child_nA
            ).astype(np.float64)
            overlap_total = child_counts.sum()
            if overlap_total > 0:
                child_probs += effective_parent_probs[pid] * (child_counts / overlap_total)
            else:
                raise ValueError("...")

        # Map STOP explicitly (it has no spatial footprint)
        child_probs[child_stop_idx] += assigned_stop

        child_probs = np.maximum(child_probs, eps)
        child_probs /= child_probs.sum()
        return child_probs

    def _inherit_warm_start_logits(self, parent_logits, child_cemnn, temp):
        """Map parent logits to child's action space and seed child's warm-start cache."""
        if parent_logits is None:
            raise ValueError("None parent_logits")
        def _softmax_parent(x, temperature):
            z = (x - np.max(x)) / max(1e-8, temperature)
            e = np.exp(z, dtype=np.float64)
            return e / np.sum(e)
        parent_cemnn = None
        # Find parent cemnn by level index - 1 if available
        if hasattr(child_cemnn, "level_idx") and child_cemnn.level_idx is not None and child_cemnn.level_idx > 0:
            parent_cemnn = self.level_CEMNNs[child_cemnn.level_idx - 1]
        parent_map = parent_cemnn.env.block_index_map
        child_map = child_cemnn.env.block_index_map
        if parent_map is None or child_map is None:
            raise ValueError("Block maps are required for warm start remapping across levels.")
        child_probs = self._remap_probs_from_maps(
            _softmax_parent(parent_logits, temp),
            parent_map,
            child_map,
            parent_cemnn.stop_action,
            child_cemnn.stop_action,
            child_cemnn.nA,
        )
        mapped_logits = temp * np.log(child_probs)
        child_key = self._make_level_key(child_cemnn)
        blended = _mix_logits_probabilities(
            np.zeros(child_cemnn.nA, dtype=np.float64),
            mapped_logits,
            weight=self.hierarchy_warm_alpha,
            temp=temp
        )
        self._update_warm_start_logits(child_key, blended)
    
    def plan(
        self, current_cemnn,
        mask_curr,
        plan_num_sims_ratio, plan_steps, plan_iters
    ):
        logits_loc, level_key = self._get_warm_start_logits(current_cemnn, temp=self.temp, weight=self.warm_alpha)
        out_logits, pi_soft, v_star, best_seq = self._teacher_targets(
            logits_loc, mask_curr, current_cemnn,
            plan_num_sims_ratio, plan_steps, plan_iters,
            self.search_target, self.train_target, self.train_target_type, self.train_pisoft_func, 
            self.temp
        )
        if level_key is not None:
            # Force to do not use the best action in the logits
            mod_logits = np.array(out_logits, copy=True)
            best_action = int(best_seq[0])
            mod_logits[best_action] = -np.inf
            self._update_warm_start_logits(level_key, mod_logits)

        return out_logits, pi_soft, v_star, best_seq

    def collect_episode(
            self, levels, max_plan_calls, x_orig=None, nun=None,
            use_inf_params: bool = True, 
            record_teacher_targets: bool = True
    ):
        # Init episodes buffer
        buf_x_orig = {lv: [] for lv in levels}
        buf_x_nun = {lv: [] for lv in levels}
        buf_x_cf = {lv: [] for lv in levels}
        buf_mask = {lv: [] for lv in levels}
        buf_pi_star = {lv: [] for lv in levels}
        buf_v_star = {lv: [] for lv in levels}
        buf_e_len = []

        # Either draw random instance:
        state, _ = self.env.reset(x_orig=x_orig, nun=nun)
        # Pull tensors/arrays from env
        x_orig = state["x_orig"]  # (C_in, T) or (1,C_in,T)
        x_nun = state["nun"]
        x_cf = state["x_cf"]  # may be absent/None
        mask_t = state["mask"]

        # Ensure shapes (N=1, C, T)
        x_orig = self._ensure_batch3(x_orig).astype(np.float32)
        x_nun = self._ensure_batch3(x_nun).astype(np.float32)
        x_cf = self._ensure_batch3(x_cf).astype(np.float32)

        # Ensure mask is temporal and reduced to 1 channel for the model
        mask = self._ensure_batch3(mask_t).astype(np.float32)  # (1, C_m, T)

        # Receding-horizon episode driven by TEACHER on temporal masks
        level_idx = 0
        cemnn_idx = levels[level_idx]
        current_cemnn = self.level_CEMNNs[cemnn_idx]
        current_cemnn.env.reset(x_orig=state["x_orig"], nun=np.expand_dims(state["nun"], axis=0))
        mask_curr = mask[0].copy()  # (C_m, T) current env mask
        if self.block_non_adj_mask:
            self._set_cemnn_action_mask(current_cemnn, mask_curr)

        force_replan = False
        plan_every = self.inf_plan_every if use_inf_params else self.train_plan_every
        plan_num_sims_ratio = self.inf_num_simulations_ratio if use_inf_params else self.train_num_simulations_ratio
        plan_steps = self.inf_planning_steps if use_inf_params else self.train_planning_steps
        plan_iters = self.inf_cem_iters if use_inf_params else self.train_cem_iters
        for ext_step in range(max_plan_calls):
            # Decide whether to (re)plan now
            do_plan = force_replan or (ext_step % plan_every == 0)

            if do_plan:
                out_logits, pi_soft, v_star, best_seq = self.plan(
                    current_cemnn,
                    mask_curr,
                    plan_num_sims_ratio, plan_steps, plan_iters
                )

                best_seq = best_seq.tolist()
                best_action = best_seq.pop(0)
                force_replan = False

                if record_teacher_targets:
                    buf_x_orig[cemnn_idx].append(x_orig)
                    buf_x_nun[cemnn_idx].append(x_nun)
                    x_cf = self.env.compute_cfe(x_orig, x_nun, mask_curr)
                    buf_x_cf[cemnn_idx].append(x_cf if x_cf is not None else None)
                    buf_mask[cemnn_idx].append(self._ensure_batch3(mask_curr))
                    buf_pi_star[cemnn_idx].append(pi_soft)
                    buf_v_star[cemnn_idx].append(v_star)

            else:
                best_action = best_seq.pop(0)
                # If best action is equal to stop action then recalculate best_seq
                if best_action == current_cemnn.stop_action:
                    _, _, _, best_seq = self.plan(
                        current_cemnn,
                        mask_curr,
                        plan_num_sims_ratio, plan_steps, plan_iters
                    )
                    best_seq = best_seq.tolist()
                    best_action = best_seq.pop(0)

            # Advance env temporal mask via the chosen action
            mask_curr = current_cemnn.env.renew_mask_vec(mask_curr[np.newaxis, ...], np.array([best_action]))[0]
                    
            # Check if we need to change level
            if best_action == current_cemnn.stop_action:
                level_idx += 1

                # Advance a level in the hierarchy
                if level_idx < len(levels):
                    cemnn_idx = levels[level_idx]
                    if cemnn_idx in self.level_CEMNNs:
                        current_cemnn = self.level_CEMNNs[cemnn_idx]
                        # Inherit warm start from parent level logits into child action space
                        parent_key = self._make_level_key(self.level_CEMNNs[levels[level_idx-1]])
                        parent_logits = self._warm_start_logits[parent_key]
                        self._inherit_warm_start_logits(parent_logits, current_cemnn, temp=self.temp)
                        current_cemnn.env.reset(x_orig=state["x_orig"], nun=np.expand_dims(state["nun"], axis=0), mask=mask_curr)
                        if self.block_non_adj_mask:
                            self._set_cemnn_action_mask(current_cemnn, mask_curr)
                        force_replan = True
                    else:
                        break

                # Crete prunning CEMNN if allowed
                elif level_idx == len(levels):
                    if (self.pruning_ch_block_pct is not None) and (self.pruning_cemnn is None):
                        sum_mask_curr = np.clip(mask_curr.sum(axis=0), a_min=0, a_max=1).astype(int)
                        n_subsequences = np.count_nonzero(np.diff(sum_mask_curr, prepend=0, axis=0) == 1, axis=0)
                        # Create pruning CEMNN
                        env_params = copy(self.env_params)
                        env_params["latent_ts_block_len"] = 1
                        prunning_channel_clusters = self.channel_hierarchy.cut(channel_pct=self.pruning_ch_block_pct)
                        env_params["channel_clusters"] = prunning_channel_clusters
                        pruning_env = SubsequenceChannelDiscreteCFEnv(base_params=env_params, n_subsequences=n_subsequences, init_mask=mask_curr)
                        pruning_cemnn_params = {"env": pruning_env, **self.cemnn_params}
                        pruning_cemnn = CEMSequencePlanningNN(**pruning_cemnn_params)
                        pruning_cemnn.env.reset(x_orig=state["x_orig"], nun=np.expand_dims(state["nun"], axis=0), mask=mask_curr)
                        self.pruning_cemnn = pruning_cemnn
                        current_cemnn = pruning_cemnn
                        force_replan = True
                    else:
                        break

                # No more levels to advance
                else:
                    break

        # Append length to buffer
        buf_e_len.append(ext_step)

        # Reset environment to the end of the episode mask
        self.env.reset(x_orig=state["x_orig"], nun=np.expand_dims(state["nun"], axis=0), mask=mask_curr)

        return buf_x_orig, buf_x_nun, buf_x_cf, buf_mask, buf_pi_star, buf_v_star, buf_e_len

    @staticmethod
    def _teacher_targets(
        logits_loc, mask, cemnn, plan_num_sims_ratio, planning_steps, cem_iters, 
        search_target, train_target, train_target_type, pisoft_func,
        temp
    ):
        allowed_mask = getattr(cemnn, "allowed_action_mask")
        if allowed_mask.shape[0] == cemnn.nA:
            num_actions = int(np.count_nonzero(allowed_mask))
        else:
            num_actions = cemnn.nA
        num_sims = max(1, int(plan_num_sims_ratio * num_actions))

        out_logits, best_seq, best_fit, actions_seq, fitness, elite_idx = cemnn.run_cem(
            mask=mask,
            logits=logits_loc,
            num_sims=num_sims,
            planning_steps=planning_steps,
            iters=cem_iters,
            temp=temp,
            tabu_mode=cemnn.tabu_mode,
            target=search_target,
        )

        # Do no calculate targets
        if train_target_type is None:
            return out_logits, 0, 0, best_seq
        
        else:
            # Sampling distribution from cem final iteration
            if train_target_type == "cem":
                target_logits = out_logits
            
            # Best-K root action policy targets
            else:
                # Count elite
                if train_target == "basic":
                    root_actions = actions_seq[elite_idx, 0]
                    counts_root = np.bincount(root_actions, minlength=cemnn.nA).astype(np.float64)
                    # target_logits = np.log(counts_root + 1e-8)
                    # target_logits -= np.max(target_logits)
                    target_logits = counts_root + 1e-3
                    # pi_soft = counts_root + 1e-3
                    # pi_soft /= pi_soft.sum()
                # Weight by fitness
                elif train_target == "fitness_weighted":
                    # ToDo: This is not the same implementation as the search target!
                    root_actions = actions_seq[:, 0]
                    weights = np.maximum(fitness - fitness.min(), 0.0) + 1e-8
                    weighted_counts_root = np.zeros(cemnn.nA, dtype=np.float64)
                    np.add.at(weighted_counts_root, root_actions, weights)
                    # target_logits = np.log(weighted_counts_root + 1e-8)
                    # target_logits -= np.max(target_logits)
                    target_logits = weighted_counts_root + 1e-3
                    # pi_soft = weighted_counts + 1e-3
                    # pi_soft /= pi_soft.sum()
                else:
                    raise NotImplementedError(f"train_target '{train_target}' not implemented.")
            
            # Get policy target 
            if pisoft_func == "softmax":
                pi_soft = cemnn._softmax(target_logits, temp=temp)
            elif pisoft_func == "normalization":
                pi_soft = target_logits
                pi_soft /= pi_soft.sum()
            else:
                raise NotImplementedError("Only 'softmax' and 'normalization' pisoft_func are implemented.")

            # Value target (z-scored per-call)
            mu, sd = float(fitness.mean()), float(fitness.std()) + 1e-8
            v_star = np.array([(best_fit - mu) / sd], dtype=np.float32)
            # v_star = best_fit

            return out_logits, pi_soft.astype(np.float32), v_star, best_seq


class CEMSequencePlanningNN:
    """
    Cross-Entropy Method planner that reuses a DecisionModel's logits during planning.
    - Samples full sequences of length H (planning_steps), including STOP.
    - Updates shared action logits toward elite usage (until first STOP).
    - Reuses logits across plan() calls (global CEM logits + optional conditional reuse).
    - NEW: Blends NN logits (from per-block mean-pooled backbone features) into the sampler.
    """

    def __init__(
        self,
        elite_frac, alpha, tabu_mode,
        use_conditional_reuse,
        planning_credit,
        env
    ):
        self.env = env
        self.nA = env.action_space.n

        self.elite_frac = float(elite_frac)
        self.alpha = float(alpha)
        self.tabu_mode = tabu_mode
        self.planning_credit = bool(planning_credit)

        self.stop_action = self.env.end_action
        assert 0 <= self.stop_action < self.nA

        self.allowed_action_mask = None

        self.use_conditional_reuse = bool(use_conditional_reuse)

        self.rng = np.random.RandomState()

    def run_cem(self, mask, logits, iters, num_sims, planning_steps, temp, tabu_mode, target):
        # Run normal CEM execution
        track_credit = bool(self.planning_credit)
        initial_fitness = None
        if track_credit:
            initial_fitness = self.env.evaluate_counterfactual_batch(np.expand_dims(mask, axis=0)).astype(np.float64)
            initial_fitness = np.repeat(initial_fitness, num_sims, axis=0)

        allowed_mask = self.allowed_action_mask
        if allowed_mask is not None:
            allowed_mask = np.asarray(allowed_mask, dtype=bool)
            if allowed_mask.shape[0] != self.nA:
                allowed_mask = None
            elif not allowed_mask[self.stop_action]:
                allowed_mask = allowed_mask.copy()
                allowed_mask[self.stop_action] = True

        best_seq = None
        best_fit = -np.inf
        last_sequences = None
        last_fitness = None
        last_elites = None
        for _ in range(iters):
            fitness_progress = np.full((num_sims, planning_steps), np.nan, dtype=np.float64) if track_credit else None
            last_step_fitness = None
            # Current sampling distribution over actions (shared across steps)
            probs = self._softmax(logits, temp=temp)
            if allowed_mask is not None:
                probs = probs * allowed_mask
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                else:
                    probs = np.zeros_like(probs)
                    probs[self.stop_action] = 1.0

            # Sample sequences [N, H] with STOP support
            actions_seq = np.full((num_sims, planning_steps), self.stop_action, dtype=np.int32)
            # Copy base masks for this iteration's sampling pass
            masks_iter = np.repeat(mask[None, ...], num_sims, axis=0)
            active = np.ones(num_sims, dtype=bool)
            # Set tabu actions init
            if tabu_mode:
                # Tabu action
                used_mask = np.zeros((num_sims, self.nA), dtype=bool)  # True = forbidden, False=allowed

            for t in range(planning_steps):
                if not np.any(active):
                    break
                n_active = np.sum(active)

                # Sample actions
                if tabu_mode:
                    masked_probs = np.broadcast_to(probs, (n_active, self.nA)).copy()
                    # Zero all previously used block columns for these rows
                    used_sub = used_mask[active]  # (M, nA) booleans
                    if used_sub.any():
                        masked_probs[used_sub] = 0.0
                    if allowed_mask is not None:
                        masked_probs[:, ~allowed_mask] = 0.0
                    # Renormalize rows; if a row is all-zero (all blocks forbidden), force STOP
                    row_sums = masked_probs.sum(axis=1, keepdims=True)
                    zero_rows = row_sums.squeeze() == 0
                    if np.any(zero_rows):
                        masked_probs[zero_rows, :] = 0.0
                        masked_probs[zero_rows, self.stop_action] = 1.0
                        row_sums[zero_rows] = 1.0
                    masked_probs = masked_probs / row_sums
                    # --- Vectorized inverse-CDF sampling per row ---
                    u = self.rng.random(size=n_active)
                    u = np.minimum(u, 1.0 - 1e-12)
                    cum = np.cumsum(masked_probs, axis=1)
                    cum[:, -1] = 1.0
                    sampled = (cum < u[:, None]).sum(axis=1).astype(np.int32)
                else:
                    # Sample actions for active sims from the same probs
                    sampled = self.rng.choice(self.nA, size=n_active, p=probs)

                # Write into the full action matrix
                actions_seq[active, t] = sampled
                # Apply actions for active sims
                masks_iter[active] = self.env.renew_mask_vec(masks_iter[active], sampled)

                # Update tabu memory
                if tabu_mode:
                    if np.any(active):
                        # Mark newly used BLOCK actions as forbidden for the future steps
                        used_mask[active, sampled] = True

                # Deactivate those that chose STOP at this step
                active_idx = np.where(active)[0]
                active[active_idx] = (sampled != self.stop_action)

                if track_credit:
                    step_fitness = self.env.evaluate_counterfactual_batch(masks_iter).astype(np.float64)
                    fitness_progress[:, t] = step_fitness
                    last_step_fitness = step_fitness

            # Evaluate final masks (terminal fitness)
            if track_credit and last_step_fitness is not None:
                fitness = last_step_fitness
            else:
                fitness = self.env.evaluate_counterfactual_batch(masks_iter).astype(np.float64)

            # Track global best sequence seen so far
            i_best = int(np.argmax(fitness))
            if fitness[i_best] > best_fit:
                best_fit = float(fitness[i_best])
                best_seq = actions_seq[i_best].copy()

            # --- UPDATE LOGITS ---
            is_stop = (actions_seq == self.stop_action)  # (S, H)
            has_stop = is_stop.any(axis=1)  # (S,)
            first_stop_pos = np.where(has_stop, is_stop.argmax(axis=1), planning_steps - 1)  # (S,)
            # Build a valid grid of actions
            t_grid = np.arange(planning_steps)[None, :]  # (1, H)
            valid = t_grid <= first_stop_pos[:, None]  # (S, H) booleans
            if track_credit:
                step_weights = self._compute_step_weights(initial_fitness, fitness_progress, valid)
            else:
                step_weights = np.ones_like(actions_seq, dtype=np.float64)
            # Elites
            n_elite = max(1, int(self.elite_frac * num_sims))
            elite_idx = np.argsort(-fitness)[:n_elite]
            # Set last elites
            last_elites = elite_idx
            # Calculate new logits
            eps = 1e-8
            if target == "basic":
                # Get elite action counts until first STOP
                elite_actions = actions_seq[elite_idx, :]  # (E, H)
                elite_valid_actions = valid[elite_idx] # (E, H)
                elite_step_weights = step_weights[elite_idx]
                valid_actions = elite_actions[elite_valid_actions]  # (K,) flattened ints
                valid_weights = elite_step_weights[elite_valid_actions]
                weighted_counts = np.zeros(self.nA, dtype=np.float64)
                np.add.at(weighted_counts, valid_actions, valid_weights)

            elif target == "fitness_weighted":
                elite_actions = actions_seq[elite_idx, :]  # (E, H)
                elite_valid_actions = valid[elite_idx]     # (E, H)
                elite_weights = np.maximum(fitness[elite_idx] - fitness.min(), 0.0) + eps  # (E,)
                elite_weights = elite_weights / elite_weights.sum()  # normalize
                seq_idx, t_idx = np.where(elite_valid_actions)  # (M,), (M,)
                actions_flat = elite_actions[seq_idx, t_idx]  # (M,)
                elite_step_weights = step_weights[elite_idx]
                weights_flat = elite_weights[seq_idx] * elite_step_weights[seq_idx, t_idx]  # (M,)

                # Accumulate weighted counts per action (elites only)
                weighted_counts = np.zeros(self.nA, dtype=np.float64)
                np.add.at(weighted_counts, actions_flat, weights_flat)

                """valid_actions = actions_seq[valid]  # (M,) flattened ints
                weights = np.maximum(fitness - fitness.min(), 0.0) + eps
                target_logits = np.zeros(self.nA, dtype=np.float64)
                np.add.at(target_logits, valid_actions, weights)"""
            else:
                raise NotImplementedError("Only 'basic' and 'fitness_weighted' target_type are implemented.")
            
            # Compute target logits
            target_logits = np.log(weighted_counts + eps)
            target_logits -= np.max(target_logits)

            # Update shared logits toward elite action usage (until first STOP, inclusive)
            logits = _mix_logits_probabilities(
                logits,
                target_logits,
                weight=self.alpha,
                temp=temp
            )

            # Save last-iteration data (used to choose output action and build conditional reuse)
            last_sequences = actions_seq
            last_fitness = fitness
            
        return logits, best_seq, best_fit, last_sequences, last_fitness, last_elites

    @staticmethod
    def _softmax(x, temp=1.5, p_floor=1e-3):
        z = (x - np.max(x)) / max(1e-8, temp)
        e = np.exp(z, dtype=np.float64)
        p = e / np.sum(e)
        # probability floor + renormalize
        if p_floor > 0:
            p = np.maximum(p, p_floor)
            p /= np.sum(p)
        return p.astype(np.float64)

    @staticmethod
    def _compute_step_weights(initial_fitness, fitness_progress, valid_mask):
        """
        Compute per-step importance weights based on how much each step moves the fitness
        toward the final value. Uses the change from the previous step to the final fitness.
        """
        if initial_fitness is None or fitness_progress is None:
            # return np.ones_like(valid_mask, dtype=np.float64)
            raise ValueError("initial_fitness and fitness_progress must be provided for planning credit.")

        # Forward-fill fitness values for stopped trajectories
        # base: añade la fitness inicial como columna 0 para poder “tirar” de ella
        base = np.concatenate([initial_fitness[:, None], fitness_progress], axis=1)  # (S, H+1)
        final_fitness = base[:, -1][:, None]   # (S, 1), F_i
        prev_fitness = base[:, :-1]           # (S, H), f_{i, t-1}

        # Contribution: how much closer to final fitness each step starts from
        contrib = np.maximum(final_fitness - prev_fitness, 0.0)  # (S, H)
        contrib *= valid_mask.astype(np.float64)
        contrib_sum = contrib.sum(axis=1, keepdims=True)  # (S, 1)

        # If there’s no signal, fail instead of silently making something up
        if np.any(contrib_sum < 0.0):
            raise ValueError("Zero total contribution for some sequences; investigate fitness values.")

        weights = np.zeros_like(contrib)
        nonzero = (contrib_sum > 0).flatten()
        weights[nonzero] = contrib[nonzero, :] / contrib_sum[nonzero]

        return weights

