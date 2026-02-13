from abc import ABC, abstractmethod
from copy import copy

import gymnasium as gym
import numpy as np
from collections import deque, Counter

from methods.FastPACE import losses
from methods.FastPACE.rewards import IncrementalRewardCalculator, FinalRewardCalculator


class CFEnv(gym.Env, ABC):
    def __init__(
        self,
        X_train,
        nuns,
        model_wrapper,
        latent_ts_block_len,
        reward_type,
        non_valid_penalization,
        channel_clusters=[],
        outlier_calculator=None,
        weights_losses=None,
        mask_init="ones",
        max_steps=100,
        include_repetition_end=True,
        include_end_action=False,
        precompute_mask_changes=True,
        device="cuda",
    ):
        super().__init__()

        self.X_train = X_train
        self.nuns = nuns
        self.model_wrapper = model_wrapper
        self.model_wrapper.to(device)
        self.outlier_calculator = outlier_calculator
        self.latent_ts_block_len = latent_ts_block_len
        self.channel_clusters = channel_clusters
        self.reward_type = reward_type
        self.non_valid_penalization = non_valid_penalization
        self.weights = self.compute_weights(weights_losses)
        assert mask_init in ["ones", "zeros", "random", "all"]
        self.mask_init = mask_init
        self.max_steps = max_steps
        self.include_repetition_end = include_repetition_end
        self.include_end_action = include_end_action
        self.device = device

        self.n_channels = X_train.shape[1]
        self.ts_length = X_train.shape[2]
        self.ts_action_dim = int(np.ceil(self.ts_length / latent_ts_block_len))
        self.ch_action_dim = len(channel_clusters)
        self.observation_space = gym.spaces.Dict(
            {
                "x_orig": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_channels, self.ts_length), dtype=np.float32
                ),
                "nun": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_channels, self.ts_length), dtype=np.float32
                ),
                "x_cf": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.n_channels, self.ts_length), dtype=np.float32
                ),
                "mask": gym.spaces.Box(
                    low=0, high=1, shape=(self.n_channels, self.ts_length), dtype=np.bool_
                ),
            }
        )

        self.action_space, self.end_action = self._create_action_space(include_end_action)

        # Precompute mask changes and block map if discrete action space
        if precompute_mask_changes:
            self.precomputed_mask_changes, self.block_index_map = self.precompute_mask_changes()

        # Set early episode check function
        if include_repetition_end and include_end_action:
            self.check_early_end = lambda action: self._check_repeated_actions(action) or self.check_end_action(action)
        elif include_repetition_end:
            self.check_early_end = self._check_repeated_actions
        elif include_end_action:
            self.check_early_end = self._check_end_action
        else:
            self.check_early_end = lambda action: False

    @abstractmethod
    def _create_action_space(self, include_end_action):
        pass

    @abstractmethod
    def traduce_action_to_mask_changes(self, action, mask=None):
        """
        Updates the mask with the new action.
        """
        pass

    def precompute_mask_changes(self):
        if isinstance(self.action_space, gym.spaces.Discrete):
            precomputed_mask_changes = []
            block_map = np.zeros((self.n_channels, self.ts_length))
            for a in range(self.action_space.n):
                a_mask_changes = self.traduce_action_to_mask_changes(a)
                precomputed_mask_changes.append(a_mask_changes)
                block_map = block_map + a * a_mask_changes
            precomputed_mask_changes = np.array(precomputed_mask_changes)
            block_map = block_map.astype(int)
        else:
            precomputed_mask_changes = None
            block_map = None
        return precomputed_mask_changes, block_map

    def renew_mask(self, mask, action):
        if self.precomputed_mask_changes:
            mask_changes = self.precomputed_mask_changes[action]
        else:
            mask_changes = self.traduce_action_to_mask_changes(action)
        mask = np.logical_xor(mask, mask_changes).astype(np.float32)
        return mask

    def renew_mask_vec(self, masks, actions):
        batch_size = actions.shape[0]
        # Get vectorized current mask by indexing the array
        if self.precomputed_mask_changes is not None:
            mask_changes = self.precomputed_mask_changes[actions]
        else:
            if len(masks.shape) > 2:
                mask_changes = np.empty(masks.shape, dtype=bool)
            else:
                mask_changes = np.empty((1, masks.shape[0], masks.shape[1]), dtype=bool)
            for i in range(batch_size):
                mask_changes[i, :, :] = self.traduce_action_to_mask_changes(actions[i])

        # calculate updated masks
        updated_masks = np.logical_xor(masks, mask_changes).astype(np.float32)
        return updated_masks

    @staticmethod
    def _convert_action_dim_to_real_dim(value, block_len):
        return value*block_len

    def _check_repeated_actions(self, action) -> bool:
        """
        Checks whether the episode should be terminated based on the repetition pattern of the recent actions.

        Termination is triggered if any of the following conditions are met:
        - The last 5 actions are identical
        - At least 2 distinct actions appear 4 or more times each
        - At least 3 distinct actions appear 3 or more times each

        :param action: The action to be added to the buffer and checked
        :return bool: True if any of the above conditions are satisfied, indicating the episode should end
        """
        try:
            action_hash = hash(tuple(action.tolist()))
        except Exception as e:
            action_hash = int(action)
        self.actions_buffer.append(action_hash)

        if len(self.actions_buffer) >= 5:
            last_actions = list(self.actions_buffer)[-5:]
            if all(a == last_actions[0] for a in last_actions):
                return True

        counts = Counter(self.actions_buffer)

        repeated_4 = sum(1 for count in counts.values() if count >= 4)
        if repeated_4 >= 2:
            return True

        repeated_3 = sum(1 for count in counts.values() if count >= 3)
        if repeated_3 >= 3:
            return True

        return False

    def _check_end_action(self, action):
        if isinstance(action, int):
            if self.end_action == action:
                return True
            else:
                return False
        else:
            return False

    def step(self, action):
        """
        Executes a simulation step.

        :param `action`: Tuple of the form (beginning of the transformation, size of the transformation)
        :return `observation`: The updated state of the environment after the action
        :return `reward`: A scalar value indicating the reward for the current step
        :return `done`: A boolean indicating if the episode has finished
        :return `truncated`: A boolean indicating if the episode was cut short
        :return `info`: A dictionary with additional information
        """
        self.steps += 1
        self.mask = self.renew_mask(copy(self.mask), action)
        x_cf = self.compute_cfe(self.x_orig, self.nun, self.mask)
        observation = {"x_orig": self.x_orig, "nun": self.nun, "x_cf": x_cf, "mask": self.mask}

        done = self.check_early_end(action)
        truncated = self.check_end()
        episode_end = done or truncated
        reward = self.reward_calculator.calculate_reward(
            self.x_orig, self.nun, x_cf, self.mask, self.desired_label,
            self.weights, self.model_wrapper, self.steps, episode_end
        )

        info = self._get_info()

        return observation, reward, done, truncated, info

    def render(self):
        super().render()
        # TODO: Add a method to render an episode
        raise NotImplementedError

    def reset(self, x_orig=None, nun=None, mask=None, seed=None, options=None):
        super().reset(seed=seed)

        # Get new x_orig and nun
        if x_orig is not None:
            self.x_orig = copy(x_orig)
            self.nun = copy(nun)[0, :, :]
        else:
            self.x_orig, self.nun = self.get_random_train_data()
        self.n_channels, self.ts_length = self.x_orig.shape[0], self.x_orig.shape[1]

        # Get mask initialization strategy
        if mask is not None:
            self.mask = mask
        else:
            if self.mask_init == "all":
                chosen_strategy = np.random.choice(["ones", "zeros", "random"])
            else:
                chosen_strategy = self.mask_init
            # Initialize mask
            if chosen_strategy == "ones":
                self.mask = np.ones(self.x_orig.shape, dtype=np.bool_)
            elif chosen_strategy == "zeros":
                self.mask = np.zeros(self.x_orig.shape, dtype=np.bool_)
            elif chosen_strategy == "random":
                block_masks = np.random.choice([False, True], size=self.action_dim)
                block_masks = block_masks.reshape((1, block_masks.shape[0]))
                block_masks = np.repeat(block_masks, self.n_channels, axis=0)
                self.mask = np.repeat(block_masks, self.latent_ts_block_len, axis=1)
            else:
                raise ValueError("Not valid mask_init")

        # Get desired label
        self.x_orig_label = self.model_wrapper.predict_class(self.x_orig)[0]
        self.desired_label = self.model_wrapper.predict_class(self.nun)[0]

        # Define reward calculator
        if self.reward_type == "incremental":
            self.reward_calculator = IncrementalRewardCalculator(self.non_valid_penalization)
        elif self.reward_type == "final":
            self.reward_calculator = FinalRewardCalculator(self.non_valid_penalization)
        else:
            raise ValueError("Not valid reward_type")

        self.steps = 0

        # Calculate initial outlier score
        self.original_outlier_score = self.outlier_calculator.get_outlier_scores(self.x_orig)[0]

        # Calculate nun initial fitness
        self.nun_fitness = self.reward_calculator.compute_losses(
            self.x_orig, self.nun, self.nun, np.ones(self.x_orig.shape, dtype=np.bool_),
            self.desired_label, self.weights, self.non_valid_penalization, self.model_wrapper, 
            self.outlier_calculator, self.original_outlier_score
        )
        x_cf = self.compute_cfe(self.x_orig, self.nun, self.mask)
        _ = self.reward_calculator.calculate_reward(
            self.x_orig, self.nun, x_cf, self.mask,
            self.desired_label, self.weights, self.model_wrapper, self.outlier_calculator,
            self.original_outlier_score,
            step=self.steps
        )

        self.actions_buffer = deque(maxlen=16)
        observation = {"x_orig": self.x_orig, "nun": self.nun, "x_cf": self.x_orig, "mask": self.mask}
        info = self._get_info()

        return observation, info

    def get_random_train_data(self):
        idx = np.random.choice(self.X_train.shape[0])
        x_orig = self.X_train[idx]
        nun = self.nuns[idx]
        # ToDo: allow multiple selection of nun. Right now only one NUN is taken
        nun = nun[0, :, :]
        return x_orig, nun

    @staticmethod
    def compute_weights(weights):
        """
        Normalizes a list of weights so that they sum to 1. Weight values equal to 0 are allowed and will be preserved in the normalization.

        :param `weights`: List of 4 numerical weights, one for each loss component
        :return `normalized`: Normalized weights whose sum is equal to 1 as a dictionary with where the keys are the different losses ("adversarial", "sparsity", "contiguity" and "plausability")
        :raise `ValueError`: If the list does not contain exactly 4 elements
        :raise `ValueError`: If any weight is less than 0
        :raise `ValueError`: If the sum of all weights is 0
        """
        if not weights:
            return [1/4, 1/4, 1/4, 1/4]

        if len(weights) != 4:
            raise ValueError("The list must be of size 4, one for each loss.")
        if any(num for num in weights) < 0:
            raise ValueError("All weights must be greater or equal to 0.")
        if sum(weights) == 0:
            raise ValueError("The sum of the weights cannot be equal to 0.")

        weights = np.array(weights)
        normalized_weights = weights / weights.sum()

        return normalized_weights

    @staticmethod
    def compute_cfe(x_orig, nun, mask):
        """
        Obtains the new mask by applying the mask.

        :return `new_signal`: The new signal
        """
        if len(mask.shape) > 2:
            batch_size = mask.shape[0]
            ext_x_orig = np.tile(x_orig, (batch_size, 1, 1))
            ext_nun = np.tile(nun, (batch_size, 1, 1))
            return np.where(mask == 1, ext_nun, ext_x_orig)
        else:
            return np.where(mask == 1, nun, x_orig)

    def check_end(self):
        """
        Verifies wether the episode is terminated by external boundary.

        :param `n`: Number of steps to compute
        :return `bool`: Boolean indicating if the episode must end up now
        """
        return False if self.steps < self.max_steps else True

    def _get_info(self):
        """
        Obtains the information of the step.

        :return `info`: Experience tuple of the step, wich is of the form => {S_t, A_t, R_t+1, S_t+1} <--- Add more info???
        """
        # TODO: Complete the method
        return {
            "step": self.steps,
            "mask": self.mask.copy(),
            # "loss": ...,
        }

    def get_cfe(self):
        best_mask = self.reward_calculator.best_fitness_mask
        x_cf = self.compute_cfe(self.x_orig, self.nun, best_mask)
        adv, y_pred = losses.adversarial_loss(x_cf, self.desired_label, self.model_wrapper)
        adv, y_pred = adv[0], y_pred[0]
        spa = -losses.sparsity_loss(best_mask, feature_dim=0, ts_dim=1)
        sub = -losses.contiguity_loss(best_mask, feature_dim=0, ts_dim=1, gamma=1)
        pla = -losses.plausibility_loss(x_cf, self.outlier_calculator, self.original_outlier_score)[0]
        valid = int(y_pred == self.desired_label)
        is_nun = np.all(self.reward_calculator.best_fitness_mask)
        return {
            "cfs": x_cf,
            "x_orig": self.x_orig,
            "nun": self.nun,
            "weights": self.weights,
            "valid": valid,
            "is_nun": is_nun,
            "adv": adv,
            "sparsity": spa,
            "subsequences": sub,
            "plausibility": pla,
            "fitness": self.reward_calculator.best_fitness,
            "step": self.reward_calculator.best_fitness_step,
            "best_mask": self.reward_calculator.best_fitness_mask,
            "nun_fitness": self.nun_fitness,
            "improvement_over_nun": self.reward_calculator.best_fitness - self.nun_fitness
        }

    def evaluate_counterfactual_batch(self, masks):
        x_cfs = self.compute_cfe(self.x_orig, self.nun, masks)
        adv, y_pred = losses.adversarial_loss(x_cfs, self.desired_label, self.model_wrapper)
        spa = losses.sparsity_loss(masks, feature_dim=1, ts_dim=2)
        sub = losses.contiguity_loss(masks, feature_dim=1, ts_dim=2)
        pla = losses.plausibility_loss(x_cfs, self.outlier_calculator, self.original_outlier_score)
        fitness_values = np.stack([adv, spa, sub, pla], axis=1)
        fitness = np.sum(fitness_values * self.weights, axis=1)
        not_valids = y_pred != self.desired_label
        penalization_vec = not_valids * self.non_valid_penalization
        fitness = fitness + penalization_vec
        return fitness

    def __str__(self):
        return f"<{self.__class__.__name__}>"

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_wrapper.__class__.__name__}, weights={self.weights})"

class SubsequenceChannelDiscreteCFEnv(CFEnv):
    def __init__(
        self,
        base_params,
        n_subsequences,
        init_mask,
    ):
        super().__init__(**base_params, precompute_mask_changes=False)
        # Override ts_action_dim to n_subsequences
        self.latent_ts_block_len = 1
        self.n_subsequences = n_subsequences
        self.ts_action_dim = n_subsequences

        # Init start and end mask points
        sum_mask = np.clip(init_mask.sum(axis=0), a_min=0, a_max=1).astype(int)
        aux_init_end = np.diff(sum_mask, prepend=0, append=0, axis=0)
        starts = np.where(aux_init_end == 1)[0]
        ends = np.where(aux_init_end == -1)[0]
        # Check current number of subsequences. If 0 then nothing to do else there is an error.
        if (len(starts) != self.n_subsequences) or (len(starts) != len(ends)):
            raise ValueError("Number of calculated subsequences does not match the passed number")
        self.starts = starts
        self.ends = ends

        self.action_space, self.end_action = self._create_action_space(base_params["include_end_action"])
        # Precompute mask changes if discrete action space
        self.precomputed_mask_changes, self.block_index_map = self.precompute_mask_changes()

    def _create_action_space(self, include_end_action):
        if include_end_action:
            total_actions = self.ts_action_dim * self.ch_action_dim + 1
            end_action = total_actions - 1
        else:
            total_actions = self.ts_action_dim * self.ch_action_dim
            end_action = -1
        action_space = gym.spaces.Discrete(n=total_actions)
        return action_space, end_action

    def traduce_action_to_mask_changes(self, action, mask=None):
        if action == self.end_action:
            # No changes
            return np.zeros((self.n_channels, self.ts_length), dtype=bool)
        else:
            # Calculate
            subsequence = action // self.ch_action_dim
            channel_group = action % self.ch_action_dim

            ts_start, ts_end = self.starts[subsequence], self.ends[subsequence]

            ch_idx = self.channel_clusters[channel_group]

            mask_changes = np.zeros((self.n_channels, self.ts_length))
            mask_changes[ch_idx, ts_start:ts_end] = np.logical_not(mask_changes[ch_idx, ts_start:ts_end])
            return mask_changes


class JointTimeChannelDiscreteCFEnv(CFEnv):

    def _create_action_space(self, include_end_action):
        if include_end_action:
            total_actions = self.ts_action_dim * self.ch_action_dim + 1
            end_action = total_actions - 1
        else:
            total_actions = self.ts_action_dim * self.ch_action_dim
            end_action = -1
        action_space = gym.spaces.Discrete(n=total_actions)
        return action_space, end_action

    @staticmethod
    def _from_linear(action, ch_action_dim):
        """Decode linear action -> (k_t, k_c)."""
        time_block = action // ch_action_dim
        channel_group = action % ch_action_dim
        return int(time_block), int(channel_group)

    # ---------- API: mask update + translation ----------
    def traduce_action_to_mask_changes(self, action, mask=None):
        """
        Return a mask of changes (same shape as mask) implied by 'action'.
        Compatible with both signatures used in your other envs (accepts optional mask).
        """
        if action == self.end_action:
            # No changes
            return np.zeros((self.n_channels, self.ts_length), dtype=bool)
        else:
            time_block, channel_group = self._from_linear(action, self.ch_action_dim)

            ts_start = self._convert_action_dim_to_real_dim(time_block, self.latent_ts_block_len)
            ts_block_length = self.latent_ts_block_len
            ts_end = min(ts_start + ts_block_length, self.ts_length)

            ch_idx = self.channel_clusters[channel_group]

            changes = np.zeros((self.n_channels, self.ts_length), dtype=bool)
            # As in your other envs, we toggle against zeros via logical_not => sets the region to True
            changes[ch_idx, ts_start:ts_end] = np.logical_not(changes[ch_idx, ts_start:ts_end])
            return changes


class SingleTSDiscreteMaskCFEnv(CFEnv):

    def _create_action_space(self, include_end_action):
        # Ignore include_end_action
        # assert include_end_action is False, "BoxCFEnv does not support include_end_action"
        end_action = -1
        action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_channels, self.ts_length), dtype=np.bool_)
        return action_space, end_action

    def traduce_action_to_mask_changes(self, action, mask=None):
        """
        Return a mask of changes (same shape as mask) implied by 'action'.
        Compatible with both signatures used in your other envs (accepts optional mask).
        """
        if isinstance(action, int):
            if action == self.end_action:
                # No changes
                return np.zeros((self.n_channels, self.ts_length), dtype=bool)
        else:
            return action
