from abc import ABC, abstractmethod
from copy import copy
import numpy as np


from methods.FastPACE import losses


class RewardCalculator(ABC):
    def __init__(self, non_valid_penalization):
        super().__init__()

        # Init best values
        self.best_fitness = -np.inf
        self.best_fitness_step = 0
        self.best_fitness_mask = None

        self.non_valid_penalization = non_valid_penalization

    @staticmethod
    def compute_losses(x_orig, nun, x_cf, mask, desired_label, weights, non_valid_penalization, model_wrapper,
                       outlier_calculator, original_outlier_score) -> float:
        adv, y_pred = losses.adversarial_loss(x_cf, desired_label, model_wrapper)
        adv, y_pred = adv[0], y_pred[0]
        spa = losses.sparsity_loss(mask, feature_dim=0, ts_dim=1)
        sub = losses.contiguity_loss(mask, feature_dim=0, ts_dim=1)
        pla = losses.plausibility_loss(x_cf, outlier_calculator, original_outlier_score)[0]
        fitness_values = np.array([adv, spa, sub, pla])
        fitness = np.sum(fitness_values * weights)
        if y_pred != desired_label:
            fitness += non_valid_penalization
        return fitness

    def calculate_reward(self, x_orig, nun, x_cf, mask, desired_label, weights, model_wrapper, outlier_calculator,
                         original_outlier_score, step, episode_end=False) -> float:
        fitness = self.compute_losses(
            x_orig, nun, x_cf=x_cf, mask=mask, desired_label=desired_label,
            weights=weights, non_valid_penalization=self.non_valid_penalization, model_wrapper=model_wrapper,
            outlier_calculator=outlier_calculator, original_outlier_score=original_outlier_score
        )
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_fitness_step = step
            self.best_fitness_mask = mask.copy()

        # Calculate reward
        reward = self.calculate_reward_specific(fitness, episode_end)
        return reward

    @abstractmethod
    def calculate_reward_specific(self, fitness, episode_end):
        pass


class IncrementalRewardCalculator(RewardCalculator):
    def __init__(self, non_valid_penalization):
        super().__init__(non_valid_penalization)
        self.last_fitness = self.best_fitness

    def calculate_reward_specific(self, fitness, episode_end):
        reward = fitness - self.last_fitness
        self.last_fitness = fitness
        return reward


class FinalRewardCalculator(RewardCalculator):
    def calculate_reward_specific(self, fitness, episode_end):
        if episode_end:
            return fitness
        else:
            return 0
