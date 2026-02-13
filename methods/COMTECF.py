import numpy as np
import copy

from .COMTE.Optimization import BruteForceSearch, OptimizedSearch
from .counterfactual_common import CounterfactualMethod


class COMTECF(CounterfactualMethod):
    def __init__(self, model_wrapper, x_train, y_train, number_distractors,
                 max_attempts=1000, max_iter=1000, restarts=5, reg=0.8, silent=False):
        super().__init__(model_wrapper)
        self.x_train = x_train
        self.y_train = y_train
        self.number_distractors = number_distractors
        self.max_attempts = max_attempts
        self.max_iter = max_iter
        self.silent = silent
        self.restarts = restarts
        self.reg = reg

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None):
        opt = OptimizedSearch(
            self.predict_function,
            self.x_train,
            self.y_train,
            silent=self.silent,
            threads=1,
            num_distractors=self.number_distractors,
            max_attempts=self.max_attempts,
            maxiter=self.max_iter,
            restarts=self.restarts,
            reg=self.reg,
        )

        try:
            x_cf, predicted_label = opt.explain(x_orig, to_maximize=desired_target)
        except AttributeError as msg:
            print(f'{msg}')
            print(f'COMTE failed to find a NUN so original instance is returned')
            x_cf = copy.deepcopy(x_orig)

        result = {'cf': x_cf}

        return result
