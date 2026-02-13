import numpy as np
import copy

from .MultiSubSpaCE.MOEvolutionaryOptimizers import IntegratedPruningNSubsequenceEvolutionaryOptimizer, SubsequencePruningEvolutionaryOptimizer
from .MultiSubSpaCE.FitnessFunctions import fitness_function_mo, fitness_function_mo_os
from .counterfactual_common import CounterfactualMethod


class MultiSubSpaCECF(CounterfactualMethod):
    def __init__(self, model_wrapper, outlier_calculator, fi_method,
                 grouped_channels_iter, individual_channels_iter, pruning_iter,
                 plausibility_objective="ios",
                 population_size=100,
                 change_subseq_mutation_prob=0.05, add_subseq_mutation_prob=0,
                 integrated_pruning_mutation_prob=0.05, final_pruning_mutation_prob=0.5,
                 init_pct=0.4, reinit=True, init_random_mix_ratio=0.5,
                 invalid_penalization=100,):
        super().__init__(model_wrapper)

        self.outlier_calculator = outlier_calculator
        self.fi_method = fi_method
        self.grouped_channels_iter = grouped_channels_iter
        self.individual_channels_iter = individual_channels_iter
        self.pruning_iter = pruning_iter
        if plausibility_objective == "ios":
            used_fitness_function = fitness_function_mo
        elif plausibility_objective == "os":
            used_fitness_function = fitness_function_mo_os
        else:
            raise ValueError('Not valid plausibility_objective. Choose "ios" or "os".')


        # Init Genetic Optimizer
        if grouped_channels_iter > 0:
            self.g_channels_optimizer = IntegratedPruningNSubsequenceEvolutionaryOptimizer(
                used_fitness_function, self.predict_function,
                population_size, grouped_channels_iter,
                change_subseq_mutation_prob, add_subseq_mutation_prob, integrated_pruning_mutation_prob,
                init_pct, reinit, init_random_mix_ratio,
                invalid_penalization,
                False
            )
        if individual_channels_iter > 0:
            self.i_channels_optimizer = IntegratedPruningNSubsequenceEvolutionaryOptimizer(
                used_fitness_function, self.predict_function,
                population_size, individual_channels_iter,
                change_subseq_mutation_prob, add_subseq_mutation_prob, integrated_pruning_mutation_prob,
                init_pct, reinit, init_random_mix_ratio,
                invalid_penalization,
                True
            )

        if pruning_iter > 0:
            self.subsequence_pruning_optimizer = IntegratedPruningNSubsequenceEvolutionaryOptimizer(
                used_fitness_function, self.predict_function,
                population_size, pruning_iter,
                0, add_subseq_mutation_prob, final_pruning_mutation_prob,
                init_pct, reinit, init_random_mix_ratio,
                invalid_penalization,
                True
            )

    def search_mask(self, subsequence_optimizer, x_orig, nun_example, desired_target, combined_heatmap, init_mask):
        subsequence_optimizer.init(
            x_orig, nun_example, desired_target,
            self.model_wrapper,
            init_mask=init_mask,
            outlier_calculator=self.outlier_calculator,
            importance_heatmap=combined_heatmap
        )

        # Calculate counterfactual
        counterfactual_mask, best_avg_fitness_evolution = subsequence_optimizer.optimize()
        if counterfactual_mask is None:
            print(f'Failed to converge for sample')
            x_cfs = copy.deepcopy(np.expand_dims(x_orig, axis=0))
        else:
            x_cfs = subsequence_optimizer.get_counterfactuals(
                x_orig, nun_example, counterfactual_mask
            )

        return counterfactual_mask, x_cfs, best_avg_fitness_evolution

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None):
        # Init values
        fitness_evolution = []

        # Calculate importance heatmap
        heatmap_x_orig = self.fi_method.calculate_feature_importance(x_orig)
        heatmap_nun = self.fi_method.calculate_feature_importance(nun_example)
        combined_heatmap = (heatmap_x_orig + heatmap_nun) / 2

        # Start optimization process:
        # If there is a combination of grouped channel iterations, execute grouped search first, then use solution as
        # starting point for the individual search.
        init_mask = None
        if self.grouped_channels_iter > 0:
            grouped_counterfactual_mask, x_cfs, fitness_evolution_grouped = self.search_mask(
                self.g_channels_optimizer, x_orig, nun_example, desired_target, combined_heatmap,
                init_mask=init_mask
            )
            # Extend mask to init shape
            grouped_counterfactual_mask = np.tile(grouped_counterfactual_mask, (1, x_orig.shape[1]))
            fitness_evolution = fitness_evolution + fitness_evolution_grouped
            init_mask = grouped_counterfactual_mask

        # Now search using the individual search if required. Grouped counterfactual mask would be active in case
        # grouped search mode has been used before, and if it reached a result.
        if self.individual_channels_iter > 0:
            individual_counterfactual_mask, x_cfs, fitness_evolution_individual = self.search_mask(
                self.i_channels_optimizer, x_orig, nun_example, desired_target, combined_heatmap,
                init_mask=init_mask
            )
            fitness_evolution = fitness_evolution + fitness_evolution_individual
            init_mask = individual_counterfactual_mask

        # Now pruning will be performed if desired. Grouped counterfactual mask would be active in case
        # grouped search mode has been used before, and if it reached a result.
        if self.pruning_iter > 0:
            individual_counterfactual_mask, x_cfs, fitness_evolution_individual = self.search_mask(
                self.subsequence_pruning_optimizer, x_orig, nun_example, desired_target, combined_heatmap,
                init_mask=init_mask
            )
            fitness_evolution = fitness_evolution + fitness_evolution_individual

        # Get final result in format
        result = {'cfs': x_cfs, 'fitness_evolution': fitness_evolution}

        return result
