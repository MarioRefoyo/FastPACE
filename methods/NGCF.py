import numpy as np
import copy

from .NG.utils import findSubarray
from .counterfactual_common import CounterfactualMethod


class NGCF(CounterfactualMethod):
    def __init__(self, model, fi_method):
        super().__init__(model)
        self.fi_method = fi_method

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None, y_true_orig=None):
        if desired_target is None:
            raise ValueError("The parameter desired_target must be provided.")

        # Init important variables
        subarray_length = 1
        # Calculate importance heatmap
        heatmap = self.fi_method.calculate_feature_importance(nun_example)

        # Init comparison of x_cf
        most_influencial_array = findSubarray(heatmap, subarray_length)
        starting_point = np.where(heatmap == most_influencial_array[0])[0][0]
        x_cf = np.concatenate((
            x_orig[:starting_point],
            nun_example[starting_point:subarray_length + starting_point],
            x_orig[subarray_length + starting_point:]
        ))
        pred_probs = self.model_wrapper.predict(x_cf)
        pred_class = np.argmax(pred_probs, axis=1)[0]

        while pred_class != desired_target:
            subarray_length += 1
            most_influencial_array = findSubarray(heatmap, subarray_length)
            starting_point = np.where(heatmap == most_influencial_array[0])[0][0]
            x_cf = np.concatenate((
                x_orig[:starting_point],
                nun_example[starting_point:subarray_length + starting_point],
                x_orig[subarray_length + starting_point:]
            ))
            pred_probs = self.model_wrapper.predict(x_cf)
            pred_class = np.argmax(pred_probs, axis=1)[0]

        result = {'cf': np.expand_dims(x_cf, axis=0)}
        return result
