from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F


class FeatureImportanceMethod(ABC):

    def __init__(self, backend='tf'):
        self.backend = backend

    @abstractmethod
    def calculate_feature_importance(self, instance, desired_target=None, **kwargs):
        pass


class NoneFI(FeatureImportanceMethod):

    def __init__(self, backend):
        super().__init__(backend)

    def calculate_feature_importance(self, instance, desired_target=None, **kwargs):
        heatmap = np.zeros(instance.shape)
        return heatmap


class GraCAMPlusFI(FeatureImportanceMethod):

    def __init__(self, backend, model_wrapper):
        super().__init__(backend)
        self.model_wrapper = model_wrapper

        if self.backend == "tf":
            # Detect convolutional layers
            conv_layer_names = [layer.name for layer in model_wrapper.model.layers if 'conv1d' in layer.name]
            if not conv_layer_names:
                raise ValueError("Model has not convolutional layers. "
                                 "GradCAM++ needs convolutional layers to generate an explanation")
            else:
                self.last_conv_layer_name = conv_layer_names[-1]
            conv_layer = model_wrapper.model.get_layer(self.last_conv_layer_name)
            self.heatmap_model = tf.keras.Model([model_wrapper.model.inputs], [conv_layer.output, model_wrapper.model.output])
        elif self.backend == "torch":
            self.activations = None
            self.gradients = {}

            conv_layers = [module for module in model_wrapper.model.modules() if isinstance(module, torch.nn.Conv1d)]
            if not conv_layers:
                raise ValueError("Model has no Conv1d layers. GradCAM++ requires convolutional layers.")
            self.last_conv_layer = conv_layers[-1]
            # Register hook to get gradients and activations
            self._register_hooks()
        else:
            raise ValueError("Not valid backend")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.last_conv_layer.register_forward_hook(forward_hook)
        self.last_conv_layer.register_full_backward_hook(backward_hook)

    def _calculate_feature_importance_tf(self, instance, desired_target=None):
        # Calculate importance heatmap
        data_tensor = np.expand_dims(instance, axis=0)

        with tf.GradientTape() as gtape1:
            with tf.GradientTape() as gtape2:
                with tf.GradientTape() as gtape3:
                    conv_output, predictions = self.heatmap_model(data_tensor)
                    if desired_target is None:
                        category_id = np.argmax(predictions[0])
                    output = predictions[:, category_id]
                    conv_first_grad = gtape3.gradient(output, conv_output)
                conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
            conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

        global_sum = np.sum(conv_output, axis=(0, 1))

        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

        alphas = alpha_num / alpha_denom
        alpha_normalization_constant = np.sum(alphas, axis=0)
        alphas /= alpha_normalization_constant
        alphas = np.where(np.isnan(alphas), 0, alphas)

        weights = np.maximum(conv_first_grad[0], 0)

        deep_linearization_weights = np.sum(weights * alphas, axis=0)
        grad_map = deep_linearization_weights * conv_output[0]
        grad_cam_map = np.sum(grad_map, axis=1)
        heatmap = np.maximum(grad_cam_map, 0)
        max_heat = np.max(heatmap)
        if max_heat == 0:
            max_heat = 1e-10
        heatmap /= max_heat

        # Expand heatmap
        expanded_heatmap = np.interp(np.linspace(0, 1, instance.shape[0]), np.linspace(0, 1, len(heatmap)), heatmap)
        expanded_heatmap = np.tile(expanded_heatmap.reshape(-1, 1), (1, instance.shape[1]))

        if np.isnan(expanded_heatmap).any():
            expanded_heatmap = np.zeros(expanded_heatmap.shape)
        return expanded_heatmap

    def _calculate_feature_importance_torch(self, instance, desired_target=None):
        # Prepare input
        device = next(self.model_wrapper.model.parameters()).device
        data_tensor = torch.tensor(instance[None], dtype=torch.float32, device=device, requires_grad=True)
        # Swap axes.... Improve this...
        time_len = instance.shape[0]
        features = instance.shape[1]
        data_tensor = torch.swapaxes(data_tensor, 1, 2)

        # Forward pass
        output = self.model_wrapper.model(data_tensor)
        if desired_target is None:
            category_id = output.argmax(dim=1).item()
        else:
            category_id = desired_target
        selected_output = output[:, category_id]

        # Backward
        self.model_wrapper.model.zero_grad()
        selected_output.backward(retain_graph=True)

        A = self.activations[0]  # [C, T]
        dY_dA = self.gradients[0]  # [C, T]

        # Closed-form alpha computation
        alpha_numer = dY_dA.pow(2)  # [C, T]
        alpha_denom = (
                2 * alpha_numer +
                A * dY_dA.pow(3)
        ).sum(dim=1, keepdim=True)  # [C, 1]
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.tensor(1e-10, device=A.device))
        alpha = alpha_numer / alpha_denom  # [C, T]

        # Weights: alpha * positive_grad
        positive_grad = F.relu(dY_dA)  # [C, T]
        weights = (alpha * positive_grad).sum(dim=1)  # [C]

        # Weighted sum of activations
        grad_cam_map = torch.sum(weights[:, None] * A, dim=0)  # [T]
        heatmap = F.relu(grad_cam_map)

        # Normalize
        heatmap -= heatmap.min()
        heatmap /= heatmap.max() + 1e-10
        heatmap = heatmap.detach().cpu().numpy()

        # Interpolate to match input shape
        interp_heatmap = np.interp(np.linspace(0, 1, time_len), np.linspace(0, 1, len(heatmap)), heatmap)
        expanded_heatmap = np.tile(interp_heatmap.reshape(-1, 1), (1, features))

        if np.isnan(expanded_heatmap).any():
            expanded_heatmap = np.zeros_like(expanded_heatmap)

        return expanded_heatmap

    def calculate_feature_importance(self, instance, desired_target=None, **kwargs):
        if self.backend == "tf":
            return self._calculate_feature_importance_tf(instance, desired_target)
        elif self.backend == "torch":
            return self._calculate_feature_importance_torch(instance, desired_target)
        else:
            raise ValueError("Not valid backend.")
