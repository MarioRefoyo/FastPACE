import numpy as np
import tensorflow as tf


def grad_cam_plus(model, data, layer_name, category_id=None):
    """Get a heatmap by Grad-CAM++.

    Args:
        model: A model object, build from tf.keras 2.X.
        data: An image ndarray.
        layer_name: A string, layer name in model.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    data_tensor = np.expand_dims(data, axis=0)
    conv_layer = model.get_layer(layer_name)
    heatmap_model = tf.keras.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(data_tensor)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=0)
    alphas /= alpha_normalization_constant
    alphas = np.where(np.isnan(alphas), 0, alphas)

    weights = np.maximum(conv_first_grad[0], 0)

    deep_linearization_weights = np.sum(weights*alphas, axis=0)
    grad_map = deep_linearization_weights*conv_output[0]
    grad_cam_map = np.sum(grad_map, axis=1)
    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    # Expand heatmap
    expanded_heatmap = np.interp(np.linspace(0, 1, data.shape[0]), np.linspace(0, 1, len(heatmap)), heatmap)
    expanded_heatmap = np.tile(expanded_heatmap.reshape(-1, 1), (1, data.shape[1]))
    return expanded_heatmap


def calculate_heatmap(model, x_orig):
    # Calculate importance heatmap
    conv_layer_names = [layer.name for layer in model.layers if 'conv1d' in layer.name]
    last_conv_layer = conv_layer_names[-1]
    heatmap = grad_cam_plus(model, x_orig, last_conv_layer)
    if np.isnan(heatmap).any():
        heatmap = np.zeros(heatmap.shape)
    return heatmap

