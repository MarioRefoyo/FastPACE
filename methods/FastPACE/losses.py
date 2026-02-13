import numpy as np
import torch


def adversarial_loss(sample, desired_label, model_wrapper):
    """
    Classifier's probability for the desired class y_nun given sample'.

    :param `sample`: New generated sample
    :param `y_nun`: Label of the desired class
    :param `device`: Device where the calculation is performed ("cpu" or "cuda")
    :return `prob`: Probability prediction of the model
    """
    y_pred_logits = model_wrapper.predict(sample)
    y_pred = np.argmax(y_pred_logits, axis=1)
    adversarial_prob = y_pred_logits[:, desired_label]
    return adversarial_prob, y_pred


def sparsity_loss(mask: np.ndarray, feature_dim, ts_dim):
    """
    Calculates the sparsity loss of a mask.

    :param `mask`: Numpy array representing the mask
    :return: Sparsity loss
    :raises `ValueError`: If the mask is empty
    """
    ones_pct = mask.sum(axis=(feature_dim, ts_dim)) / (mask.shape[feature_dim] * mask.shape[ts_dim])
    return -ones_pct


def contiguity_loss(mask: np.ndarray, feature_dim, ts_dim, gamma: float = 0.25):
    """
    Calculates the contiguity loss of a mask.

    :param `mask`: Numpy array representing the mask
    :param `gamma`: Adjustment parameter, default is 0.25
    :return: Contiguity loss
    :raises `ValueError`: If the mask is empty
    """
    subsequences = np.count_nonzero(np.diff(mask, prepend=0, axis=ts_dim) == 1, axis=(feature_dim, ts_dim))
    feature_avg_subsequences = subsequences / mask.shape[feature_dim]
    subsequences_pct = feature_avg_subsequences / (mask.shape[ts_dim] // 2)
    return -(subsequences_pct ** gamma)


def plausibility_loss(x_cfs: np.ndarray, outlier_calculator, original_outlier_score):
    """
    Calculates the plausibility loss of a mask.

    :param `mask`: Numpy array representing the mask
    :param `sample`: Original sample on which to perform the calculations
    :param `nun`: NUN of the original sample
    :return: Plausibility loss
    """
    if len(x_cfs.shape) == 2:
        x_cfs_ = np.expand_dims(x_cfs, axis=0)
    else:
        x_cfs_ = x_cfs

    if outlier_calculator is not None:
        outlier_scores = outlier_calculator.get_outlier_scores(x_cfs_)
        increase_outlier_scores = outlier_scores - original_outlier_score
        increase_outlier_scores[increase_outlier_scores < 0] = 0
        return -increase_outlier_scores
    else:
        return np.array([0] * x_cfs.shape[0])
