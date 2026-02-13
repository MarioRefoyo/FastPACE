import numpy as np


def fitness_function_mo(ms, predicted_probs, desired_class, outlier_scores, original_outlier_score,
                        invalid_penalization_scalar):

    # Init objective fitness matrix with nans
    objectives_fitness = np.empty((ms.shape[0], 4))
    objectives_fitness[:] = np.nan

    # Predicted probs
    objectives_fitness[:, 0] = predicted_probs[:, desired_class]

    # Sparsity
    ones_pct = ms.sum(axis=(1, 2)) / (ms.shape[1] * ms.shape[2])
    objectives_fitness[:, 1] = - ones_pct

    # Subsequences
    subsequences = np.count_nonzero(np.diff(ms, prepend=0, axis=1) == 1, axis=(1, 2))
    feature_avg_subsequences = subsequences / ms.shape[2]
    subsequences_pct = feature_avg_subsequences / (ms.shape[1] // 2)
    # objectives_fitness[:, 2] = subsequences_pct ** gamma
    objectives_fitness[:, 2] = -1 * (subsequences_pct**0.25)

    # Outlier scores
    increase_outlier_scores = outlier_scores - original_outlier_score
    increase_outlier_scores[increase_outlier_scores < 0] = 0
    objectives_fitness[:, 3] = - increase_outlier_scores

    # Apply penalization to all objectives, so it goes behind the pareto front
    predicted_classes = np.argmax(predicted_probs, axis=1)
    penalization_vector = (predicted_classes != desired_class).astype(int)
    penalization_matrix = np.repeat(penalization_vector.reshape(-1, 1), repeats=4, axis=1)
    objectives_fitness = objectives_fitness - invalid_penalization_scalar * penalization_matrix

    return objectives_fitness


def fitness_function_mo_os(ms, predicted_probs, desired_class, outlier_scores, original_outlier_score,
                           invalid_penalization_scalar):

    # Init objective fitness matrix with nans
    objectives_fitness = np.empty((ms.shape[0], 4))
    objectives_fitness[:] = np.nan

    # Predicted probs
    objectives_fitness[:, 0] = predicted_probs[:, desired_class]

    # Sparsity
    ones_pct = ms.sum(axis=(1, 2)) / (ms.shape[1] * ms.shape[2])
    objectives_fitness[:, 1] = - ones_pct

    # Subsequences
    subsequences = np.count_nonzero(np.diff(ms, prepend=0, axis=1) == 1, axis=(1, 2))
    feature_avg_subsequences = subsequences / ms.shape[2]
    subsequences_pct = feature_avg_subsequences / (ms.shape[1] // 2)
    # objectives_fitness[:, 2] = subsequences_pct ** gamma
    objectives_fitness[:, 2] = -1 * (subsequences_pct**0.25)

    # Outlier scores
    objectives_fitness[:, 3] = - outlier_scores

    # Apply penalization to all objectives, so it goes behind the pareto front
    predicted_classes = np.argmax(predicted_probs, axis=1)
    penalization_vector = (predicted_classes != desired_class).astype(int)
    penalization_matrix = np.repeat(penalization_vector.reshape(-1, 1), repeats=4, axis=1)
    objectives_fitness = objectives_fitness - invalid_penalization_scalar * penalization_matrix

    return objectives_fitness
