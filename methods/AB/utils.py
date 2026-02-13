# Code from https://sites.google.com/view/attention-based-cf
import numpy as np

import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeries


def sliding_window_3d(data, window_size, stride):
    """
    Extracts 3D subsequences from a multivariate time series dataset using a sliding window approach.

    Arguments:
    - data: numpy array of shape (num_timesteps, num_features) representing the time series data.
    - window_size: int representing the length of the sliding window.
    - stride: int representing the stride of the sliding window.

    Returns:
    - numpy array of shape (num_subsequences, num_features, window_size) representing the 3D subsequences.
    """
    num_features, num_timesteps = data.shape
    num_subsequences = ((num_timesteps - window_size) // stride) + 1
    subsequences = np.zeros((num_subsequences, num_features, window_size))
    for j in range(num_subsequences):
        start = j * stride
        end = start + window_size
        subsequences[j] = data[:, start:end]

    return subsequences


def entropy(predict_proba):
    """
    Calculates the entropy of a set of predicted probabilities.

    Args:
        predict_proba (array-like): A 1D array of predicted probabilities.

    Returns:
        float: The entropy of the predicted probabilities.
    """
    predict_proba = predict_proba[np.nonzero(predict_proba)]
    return -np.sum(predict_proba * np.log2(predict_proba))


def native_guide_retrieval(query, target_label, distance, n_neighbors, X_train, y_train):
    """
    Discover the nearest neighbor from the target class (or we can say NUN from the predict class)

    Args:
        query: the query instance we want to generate cf from
        target_label: the target class of cf
        distance: which disatnce metric we want to use while retriving the NUN
        n_neighbors: the number of neighbors we consider while using K nearest neighbor algorithm
    Returns:
        index: the index of the NUN in the training set
    """
    df = pd.DataFrame(y_train, columns=['label'])
    df.index.name = 'index'
    ts_length = X_train.shape[1] * X_train.shape[2]

    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
    knn.fit(X_train[list(df[df['label'] == target_label].index.values)])

    _, ind = knn.kneighbors(query.reshape(1, query.shape[0], query.shape[1]), return_distance=True)
    return df[df['label'] == target_label].index[ind[0][:]]


# Original implementation
def target_(model, instance):
    target = np.argsort((model.predict_proba(instance.reshape(1,instance.shape[0],instance.shape[1]))))[0][-2:-1][0]
    return target


def target_adapted(model_wrapper, instance):
    preds = model_wrapper.predict(instance)
    target = np.argsort(preds)[0][-2:-1][0]
    return target
