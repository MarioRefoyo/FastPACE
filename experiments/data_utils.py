import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tslearn.datasets import UCR_UEA_datasets


def min_max_scale_data(X_train, X_test):
    max = 1
    min = 0
    """maximums = X_train.max(axis=(0, 1))
    minimums = X_train.min(axis=(0, 1))"""
    data_max = X_train.max()
    data_min = X_train.min()

    # Min Max scale data between 0 and 1
    X_train_scaled = (X_train - data_min) / (data_max - data_min)
    X_train_scaled = X_train_scaled * (max - min) + min

    X_test_scaled = (X_test - data_min) / (data_max - data_min)
    X_test_scaled = X_test_scaled * (max - min) + min

    return X_train_scaled, X_test_scaled


def standard_scale_data(X_train, X_test):
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train, X_test


def ucr_data_loader(dataset, scaling, backend="torch", store_path="../../data/UCR"):
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    if X_train is not None:
        np.save(f"{store_path}/{dataset}/X_train.npy", X_train)
        np.save(f"{store_path}/{dataset}/X_test.npy", X_test)
        np.save(f"{store_path}/{dataset}/y_train.npy", y_train)
        np.save(f"{store_path}/{dataset}/y_test.npy", y_test)

        # Scaling
        if scaling == "min_max":
            X_train, X_test = min_max_scale_data(X_train, X_test)
        elif scaling == "standard":
            X_train, X_test = standard_scale_data(X_train, X_test)
        elif scaling == "none":
            pass
        else:
            raise ValueError("Not valid scaling value")

        # Backend
        if backend == "torch":
            X_train = X_train.transpose(0, 2, 1)
            X_test = X_test.transpose(0, 2, 1)
        elif backend == "tf":
            pass
        else:
            raise ValueError("backend not valid. Choose torch or tf")
    return X_train, y_train, X_test, y_test


def local_data_loader(dataset, scaling, backend="torch", data_path="../../data"):
    X_train = np.load(f'{data_path}/UCR/{dataset}/X_train.npy', allow_pickle=True)
    X_test = np.load(f'{data_path}/UCR/{dataset}/X_test.npy', allow_pickle=True)
    y_train = np.load(f'{data_path}/UCR/{dataset}/y_train.npy', allow_pickle=True)
    y_test = np.load(f'{data_path}/UCR/{dataset}/y_test.npy', allow_pickle=True)

    # Scaling
    if scaling == "min_max":
        X_train, X_test = min_max_scale_data(X_train, X_test)
    elif scaling == "standard":
        X_train, X_test = standard_scale_data(X_train, X_test)
    elif scaling == "none":
        pass
    else:
        raise ValueError("Not valid scaling value")

    # Backend
    if backend == "torch":
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)
        ts_length = X_train.shape[2]
        n_channels = X_train.shape[1]
    elif backend == "tf":
        ts_length = X_train.shape[1]
        n_channels = X_train.shape[2]
    else:
        raise ValueError("backend not valid. Choose torch or tf")

    return X_train, y_train, X_test, y_test, ts_length, n_channels


def label_encoder(training_labels, testing_labels):
    # If label represent integers, try to cast it. If it is not possible, then resort to Label Encoding
    try:
        y_train = []
        for label in training_labels:
            y_train.append(int(float(label)))
        y_test = []
        for label in testing_labels:
            y_test.append(int(float(label)))

        # Check if labels are consecutive
        if sorted(y_train) == list(range(min(y_train), max(y_train) + 1)):
            # Add class 0 in case it does not exist
            y_train, y_test = np.array(y_train).reshape(-1, 1), np.array(y_test).reshape(-1, 1)
            classes = np.unique(y_train)
            if 0 not in classes:
                y_train = y_train - 1
                y_test = y_test - 1
        else:
            # Raise exception so each class is treated as a category
            raise ValueError("The classes can be casted to integers but they are non consecutive numbers. Treating them as categories")

    except Exception:
        le = LabelEncoder()
        le.fit(np.concatenate((training_labels, testing_labels), axis=0))
        y_train = le.transform(training_labels)
        y_test = le.transform(testing_labels)
    return y_train, y_test
