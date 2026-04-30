import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tslearn.datasets import UCR_UEA_datasets

REQUIRED_DATASET_FILES = ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy")


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


def _dataset_folder(dataset, data_path):
    return os.path.join(data_path, "UCR", str(dataset))


def _is_invalid_scalar_none_array(arr):
    if not isinstance(arr, np.ndarray):
        return True
    if arr.shape == () and arr.dtype == object:
        try:
            return arr.item() is None
        except Exception:
            return True
    return False


def _validate_arrays(X_train, y_train, X_test, y_test):
    arrays = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test
    }
    for arr_name, arr in arrays.items():
        if arr is None:
            return False, f"{arr_name} is None"
        if _is_invalid_scalar_none_array(arr):
            return False, f"{arr_name} is a scalar object placeholder (corrupted cache)"
        if not isinstance(arr, np.ndarray):
            return False, f"{arr_name} is not a numpy array"
        if arr.size == 0:
            return False, f"{arr_name} is empty"

    if X_train.ndim != 3 or X_test.ndim != 3:
        return False, "X_train/X_test must be 3D arrays"

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        return False, "Mismatch between series and labels lengths"

    return True, ""


def _load_local_arrays(dataset, data_path):
    dataset_dir = _dataset_folder(dataset, data_path)
    arrays = {}
    for file_name in REQUIRED_DATASET_FILES:
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        arrays[file_name] = np.load(file_path, allow_pickle=True)
    X_train = arrays["X_train.npy"]
    X_test = arrays["X_test.npy"]
    y_train = arrays["y_train.npy"]
    y_test = arrays["y_test.npy"]
    is_valid, msg = _validate_arrays(X_train, y_train, X_test, y_test)
    if not is_valid:
        raise ValueError(f"Invalid local dataset cache for '{dataset}': {msg}")
    return X_train, y_train, X_test, y_test


def dataset_cache_is_valid(dataset, data_path="../../data"):
    try:
        _load_local_arrays(dataset, data_path)
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


def _tslearn_cache_root_from_store_path(store_path):
    # Keep tslearn cache in-project so downloads are not blocked by user-profile permissions.
    base_dir = os.path.dirname(os.path.abspath(store_path.rstrip("/\\")))
    return os.path.join(base_dir, "_tslearn_cache")


@contextmanager
def _temporary_tslearn_cache_env(cache_root):
    fake_home = os.path.join(cache_root, "home")
    xdg_cache_home = os.path.join(cache_root, "xdg_cache")
    os.makedirs(fake_home, exist_ok=True)
    os.makedirs(xdg_cache_home, exist_ok=True)
    os.makedirs(os.path.join(fake_home, ".cache"), exist_ok=True)
    os.makedirs(os.path.join(fake_home, ".tslearn"), exist_ok=True)

    env_updates = {
        "HOME": fake_home,
        "USERPROFILE": fake_home,
        "XDG_CACHE_HOME": xdg_cache_home,
        # Best-effort variables for versions that may honor one of these.
        "TSLEARN_HOME": os.path.join(fake_home, ".tslearn"),
        "TSLEARN_CACHE_DIR": os.path.join(fake_home, ".tslearn"),
    }

    previous_env = {k: os.environ.get(k) for k in env_updates}
    try:
        for key, value in env_updates.items():
            os.environ[key] = value
        yield env_updates
    finally:
        for key, old_value in previous_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def ucr_data_loader(dataset, scaling, backend="torch", store_path="../../data/UCR"):
    dataset = str(dataset)
    cache_root = _tslearn_cache_root_from_store_path(store_path)
    os.makedirs(cache_root, exist_ok=True)

    with _temporary_tslearn_cache_env(cache_root) as env_info:
        ucr_uea_class = UCR_UEA_datasets(use_cache=True)
        aliases = [dataset]

        try:
            available_datasets = ucr_uea_class.list_datasets()
            lower_to_canonical = {d.lower(): d for d in available_datasets}
            canonical_name = lower_to_canonical.get(dataset.lower())
            if canonical_name is not None and canonical_name not in aliases:
                aliases.append(canonical_name)
            manual_aliases = {
                "SpokenArabicDigits": "ArabicDigits",
                "ArabicDigits": "SpokenArabicDigits",
                "ERing": "Ering",
                "Ering": "ERing",
                "GunPoint": "Gunpoint",
                "Gunpoint": "GunPoint",
            }
            alias_candidate = manual_aliases.get(dataset)
            if alias_candidate is not None and alias_candidate not in aliases:
                aliases.append(alias_candidate)
        except Exception:
            # If listing fails, keep trying with the requested name.
            pass

        last_error = None
        for dataset_name in aliases:
            try:
                X_train, y_train, X_test, y_test = ucr_uea_class.load_dataset(dataset_name)
            except PermissionError as exc:
                raise ValueError(
                    f"Permission error while downloading '{dataset_name}'. "
                    f"tslearn cache redirected to: {env_info['TSLEARN_HOME']}. "
                    "If this still fails, check antivirus/file-locking on this folder."
                ) from exc
            except OSError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue

            is_valid, msg = _validate_arrays(X_train, y_train, X_test, y_test)
            if not is_valid:
                last_error = ValueError(f"Invalid downloaded dataset '{dataset_name}': {msg}")
                continue

            dataset_dir = os.path.join(store_path, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(os.path.join(dataset_dir, "X_train.npy"), X_train)
            np.save(os.path.join(dataset_dir, "X_test.npy"), X_test)
            np.save(os.path.join(dataset_dir, "y_train.npy"), y_train)
            np.save(os.path.join(dataset_dir, "y_test.npy"), y_test)

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

        if last_error is not None:
            print(f"[WARN] Could not load dataset '{dataset}' from tslearn: {last_error}")
        return None, None, None, None


def local_data_loader(dataset, scaling, backend="torch", data_path="../../data"):
    X_train, y_train, X_test, y_test = _load_local_arrays(dataset, data_path)

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
