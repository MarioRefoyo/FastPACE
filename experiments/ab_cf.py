import os
import copy
import pickle
import json
import time
import sys
import random
from multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import torch


from experiments.experiment_utils import store_partial_cfs, load_parameters_from_json
from experiments.results.results_concatenator import concatenate_result_files
from methods.ABCF import ABCF
from experiments.experiment_utils import prepare_experiment, load_model


DATASETS = [
    # 'BasicMotions', 'NATOPS', 'UWaveGestureLibrary', 'Cricket',
    # 'ArticularyWordRecognition', 'Epilepsy',
    'PenDigits',
    # 'PEMS-SF', 'RacketSports', 'SelfRegulationSCP1'
]
DATASETS = [
    # 'CBF', 'CinCECGTorso', 'Coffee', "ECG200",
    # "ECG5000", 'FacesUCR', 'FordA', 'GunPoint', 'HandOutlines',
    'ItalyPowerDemand',
    # 'NonInvasiveFatalECGThorax2', 'Plane',
    # 'ProximalPhalanxOutlineCorrect',
    # 'Strawberry', 'TwoPatterns'
]

# PARAMS_PATH = 'experiments/params_cf/baseline_abcf.json'
# MODEL_TO_EXPLAIN_EXPERIMENT_NAME = 'cls_basic_train'
# MODEL_TO_EXPLAIN_EXPERIMENT_NAME = 'inceptiontime_noscaling'
# PARAMS_PATH = 'experiments/params_cf/baseline_abcf_torch.json'
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = 'inceptiontime_pytorch'
PARAMS_PATH = 'experiments/params_cf/baseline_abcf_longer_window_torch.json'
# MODEL_TO_EXPLAIN_EXPERIMENT_NAME = 'fcn_pytorch'

MULTIPROCESSING = False
I_START = 0
THREAD_SAMPLES = 100
POOL_SIZE = 1


def get_counterfactual_worker(sample_dict):
    dataset = sample_dict["dataset"]
    X_train, y_train = sample_dict["train_data_tuple"]
    exp_name = sample_dict["exp_name"]
    params = sample_dict["params"]
    first_sample_i = sample_dict["first_sample_i"]
    x_orig_samples_worker = sample_dict["x_orig_samples"]
    y_orig_samples_worker = sample_dict["y_orig_samples"]
    n_classes = sample_dict["n_classes"]
    ts_length = x_orig_samples_worker.shape[1]
    n_channels = x_orig_samples_worker.shape[2]

    # Set seed in thread. ToDo: is it really necessary?
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        tf.random.set_seed(params["seed"])
        torch.manual_seed(params["seed"])
        torch.cuda.manual_seed(params["seed"])
        random.seed(params["seed"])

    # Get model
    model_folder = f'experiments/models/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}'
    model_wrapper = load_model(model_folder, dataset, params["data_format"], n_channels, ts_length, n_classes)
    backend = model_wrapper.backend

    # Instantiate the Counterfactual Explanation method
    cf_explainer = ABCF(model_wrapper, X_train, y_train, window_pct=params["window_pct"], data_format=params["data_format"])

    # Generate counterfactuals
    results = []
    for i in tqdm(range(0, len(x_orig_samples_worker), 1)):
        x_orig = x_orig_samples_worker[i]
        y_orig = y_orig_samples_worker[i]
        result = cf_explainer.generate_counterfactual(x_orig, y_true_orig=y_orig)
        results.append(result)

    # Store results of cf in list
    store_partial_cfs(results, first_sample_i, first_sample_i+THREAD_SAMPLES-1,
                      dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, file_suffix_name=exp_name)
    return 1


def experiment_dataset(dataset, exp_name, params):
    X_train, y_train, X_test, y_test, subset_idx, model_wrapper, y_pred_train, y_pred_test, ts_length, n_channels, n_classes = prepare_experiment(
        dataset, params, MODEL_TO_EXPLAIN_EXPERIMENT_NAME)

    # Get counterfactuals
    if MULTIPROCESSING:
        # Prepare dict to iterate optimization problem
        samples = []
        for i in range(I_START, len(X_test), THREAD_SAMPLES):
            # Init optimizer
            x_orig_samples = X_test[i:i + THREAD_SAMPLES]
            y_orig_samples = y_test[i:i + THREAD_SAMPLES]

            sample_dict = {
                "dataset": dataset,
                "train_data_tuple": (X_train, y_train),
                "exp_name": exp_name,
                "params": params,
                "first_sample_i": i,
                "x_orig_samples": x_orig_samples,
                "y_orig_samples": y_orig_samples,
                "n_classes": n_classes
            }
            samples.append(sample_dict)

        # Execute counterfactual generation
        print('Starting counterfactual generation using multiprocessing...')
        with Pool(POOL_SIZE) as p:
            _ = list(tqdm(p.imap(get_counterfactual_worker, samples), total=len(samples)))

        # Concatenate the results
        concatenate_result_files(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, exp_name)

    else:
        cf_explainer = ABCF(model_wrapper, X_train, y_train, window_pct=params["window_pct"], data_format=params["data_format"])

        # Generate counterfactuals
        results = []
        for i in tqdm(range(len(X_test))):
            x_orig = X_test[i]
            y_orig = y_test[i]
            result = cf_explainer.generate_counterfactual(x_orig, y_true_orig=y_orig)
            results.append(result)

        # Store experiment results
        with open(f'./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}/counterfactuals.pickle', 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    # Store experiment metadata
    params["X_test_indexes"] = subset_idx.tolist()
    with open(f'./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}/params.json', 'w') as fp:
        json.dump(params, fp, sort_keys=True)


if __name__ == "__main__":
    # Load parameters
    exp_name = "abcf"
    all_params = load_parameters_from_json(PARAMS_PATH)
    for dataset in DATASETS:
        if not os.path.isdir(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}"):
            os.makedirs(f"./experiments/results/{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{exp_name}")
        print(f'Starting experiment for dataset {dataset}...')
        experiment_dataset(
            dataset,
            exp_name,
            all_params
        )
    print('Finished')
