import copy
import os
import json
import pickle
import random
import shutil
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, silhouette_samples
import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from matplotlib import pyplot as plt

from experiments.experiment_utils import (local_data_loader, ucr_data_loader, label_encoder,
                                          load_parameters_from_json, generate_settings_combinations)
from experiments.models.utils import AEModelConstructorV1

from methods.outlier_calculators import LOFOutlierCalculator


"""DATASETS = [
    "BasicMotions", "NATOPS", "UWaveGestureLibrary",
    'ArticularyWordRecognition', 'Cricket', 'Epilepsy', 'PenDigits',
    'PEMS-SF', 'RacketSports', 'SelfRegulationSCP1'
]"""

DATASETS = [
    'CBF', 'CinCECGTorso', 'Coffee', "ECG200",
    "ECG5000", 'FacesUCR', 'FordA', 'GunPoint', 'HandOutlines',
    'ItalyPowerDemand',
    'NonInvasiveFatalECGThorax2', 'Plane',
    'ProximalPhalanxOutlineCorrect',
    'Strawberry', 'TwoPatterns'
]

PARAMS_PATH = 'experiments/params_model_training/lof_basic_train_scaling.json'


def get_permutation_outlier_scores(lof_model, X_train, X_test, y_train, y_test):
    # Create test for AE reconstruction errors
    # For test set
    X_test_flat = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
    X_test_preds = lof_model.predict(X_test_flat)
    s_samples_base = silhouette_samples(X_test_flat, X_test_preds)

    # For same true class channel permutations
    classes = np.unique(y_train)
    class_indexes = {c: np.where(y_test == c)[0] for c in classes}
    X_perm_same = []
    for i, instance in enumerate(X_test):
        c = y_test[i]
        idx_of_interest = class_indexes[c]
        # Get random sample to use
        perm_idx = random.choice(idx_of_interest)
        # Create permutation mask
        perm_channels = np.random.randint(2, size=X_test.shape[2]).astype(bool)
        perm_mask = np.tile(perm_channels, (X_test.shape[1], 1)).astype(bool)
        # Permute sample
        new_instance = copy.deepcopy(instance)
        new_instance[perm_mask] = X_test[perm_idx][perm_mask]
        X_perm_same.append(new_instance)
    X_perm_same = np.array(X_perm_same)
    X_perm_same_flat = X_perm_same.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    X_perm_same_preds = lof_model.predict(X_perm_same_flat)
    s_samples_perm_same = silhouette_samples(X_perm_same_flat, X_perm_same_preds)

    # For different true class channel permutations
    classes = np.unique(y_train)
    class_indexes = {c: np.where(y_test != c)[0] for c in classes}
    X_perm_diff = []
    for i, instance in enumerate(X_test):
        c = y_test[i]
        idx_of_interest = class_indexes[c]
        # Get random sample to use
        perm_idx = random.choice(idx_of_interest)
        # Create permutation mask
        perm_channels = np.random.randint(2, size=X_test.shape[2]).astype(bool)
        perm_mask = np.tile(perm_channels, (X_test.shape[1], 1)).astype(bool)
        # Permute sample
        new_instance = copy.deepcopy(instance)
        new_instance[perm_mask] = X_test[perm_idx][perm_mask]
        X_perm_diff.append(new_instance)
    X_perm_diff = np.array(X_perm_diff)
    X_perm_diff_flat = X_perm_diff.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    X_perm_diff_preds = lof_model.predict(X_perm_diff_flat)
    s_samples_perm_diff = silhouette_samples(X_perm_diff_flat, X_perm_diff_preds)

    # Compare
    silhouette_samples_df = pd.DataFrame()
    silhouette_samples_df['base'] = s_samples_base
    silhouette_samples_df['perm_same'] = s_samples_perm_same
    silhouette_samples_df['perm_diff'] = s_samples_perm_diff
    return silhouette_samples_df


def train_lof_experiment(dataset, exp_name, exp_hash, params):
    # Set seed
    np.random.seed(params["seed"])
    tf.random.set_seed(params["seed"])
    random.seed(params["seed"])

    scaling = params["scaling"]
    if os.path.isdir(f"./experiments/data/UCR/{dataset}"):
        X_train, y_train, X_test, y_test, ts_length, n_channels = local_data_loader(str(dataset), scaling, backend="tf",
                                                                                    data_path="./experiments/data")
    else:
        os.makedirs(f"./experiments/data/UCR/{dataset}")
        X_train, y_train, X_test, y_test = ucr_data_loader(dataset, scaling, backend="tf",
                                                           store_path="./experiments/data/UCR")
        if X_train is None:
            raise ValueError(f"Dataset {dataset} could not be downloaded")
    ts_length, n_channels = X_train.shape[1], X_train.shape[2]
    X_train_flat = X_train.reshape(X_train.shape[0], ts_length * n_channels)
    X_test_flat = X_test.reshape(X_test.shape[0], ts_length * n_channels)

    # Define model architecture
    model_params = {
        "n_neighbors": params["n_neighbors"],
        "contamination": params["contamination"],
        "p": params["p"],
        "n_jobs": params["n_jobs"],
        "novelty": True,
    }
    lof_model = LocalOutlierFactor(**model_params)

    # Create model folder if it does not exist
    results_path = f"./experiments/models/{dataset}/{exp_name}/{exp_hash}"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Model fit
    lof_model.fit(X_train_flat)

    # Expert result metrics
    X_train_pred = lof_model.predict(X_train_flat)
    train_ss = silhouette_score(X_train_flat, X_train_pred)
    X_test_pred = lof_model.predict(X_test_flat)
    test_ss = silhouette_score(X_test_flat, X_test_pred)
    print(f"Train Silhouette Score: {train_ss:.2f} --- Test Silhouette Score: {test_ss:.2f}")
    silhouette_samples_df = get_permutation_outlier_scores(lof_model, X_train, X_test, y_train, y_test)
    plt.figure()
    silhouette_samples_df.hist(layout=(3, 1), sharex=True, sharey=True)
    plt.savefig(f"{results_path}/reconstruction_test.png")
    result_metrics = {
        'test_ss_base': silhouette_samples_df['base'].median(),
        'test_ss_perm_same': silhouette_samples_df['perm_same'].median() - silhouette_samples_df['base'].median(),
        'test_ss_perm_diff': silhouette_samples_df['perm_diff'].median() - silhouette_samples_df['base'].median(),
        'train_ss': train_ss,
        'test_ss': test_ss
    }
    with open(f"{results_path}/metrics.json", "w") as outfile:
        json.dump(result_metrics, outfile)

    # Export training params
    params = {**{'experiment_hash': exp_hash}, **params}
    with open(f"{results_path}/train_params.json", "w") as outfile:
        json.dump(params, outfile)

    # Store model
    with open(f'{results_path}/model.pickle', 'wb') as f:
        pickle.dump(lof_model, f, pickle.HIGHEST_PROTOCOL)

    # Create Outlier calculator and store it
    outlier_calculator = LOFOutlierCalculator(lof_model, X_train, data_format="tf")
    with open(f'{results_path}/outlier_calculator.pickle', 'wb') as f:
        pickle.dump(outlier_calculator, f, pickle.HIGHEST_PROTOCOL)


def select_best_model(dataset, exp_name):
    experiment_folder = f"./experiments/models/{dataset}/{exp_name}"
    # Locate all experiment hashes for the given dataset by inspecting the folders
    experiment_sub_dirs = [f for f in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, f))]
    # Iterate through the combinations and retrieve the results file
    experiment_info_list = []
    for experiment_sub_dir in experiment_sub_dirs:
        results_path = f'{experiment_folder}/{experiment_sub_dir}'
        try:
            # Read the params file
            with open(f"{results_path}/train_params.json") as f:
                train_params = json.load(f)
            # Read the metrics file
            with open(f"{results_path}/metrics.json") as f:
                metrics = json.load(f)
        except Exception:
            continue
        # Merge all info
        experiment_info = {**train_params, **metrics}
        experiment_info_list.append(experiment_info)

    # Create the a dataframe containing all info and store it
    experiment_results_df = pd.DataFrame.from_records(experiment_info_list).sort_values("test_ss", ascending=False)
    best_experiment_hash = experiment_results_df.iloc[0]['experiment_hash']
    experiment_results_df.to_excel(f"{experiment_folder}/all_combination_results.xlsx")

    # Move the model best model to the experiment folder
    shutil.copyfile(
        f"{experiment_folder}/{best_experiment_hash}/model.pickle",
        f"{experiment_folder}/model.pickle"
    )
    shutil.copyfile(
        f"{experiment_folder}/{best_experiment_hash}/outlier_calculator.pickle",
        f"{experiment_folder}/outlier_calculator.pickle"
    )
    shutil.copyfile(
        f"{experiment_folder}/{best_experiment_hash}/train_params.json",
        f"{experiment_folder}/train_params.json"
    )


if __name__ == "__main__":
    # Load parameters
    all_params = load_parameters_from_json(PARAMS_PATH)
    experiment_name = all_params['experiment_name']
    params_combinations = generate_settings_combinations(all_params)
    for dataset in DATASETS:
        # Train all combinations
        for experiment_hash, experiment_params in params_combinations.items():
            print(f'Starting experiment {experiment_hash} for dataset {dataset}...')
            try:
                train_lof_experiment(
                    dataset,
                    experiment_name,
                    experiment_hash,
                    experiment_params
                )
            except (ValueError, FileNotFoundError, TypeError, ResourceExhaustedError) as msg:
                print(msg)

        # Compare performance of combinations and select the best one
        if os.path.isdir(f"./experiments/models/{dataset}/{experiment_name}"):
            select_best_model(dataset, experiment_name)

    print('Finished')

