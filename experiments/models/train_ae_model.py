import copy
import os
import json
import pickle
import random
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from matplotlib import pyplot as plt

from experiments.experiment_utils import (local_data_loader, ucr_data_loader, label_encoder,
                                          load_parameters_from_json, generate_settings_combinations)
from experiments.models.utils import AEModelConstructorV1

from methods.outlier_calculators import AEOutlierCalculator


"""DATASETS = ["NATOPS"]
TRAIN_PARAMS = {
    'seed': 42,
    'encoder_filters_kernels': [(64, 5), (32, 5), (16, 5)],
    'temporal_strides': 2,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""

"""DATASETS = ["UWaveGestureLibrary"]
TRAIN_PARAMS = {
    'seed': 42,
    'encoder_filters_kernels': [(32, 7), (16, 7), (8, 5), (4, 5)],
    'temporal_strides': 2,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""

"""DATASETS = ["BasicMotions"]
TRAIN_PARAMS = {
    'seed': 42,
    'encoder_filters_kernels': [(32, 5), (4, 5)],
    'temporal_strides': 2,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""

DATASETS = [
    # 'BasicMotions', 'NATOPS', 'UWaveGestureLibrary',
    'ArticularyWordRecognition', 'AtrialFibrillation', 'CharacterTrajectories', 'Cricket',
    'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing',  'FaceDetection',
    'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels',
    'Libras', 'LSST', 'MotorImagery', 'PenDigits', 'PEMS-SF', 'Phoneme', 'RacketSports',
    'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump'
]
DATASETS = [
    # 'BeetleFly', # Doubt (0.79) and bad reconstructions
    # 'ChlorineConcentration', 'CinCECGTorso',
    'DistalPhalanxOutlineCorrect',
    'ECG5000',
    'ECGFiveDays', # Doubt bad reconstructions
    'FaceAll', # Review reconstructions
    'FaceFour', # BAD RECONSTRUCTIONS
    'FacesUCR', 'Fish', 'FordA',
    'HandOutlines', 'ItalyPowerDemand',
    'MiddlePhalanxOutlineCorrect',
    'MoteStrain', # BAD RECONSTRUCTIONS
    'NonInvasiveFatalECGThorax1', 'NonInvasiveFatalECGThorax2',
    'PhalangesOutlinesCorrect', 'Plane',
    'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect',
    'SonyAIBORobotSurface2', # BAD RECONSTRUCTIONS
    'StarLightCurves', 'Strawberry', 'SwedishLeaf',
    'SyntheticControl',
    'Trace', 'TwoPatterns',
    'ToeSegmentation2', # Doubt (0.79)
    'Wafer',
    'Yoga', # Doubt (0.79)
]
DATASETS = [
    'CBF', 'CinCECGTorso', 'Coffee', "ECG200", "ECG5000", 'FacesUCR', 'FordA', 'GunPoint', 'HandOutlines',
    'ItalyPowerDemand', 'NonInvasiveFatalECGThorax2', 'Plane', 'ProximalPhalanxOutlineCorrect',
    'Strawberry', 'TwoPatterns'
]
DATASETS = [
    'BasicMotions', 'NATOPS', 'UWaveGestureLibrary', 'Cricket',
    'ArticularyWordRecognition', 'Epilepsy',
    'PenDigits',
    'PEMS-SF',
    'RacketSports', 'SelfRegulationSCP1'
]
PARAMS_PATH = 'experiments/params_model_training/ae_basic_train_scaling.json'


def get_permutation_reconstruction_scores(ae, X_train, X_test, y_train, y_test):
    # Create test for AE reconstruction errors
    # For test set
    X_test_recons = ae.predict(X_test)
    recons_errors_base = np.mean(np.abs(X_test - X_test_recons), axis=(1, 2))

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
        perm_channels = np.random.randint(2, size=X_train.shape[2]).astype(bool)
        perm_mask = np.tile(perm_channels, (X_train.shape[1], 1)).astype(bool)
        # Permute sample
        new_instance = copy.deepcopy(instance)
        new_instance[perm_mask] = X_test[perm_idx][perm_mask]
        X_perm_same.append(new_instance)
    X_perm_same = np.array(X_perm_same)
    X_perm_same_recons = ae.predict(X_perm_same)
    recons_errors_perm_same = np.mean(np.abs(X_perm_same - X_perm_same_recons), axis=(1, 2))

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
        perm_channels = np.random.randint(2, size=X_train.shape[2]).astype(bool)
        perm_mask = np.tile(perm_channels, (X_train.shape[1], 1)).astype(bool)
        # Permute sample
        new_instance = copy.deepcopy(instance)
        new_instance[perm_mask] = X_test[perm_idx][perm_mask]
        X_perm_diff.append(new_instance)
    X_perm_diff = np.array(X_perm_diff)
    X_perm_diff_recons = ae.predict(X_perm_diff)
    recons_errors_perm_diff = np.mean(np.abs(X_perm_diff - X_perm_diff_recons), axis=(1, 2))

    # Compare
    recons_errors_df = pd.DataFrame()
    recons_errors_df['base'] = recons_errors_base
    recons_errors_df['perm_same'] = recons_errors_perm_same
    recons_errors_df['perm_diff'] = recons_errors_perm_diff
    return recons_errors_df


def train_ae_experiment(dataset, exp_name, exp_hash, params):
    # Set seed
    np.random.seed(params["seed"])
    tf.random.set_seed(params["seed"])
    random.seed(params["seed"])

    # Load data
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
    y_train, y_test = label_encoder(y_train, y_test)

    # Define model architecture
    input_shape = X_train.shape[1:]
    ae = AEModelConstructorV1(input_shape, params['temporal_strides'], params['compression_rate']).get_model(params['model_type'])
    ae.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    print(ae.summary())
    output_shape = ae.predict(X_train).shape
    if output_shape[1:] != input_shape:
        raise ValueError("Model is too deep for the given stride and number of layers")

    # Create model folder if it does not exist
    results_path = f"./experiments/models/{dataset}/{exp_name}/{exp_hash}"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Model fit
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["es_patience"], verbose=1)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=params["lrs_patience"], verbose=1)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(f'{results_path}/model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    training_history = ae.fit(
        X_train, X_train,
        batch_size=params['batch_size'],
        shuffle=True,
        validation_split=0.1,
        epochs=params["epochs"],
        callbacks=[early_stopping, lr_scheduler, mcp_save],
    )

    # Expert result metrics
    X_train_reconst = ae.predict(X_train)
    train_mae = np.mean(np.abs(X_train - X_train_reconst))
    X_test_reconst = ae.predict(X_test)
    test_mae = np.mean(np.abs(X_test - X_test_reconst))
    print(f"Train MAE: {train_mae:.2f} --- Test MAE: {test_mae:.2f}")
    recons_errors_df = get_permutation_reconstruction_scores(ae, X_train, X_test, y_train, y_test)
    plt.figure()
    recons_errors_df.hist(layout=(3, 1), sharex=True, sharey=True)
    plt.savefig(f"{results_path}/reconstruction_test.png")
    result_metrics = {
        'test_recons_base': recons_errors_df['base'].median(),
        '+test_recons_perm_same': recons_errors_df['perm_same'].median() - recons_errors_df['base'].median(),
        '+test_recons_perm_diff': recons_errors_df['perm_diff'].median() - recons_errors_df['base'].median(),
        'train_mae': train_mae,
        'test_mae': test_mae
    }
    with open(f"{results_path}/metrics.json", "w") as outfile:
        json.dump(result_metrics, outfile)

    # Export training params
    params = {**{'experiment_hash': exp_hash}, **params}
    with open(f"{results_path}/train_params.json", "w") as outfile:
        json.dump(params, outfile)

    # Export sample reconstructions
    n_features = X_test.shape[2]
    for i in np.random.choice(len(X_test), 10):
        fig = plt.figure(figsize=(8, 6))
        f, axs = plt.subplots(n_features, 1)
        if n_features==1:
            axs = [axs]
        for j in range(n_features):
            axs[j].plot(list(range(X_test.shape[1])), X_test[i, :, j].flatten())
            axs[j].plot(list(range(X_test.shape[1])), X_test_reconst[i, :, j].flatten())
        plt.title(f"Instance: {i}, Label: {y_test[i]}")
        plt.savefig(f"{results_path}/reconstruction_{i}.png")

    # Create Outlier calculator and store it
    outlier_calculator = AEOutlierCalculator(ae, X_train)
    with open(f'{results_path}/outlier_calculator.pickle', 'wb') as f:
        pickle.dump(outlier_calculator, f, pickle.HIGHEST_PROTOCOL)

    # Export loss training loss evolution
    fig, (ax1) = plt.subplots(1, 1)
    pd.DataFrame(training_history.history)[['loss', 'val_loss']].plot(ax=ax1)
    plt.savefig(f"{results_path}/loss_curve.png")


def select_best_model(dataset, exp_name):
    experiment_folder = f"./experiments/models/{dataset}/{exp_name}"
    # Locate all experiment hashes for the given dataset by inspecting the folders
    experiment_sub_dirs = [f for f in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, f))]
    # Iterate through the combinations and retrieve the results file
    experiment_info_list = []
    for experiment_sub_dir in experiment_sub_dirs:
        results_path = f'{experiment_folder}/{experiment_sub_dir}'
        # Read the params file
        with open(f"{results_path}/train_params.json") as f:
            train_params = json.load(f)
        # Read the metrics file
        with open(f"{results_path}/metrics.json") as f:
            metrics = json.load(f)
        # Merge all info
        experiment_info = {**train_params, **metrics}
        experiment_info_list.append(experiment_info)

    # Create the a dataframe containing all info and store it
    experiment_results_df = pd.DataFrame.from_records(experiment_info_list).sort_values("test_mae", ascending=True)
    best_experiment_hash = experiment_results_df.iloc[0]['experiment_hash']
    experiment_results_df.to_excel(f"{experiment_folder}/all_combination_results.xlsx")

    # Move the model best model to the experiment folder
    shutil.copyfile(
        f"{experiment_folder}/{best_experiment_hash}/model.hdf5",
        f"{experiment_folder}/model.hdf5"
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
                train_ae_experiment(
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

