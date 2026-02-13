import os
import json
import random
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, f1_score, accuracy_score

from experiments.experiment_utils import (local_data_loader, ucr_data_loader, label_encoder,
                                          load_parameters_from_json, generate_settings_combinations)
from experiments.models.utils import InceptionModelConstructorV1

"""DATASETS = ["NATOPS", "UWaveGestureLibrary"]
TRAIN_PARAMS = {
    'seed': 42,
    'conv_filters_kernels': [(32, 3), (64, 3), (128, 3)],
    'dense_units': [128],
    'dropout': 0.25,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""

"""DATASETS = ["BasicMotions"]
TRAIN_PARAMS = {
    'seed': 42,
    'conv_filters_kernels': [(32, 3), (64, 3)],
    'dense_units': [64],
    'dropout': 0.,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""

"""DATASETS = ["Epilepsy"]
TRAIN_PARAMS = {
    'seed': 42,
    'conv_filters_kernels': [(16, 7), (32, 7), (64, 3)],
    'dense_units': [64],
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 200,
    'es_patience': 30,
    'lrs_patience': 10,
}"""


# DATASETS = ['CBF', 'ECG200', 'Gunpoint', 'Chinatown', 'Coffee']
# Multivariate Datasets
DATASETS = [
    'BasicMotions', 'NATOPS', 'UWaveGestureLibrary',
    'ArticularyWordRecognition',
    'AtrialFibrillation', 'CharacterTrajectories', 'Cricket',
    'DuckDuckGeese', 'EigenWorms', 'Epilepsy',
    'EthanolConcentration', 'ERing',
    'FaceDetection',
    'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
    'InsectWingbeat', 'JapaneseVowels', 'Libras',
    'LSST', 'MotorImagery', 'PenDigits', 'PEMS-SF',
    'Phoneme', 'RacketSports', 'SelfRegulationSCP1',
    'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump'
]
"""DATASETS = [
    'BasicMotions', 'NATOPS', 'UWaveGestureLibrary',
    'ArticularyWordRecognition', 'CharacterTrajectories', 'Cricket',
    'Epilepsy', 'ERing',
    'Heartbeat', 'JapaneseVowels',
    'Libras', 'PenDigits',
    'PEMS-SF', 'RacketSports',
    'SelfRegulationSCP1', 'SpokenArabicDigits'
]"""
DATASETS = [
    'AtrialFibrillation',
    'DuckDuckGeese', 'EigenWorms', 'EthanolConcentration', 'FaceDetection',
    'FingerMovements', 'HandMovementDirection', 'Handwriting', 'InsectWingbeat',
    'LSST', 'MotorImagery', 'Phoneme',
    'SelfRegulationSCP2', 'StandWalkJump'
]
# Univariate Datasets
"""DATASETS = [
    'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
    'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers',
    'DiatomSizeReduction',
    'DistalPhalanxOutlineCorrect', 'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays',
    'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish',
    'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines', 'Haptics', 'Herring',
    'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand',
    'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat',
    'MedicalImages', 'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxTW', 'MoteStrain',
    'NonInvasiveFatalECGThorax1', 'NonInvasiveFatalECGThorax2', 'OliveOil',
    'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane',
    'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim',
    'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf',
    'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2',
    'Trace', 'TwoLeadECG', 'TwoPatterns',
    'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
]
DATASETS = ['Coffee', 'CinCECGTorso']"""

PARAMS_PATH = 'experiments/params_model_training/inceptiontime_noscaling.json'


def train_experiment(dataset, exp_name, exp_hash, params):
    # Set seed
    np.random.seed(params["seed"])
    tf.random.set_seed(params["seed"])
    random.seed(params["seed"])

    # Load data
    scaling = params["scaling"]
    if os.path.isdir(f"./experiments/data/UCR/{dataset}"):
        X_train, y_train, X_test, y_test = local_data_loader(str(dataset), scaling, backend="tf", data_path="./experiments/data")
    else:
        os.makedirs(f"./experiments/data/UCR/{dataset}")
        X_train, y_train, X_test, y_test = ucr_data_loader(dataset, scaling, backend="tf", store_path="./experiments/data/UCR")
        if X_train is None:
            raise ValueError(f"Dataset {dataset} could not be downloaded")
    y_train, y_test = label_encoder(y_train, y_test)

    # Define model architecture
    input_shape = X_train.shape[1:]
    classes = np.unique(y_train)
    n_classes = len(classes)
    model = InceptionModelConstructorV1(
        input_shape, n_classes, depth=params["depth"], n_filters=params["out_channels"],
        bottleneck_size=params["bottleneck_channels"], kernel_size=params["kernel_sizes"]).build_network()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )
    # print(model.summary())

    # Create model folder if it does not exist
    results_path = f"./experiments/models/{dataset}/{exp_name}/{exp_hash}"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Model fit
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=params["es_patience"], verbose=1)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=params["lr_patience"], min_lr=0.0001, verbose=1)
    # mcp_save = tf.keras.callbacks.ModelCheckpoint(f"{results_path}/model.hdf5", save_best_only=True, monitor='val_loss', mode='min')
    training_history = model.fit(
        X_train, y_train,
        batch_size=params['batch_size'],
        shuffle=True,
        validation_split=0.1,
        epochs=params["epochs"],
        callbacks=[early_stopping, lr_scheduler],
    )
    model.save(f"{results_path}/model.hdf5")

    # Evaluation
    y_train_probs = model.predict(X_train)
    predicted_train_labels = np.argmax(y_train_probs, axis=1)
    print(classification_report(y_train, predicted_train_labels))
    y_test_probs = model.predict(X_test)
    predicted_test_labels = np.argmax(y_test_probs, axis=1)
    print(classification_report(y_test, predicted_test_labels))

    # Export training params
    params = {**{'experiment_hash': exp_hash}, **params}
    with open(f"{results_path}/train_params.json", "w") as outfile:
        json.dump(params, outfile)

    # Export result metrics
    if n_classes > 2:
        train_roc_auc = roc_auc_score(y_train, y_train_probs, multi_class='ovr', average='macro')
        test_roc_auc = roc_auc_score(y_test, y_test_probs, multi_class='ovr', average='macro')
    else:
        train_roc_auc = roc_auc_score(y_train, y_train_probs[:, 1], average='macro')
        test_roc_auc = roc_auc_score(y_test, y_test_probs[:, 1], average='macro')
    train_f1 = f1_score(y_train, predicted_train_labels, average='weighted')
    test_f1 = f1_score(y_test, predicted_test_labels, average='weighted')
    train_acc = accuracy_score(y_train, predicted_train_labels)
    test_acc = accuracy_score(y_test, predicted_test_labels)
    result_metrics = {
        'train_roc_auc': train_roc_auc, 'test_roc_auc': test_roc_auc,
        'train_f1': train_f1, 'test_f1': test_f1,
        'train_acc': train_acc, 'test_acc': test_acc,
    }
    with open(f"{results_path}/metrics.json", "w") as outfile:
        json.dump(result_metrics, outfile)

    # Export confusion matrix
    cm = confusion_matrix(y_test, predicted_test_labels)
    cmp = ConfusionMatrixDisplay(cm, display_labels=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(12, 12))
    cmp.plot(ax=ax).figure_.savefig(f"{results_path}/confusion_matrix.png")

    # Export loss training loss evolution
    fig, (ax1, ax2) = plt.subplots(1, 2)
    pd.DataFrame(training_history.history)[['loss', 'val_loss']].plot(ax=ax1)
    pd.DataFrame(training_history.history)[['acc', 'val_acc']].plot(ax=ax2)
    plt.savefig(f"{results_path}/loss_curve.png")


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
            # Merge all info
            experiment_info = {**train_params, **metrics}
            experiment_info_list.append(experiment_info)
        except Exception:
            pass

    # Create the a dataframe containing all info and store it
    experiment_results_df = pd.DataFrame.from_records(experiment_info_list).sort_values("test_f1", ascending=False)
    best_experiment_hash = experiment_results_df.iloc[0]['experiment_hash']
    experiment_results_df.to_excel(f"{experiment_folder}/all_combination_results.xlsx")

    # Move the model best model to the experiment folder
    shutil.copyfile(
        f"{experiment_folder}/{best_experiment_hash}/model.hdf5",
        f"{experiment_folder}/model.hdf5"
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
                train_experiment(
                    dataset,
                    experiment_name,
                    experiment_hash,
                    experiment_params
                )
            except (ValueError, FileNotFoundError, TypeError) as msg:
                print(msg)

        # Compare performance of combinations and select the best one
        if os.path.isdir(f"./experiments/models/{dataset}/{experiment_name}"):
            select_best_model(dataset, experiment_name)

    print('Finished')

