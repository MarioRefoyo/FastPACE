import os

from pytorch_utils import select_best_model, train_experiment
from experiments.experiment_utils import load_parameters_from_json, generate_settings_combinations


# Univariate Datasets
DATASETS = [
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
# Multivariate datasets
DATASETS = [
    # 'BasicMotions', 'NATOPS', 'UWaveGestureLibrary',
    # 'ArticularyWordRecognition', 'AtrialFibrillation', 'CharacterTrajectories', 'Cricket',
    # 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection',
    # 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels',
    # 'Libras', 'LSST', # 'MotorImagery',
    # 'PenDigits',
    # 'PEMS-SF', 'RacketSports',
    'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump'
]
DATASETS = [
    # 'BasicMotions', 'NATOPS', 'UWaveGestureLibrary',
    # 'ArticularyWordRecognition', 'Epilepsy', 'SelfRegulationSCP1',
    'LSST', 'PenDigits', 'Heartbeat'
]
"""DATASETS = [
    'CBF', 'CinCECGTorso', 'Coffee', "ECG200", "ECG5000", 'FacesUCR', 'FordA', 'GunPoint', 'HandOutlines',
    'ItalyPowerDemand', 'NonInvasiveFatalECGThorax2', 'Plane', 'ProximalPhalanxOutlineCorrect',
    'Strawberry', 'TwoPatterns'
]"""
DATASETS = [
    'BasicMotions', 'NATOPS', 'UWaveGestureLibrary', 'Cricket',
    'ArticularyWordRecognition', 'Epilepsy',
    'PenDigits',
    'PEMS-SF',
    'RacketSports', 'SelfRegulationSCP1'
]
PARAMS_PATH = 'experiments/params_model_training/pytorch_inceptiontime.json'


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
                    experiment_params,
                )
            except (ValueError, FileNotFoundError, TypeError) as msg:
                print(msg)

        # Compare performance of combinations and select the best one
        if os.path.isdir(f"./experiments/models/{dataset}/{experiment_name}"):
            select_best_model(dataset, experiment_name)

    print('Finished')

