import os
import random
import json
import shutil

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, f1_score, accuracy_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.models.pytorch_FCN import FCN
from experiments.models.pytorch_InceptionTime import InceptionModel
from experiments.models.pytorch_learners import BasicLearner
from experiments.data_utils import local_data_loader, ucr_data_loader, label_encoder

from torchsummary import summary


def select_best_model(dataset, exp_name):
    experiment_folder = f"./experiments/models/{dataset}/{exp_name}"
    # Locate all experiment hashes for the given dataset by inspecting the folders
    experiment_sub_dirs = [f for f in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, f))]
    # Iterate through the combinations and retrieve the results file
    experiment_info_list = []
    train_params_set = set()
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
            train_params_set.update(list(train_params.keys()))
        except FileNotFoundError:
            print(f"Experiment {experiment_sub_dir} not saved.")

    # Create the a dataframe containing all info and store it
    all_results_df = pd.DataFrame.from_records(experiment_info_list)
    param_list = list(train_params_set)
    param_list.remove("seed")
    param_list.remove("experiment_hash")
    param_list.remove("total_params")
    for column in all_results_df.columns:
        if all_results_df[column].dtype == object:
            all_results_df[column] = all_results_df[column].astype(str)

    # Create averaged results across seeds
    results_mean = all_results_df.drop("experiment_hash", axis=1).groupby(param_list).mean()
    results_std = all_results_df.drop("experiment_hash", axis=1).groupby(param_list).std()
    results_counts = all_results_df.drop("experiment_hash", axis=1).groupby(param_list).size()
    average_results_df = pd.DataFrame(index=results_mean.index)
    average_results_df["total_params"] = results_mean["total_params"]
    for metric in metrics.keys():
        average_results_df[f"{metric}_mean"] = results_mean[metric]
        average_results_df[f"{metric}_std"] = results_std[metric]
    average_results_df["n_seeds"] = results_counts
    aux_seed_0 = all_results_df[all_results_df["seed"] == 0][param_list + ["experiment_hash"]].rename(
        columns={"experiment_hash": "seed_0_experiment_hash"}).set_index(param_list)
    average_results_df = average_results_df.merge(aux_seed_0, how="left", left_index=True, right_index=True)

    experiment_results_df = all_results_df.sort_values("test_f1", ascending=False)
    best_experiment_hash = experiment_results_df.iloc[0]['experiment_hash']
    experiment_results_df.to_excel(f"{experiment_folder}/all_results.xlsx")
    experiment_average_results_df = average_results_df.sort_values("test_f1_mean", ascending=False)
    experiment_average_results_df.to_excel(f"{experiment_folder}/average_seed_results.xlsx")

    # Move the model best model to the experiment folder
    shutil.copyfile(
        f"{experiment_folder}/{best_experiment_hash}/model_weights.pth",
        f"{experiment_folder}/model_weights.pth"
    )
    shutil.copyfile(
        f"{experiment_folder}/{best_experiment_hash}/train_params.json",
        f"{experiment_folder}/train_params.json"
    )


def model_selector(dataset, in_channels, ts_len, n_classes, params):
    # Model trainer
    if "weight_decay" in params:
        weight_decay = params["weight_decay"]
    else:
        weight_decay = 0

    # Select criterion
    try:
        if params["criterion"] == "NLL":
            criterion = nn.NLLLoss(reduction='mean')
        elif params["criterion"] == "CE":
            criterion = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError("Not valid criterion")
    except KeyError:
        criterion = nn.CrossEntropyLoss(reduction="mean")

    # Create model and learner
    model_type = params["model_type"]
    if model_type == "FCN":
        model = FCN(
            in_channels=in_channels, channels=params["channels"], kernel_sizes=params["kernel_sizes"],
            num_classes=n_classes
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
        trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    elif model_type == "InceptionTime":
        model = InceptionModel(
            depth=params["depth"],
            in_channels=in_channels, out_channels=params["out_channels"],
            bottleneck_channels=params["bottleneck_channels"], kernel_sizes=params["kernel_sizes"],
            num_classes=n_classes
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params["lr_patience"], factor=0.5)
        trainer = BasicLearner(model, criterion, num_epochs=params["epochs"], es_patience=params["es_patience"])
    else:
        raise ValueError("Not valid model type")

    return model, optimizer, scheduler, trainer


def load_back_bone(back_bone_type, params, dataset):
    model_params = {k.replace("back_bone_model_", ""): v for k, v in params.items() if "back_bone_model" in k}
    pretrained_model = infer_model_folder(
        dataset,
        scaling=params["scaling"], seed=params["seed"],
        model_type=back_bone_type, model_params=model_params)
    back_bone = pretrained_model.back_bone
    return back_bone


def infer_model_folder(dataset, scaling, seed, model_type, model_params):
    results_path = f"./models/{dataset}/{model_type}"
    all_results_df = pd.read_excel(f"{results_path}/all_results.xlsx")

    # Track experiment
    filtered_results = all_results_df[
        (all_results_df["scaling"] == scaling) & (all_results_df["seed"] == seed)]
    for param, value in model_params.items():
        filtered_results = filtered_results[filtered_results[param] == str(value)]
    experiment_hash = filtered_results.iloc[0]["experiment_hash"]
    # Load model
    # model_params = {**{"in_channels": in_channels}, **model_params}
    # model.load_state_dict(torch.load(f"{results_path}/{experiment_hash}/model.pth"))
    model = torch.load(f"{results_path}/{experiment_hash}/model.pth")
    return model


def train_experiment(dataset, exp_name, exp_hash, params):
    # Set seed
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    random.seed(params["seed"])

    # Load data
    scaling = params["scaling"]
    if os.path.isdir(f"./experiments/data/UCR/{dataset}"):
        X_train, y_train, X_test, y_test, _, _ = local_data_loader(str(dataset), scaling, backend="torch", data_path="./experiments/data")
    else:
        os.makedirs(f"./experiments/data/UCR/{dataset}")
        X_train, y_train, X_test, y_test = ucr_data_loader(dataset, scaling, backend="torch", store_path="./experiments/data/UCR")
        if X_train is None:
            raise ValueError(f"Dataset {dataset} could not be downloaded")
    y_train, y_test = label_encoder(y_train, y_test)
    classes = np.unique(y_train)
    n_classes = len(classes)
    n_channels = X_train.shape[1]
    ts_length = X_train.shape[2]

    # y_train, y_test = one_hot_encoder(y_train, y_test)
    # n_classes = y_train.shape[1]

    # Create model folder if it does not exist
    results_path = f"./experiments/models/{dataset}/{exp_name}/{exp_hash}"
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Define model architecture
    model, optimizer, scheduler, trainer = model_selector(dataset, n_channels, ts_length, n_classes, params)
    # summary(model.cuda(), X_train.shape[1:])

    # Model Fit
    epoch_metrics = trainer.fit(
        X_train, y_train,
        optimizer, scheduler,
        batch_size=params['batch_size'],
        val_size=0.1
    )
    training_history = pd.DataFrame(epoch_metrics)

    # Evaluation
    y_train_probs = trainer.predict(X_train)
    predicted_train_labels = np.argmax(y_train_probs, axis=1)
    # print(classification_report(y_train, predicted_train_labels))
    y_test_probs = trainer.predict(X_test)
    predicted_test_labels = np.argmax(y_test_probs, axis=1)
    print(classification_report(y_test, predicted_test_labels))

    # Export model
    torch.save(model.state_dict(), f"{results_path}/model_weights.pth")

    # Export training params
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    params = {**{'experiment_hash': exp_hash, "total_params": pytorch_total_params/1e6}, **params}
    with open(f"{results_path}/train_params.json", "w") as outfile:
        json.dump(params, outfile)

    # Clear GPU cache
    torch.cuda.empty_cache()

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
    training_history[['train_loss', 'val_loss']].plot(ax=ax1)
    # pd.DataFrame(training_history.history)[['acc', 'val_acc']].plot(ax=ax2)
    plt.savefig(f"{results_path}/loss_curve.png")
