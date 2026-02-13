import os
import random
import json
import pickle
import itertools
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tslearn.neighbors import KNeighborsTimeSeries
import torch
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.data_utils import local_data_loader, ucr_data_loader, label_encoder
from experiments.models.pytorch_utils import model_selector

from methods.outlier_calculators import AEOutlierCalculator
from experiments.models.pytorch_Autoencoders import AEModelConstructorV1


def get_subsample(X_test, y_test, n_instances, seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    subset_idx = np.random.choice(len(X_test), n_instances, replace=False)
    subset_idx = np.sort(subset_idx)
    X_test = X_test[subset_idx]
    y_test = y_test[subset_idx]
    return X_test, y_test, subset_idx


def get_hash_from_params(params):
    params_str = ''.join(f'{key}={value},' for key, value in sorted(params.items()))
    params_hash = hashlib.sha1(params_str.encode()).hexdigest()
    return params_hash


def generate_settings_combinations(original_dict):
    # Create a list of keys with lists as values
    list_keys = [key for key, value in original_dict.items() if isinstance(value, list)]
    # Generate all possible combinations
    combinations = list(itertools.product(*[original_dict[key] for key in list_keys]))
    # Create a set of experiments dictionaries with unique combinations
    result = {}
    for combo in combinations:
        new_dict = original_dict.copy()
        for key, value in zip(list_keys, combo):
            new_dict[key] = value
        experiment_hash = get_hash_from_params(new_dict)
        result[experiment_hash] = new_dict
    return result


def load_parameters_from_json(json_filename):
    with open(json_filename, 'r') as json_file:
        params = json.load(json_file)
    return params


def store_partial_cfs(results, s_start, s_end, dataset, model_to_explain_name, file_suffix_name):
    # Create folder for dataset if it does not exist
    os.makedirs(f'./experiments/results/{dataset}/', exist_ok=True)
    os.makedirs(f'./experiments/results/{dataset}/{model_to_explain_name}/', exist_ok=True)
    os.makedirs(f'./experiments/results/{dataset}/{model_to_explain_name}/{file_suffix_name}/', exist_ok=True)
    with open(f'./experiments/results/{dataset}/{model_to_explain_name}/{file_suffix_name}/{file_suffix_name}_{s_start:04d}-{s_end:04d}.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


def nun_retrieval(query, predicted_label, distance, n_neighbors, X_train, y_train, y_pred, from_true_labels=False):
    df_init = pd.DataFrame(y_train, columns=['true_label'])
    df_init["pred_label"] = y_pred
    df_init.index.name = 'index'

    if from_true_labels:
        label_name = 'true_label'
    else:
        label_name = 'pred_label'
    df = df_init[[label_name]]
    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
    knn.fit(X_train[list(df[df[label_name] != predicted_label].index.values)])
    dist, ind = knn.kneighbors(np.expand_dims(query, axis=0), return_distance=True)
    distances = dist[0]
    index = df[df[label_name] != predicted_label].index[ind[0][:]]
    label = df[df.index.isin(index.tolist())].values[0]
    return distances, index, label


def prepare_experiment(dataset, params, model_to_explain):
    # Set seed
    if params["seed"] is not None:
        np.random.seed(params["seed"])
        random.seed(params["seed"])

    # Load dataset data
    scaling = params["scaling"]
    data_format = params["data_format"]
    X_train, y_train, X_test, y_test, ts_length, n_channels = local_data_loader(
        str(dataset), scaling, backend=data_format, data_path="./experiments/data"
    )
    y_train, y_test = label_encoder(y_train, y_test)
    classes = np.unique(y_train)
    n_classes = len(classes)

    # Get a subset of testing data if specified
    if (params["subset"]) & (len(y_test) > params["subset_number"]):
        X_test, y_test, subset_idx = get_subsample(X_test, y_test, params["subset_number"], params["seed"])
    else:
        subset_idx = np.arange(len(X_test))

    # Get model
    model_folder = f'experiments/models/{dataset}/{model_to_explain}'
    model_wrapper = load_model(model_folder, dataset, data_format, n_channels, ts_length, n_classes)

    # Predict
    y_pred_test_logits = model_wrapper.predict(X_test)
    y_pred_train_logits = model_wrapper.predict(X_train)
    y_pred_test = np.argmax(y_pred_test_logits, axis=1)
    y_pred_train = np.argmax(y_pred_train_logits, axis=1)
    # Classification report
    print(classification_report(y_test, y_pred_test))

    return X_train, y_train, X_test, y_test, subset_idx, model_wrapper, y_pred_train, y_pred_test, ts_length, n_channels, n_classes


def load_model(model_folder, dataset, data_format, n_channels, ts_length, n_classes, compile=True):
    if os.path.exists(f'{model_folder}/model.hdf5'):
        backend = "tf"
        model = tf.keras.models.load_model(f'{model_folder}/model.hdf5')
        model_wrapper = ModelWrapper(model, backend, data_format)
    elif os.path.exists(f'{model_folder}/model_weights.pth'):
        backend = "torch"
        # Load train params
        with open(f"{model_folder}/train_params.json") as f:
            train_params = json.load(f)
        model, _, _, _ = model_selector(dataset, n_channels, ts_length, n_classes, train_params)
        model_weights = torch.load(f'{model_folder}/model_weights.pth', weights_only=True)
        model.load_state_dict(model_weights)
        model_wrapper = ModelWrapper(model, backend, data_format)
        if compile:
            # Trace it with JIT
            if data_format == "tf":
                example_input = np.random.rand(1, ts_length, n_channels).astype(np.float32)
            else:
                example_input = np.random.rand(1, n_channels, ts_length).astype(np.float32)
            model_wrapper.compile_with_jit(example_input)
    else:
        raise ValueError("Not valid model path or backend")

    return model_wrapper

def load_ae_outlier_calculator(dataset, OC_EXPERIMENT_NAME, X_train, data_format):

    if OC_EXPERIMENT_NAME == "ae_basic_train_scaling":
        ae_model = tf.keras.models.load_model(f'./experiments/models/{dataset}/{OC_EXPERIMENT_NAME}/model.hdf5')
        outlier_calculator = AEOutlierCalculator(ae_model, X_train, "tf", data_format)
    elif OC_EXPERIMENT_NAME == "pytorch_ae_basic_train_scaling":
        if data_format == "tf":
            input_shape = np.transpose(X_train, (0, 2, 1)).shape[1:]
        else:
            input_shape = X_train.shape[1:]
        model_folder = f'./experiments/models/{dataset}/{OC_EXPERIMENT_NAME}'
        with open(f"{model_folder}/train_params.json") as f:
            train_params = json.load(f)
        ae_model = AEModelConstructorV1(
            input_shape, train_params["temporal_strides"], train_params["compression_rate"]
        ).get_model(train_params["model_type"])
        state_dict = torch.load(f"{model_folder}/model_weights.pth", map_location="cpu")
        ae_model.load_state_dict(state_dict)
        ae_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ae_model.eval()
        outlier_calculator = AEOutlierCalculator(ae_model, X_train, "torch", data_format)
    else:
        print("Not know outlier calculator.")
        outlier_calculator = None
    
    return outlier_calculator


class ModelWrapper:
    def __init__(self, model, backend, data_format):
        assert backend in ["torch", "tf"]
        assert data_format in ["torch", "tf"]

        self.model = model
        self.backend = backend
        self.data_format = data_format

        # Prepare for backend
        if self.backend == "torch":
            self.framework = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        elif self.backend == "tf":
            self.framework = tf
        else:
            raise ValueError("Unsupported backend: choose 'torch' or 'tf'.")

    def compile_with_jit(self, example_input: np.ndarray):
        if self.backend != "torch":
            raise RuntimeError("JIT compilation is only available for PyTorch models.")

        # Prepare example input for tracing
        if len(example_input.shape) == 2:
            example_input = np.expand_dims(example_input, axis=0)
        if self.backend != self.data_format:
            example_input = np.transpose(example_input, (0, 2, 1))

        example_tensor = torch.from_numpy(example_input).float().to(self.device)

        # Trace the model and replace the original
        with torch.inference_mode():
            traced = torch.jit.trace(self.model, example_tensor)
            traced = torch.jit.freeze(traced.eval())
            self.base_model = self.model
            self.model = traced
            self.model = torch.jit.optimize_for_inference(self.model)

    def predict(self, x: np.ndarray, try_half: bool = False) -> np.ndarray:

        # Append if necessary
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=0)

        # Transpose if backend and data format are not equal
        if self.backend != self.data_format:
            x = np.transpose(x, (0, 2, 1))

        # Get model predictions
        if self.backend == "torch":
            # x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            x_tensor = torch.from_numpy(x).float().to(self.device)
            with torch.inference_mode():
                if try_half:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        output = self.model(x_tensor)
                        output = torch.nn.functional.softmax(output, dim=1)
                else:
                    output = self.model(x_tensor)
                    output = torch.nn.functional.softmax(output, dim=1)
            return output.detach().cpu().numpy()
        elif self.backend == "tf":
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            output = self.model.predict(x_tensor, verbose=0)
            return output

    def predict_class(self, x: np.ndarray) -> np.ndarray:
        y_pred_logits = self.predict(x)
        y_pred = np.argmax(y_pred_logits, axis=1)
        return y_pred

    def to(self, device):
        if self.backend == "torch":
            self.device = torch.device(device)
            self.model.to(device)


def plot_counterfactuals(x_origs, nuns, x_cfs, data_format, plots_rows=5, plot_columns=5, store_path=None, file_end=""):
    assert x_origs.shape == x_cfs.shape == nuns.shape

    # Adapt data format to image
    if data_format == "torch":
        x_origs = x_origs.transpose(0, 2, 1)
        nuns = nuns.transpose(0, 2, 1)
        x_cfs = x_cfs.transpose(0, 2, 1)
    elif data_format == "tf":
        pass
    else:
        raise ValueError("Not valid data format")
    ts_length = x_origs.shape[1]
    n_channels = x_origs.shape[2]

    # Calculate the total amount of figures needed
    subplots_per_figure = plots_rows*plot_columns
    total_figures = int(np.ceil(x_origs.shape[0] / subplots_per_figure))

    # Create figures
    for figure_number in range(total_figures):

        # Select data in figure
        figure_x_origs = x_origs[(figure_number)*subplots_per_figure:(figure_number+1)*subplots_per_figure]
        figure_nuns = nuns[(figure_number)*subplots_per_figure:(figure_number+1)*subplots_per_figure]
        figure_x_cfs = x_cfs[(figure_number) * subplots_per_figure:(figure_number + 1) * subplots_per_figure]

        # Define outer grid
        fig = plt.figure(figsize=(12, 12))
        outer_grid = gridspec.GridSpec(plots_rows, plot_columns, wspace=0, hspace=0)

        # Iterate through outer grid plots
        for i_outer, outer in enumerate(outer_grid):
            # Get things to plot
            try:
                x_orig = figure_x_origs[i_outer]
                nun = figure_nuns[i_outer]
                x_cf = figure_x_cfs[i_outer]
            except:
                continue

            # Calculate mask
            found_counterfactual_mask = (x_orig != x_cf.reshape(1, ts_length, n_channels)).astype(int)
            diff_mask = np.diff(found_counterfactual_mask, prepend=0, append=0, axis=1)

            # Create inner grid
            inner_grid = gridspec.GridSpecFromSubplotSpec(n_channels, 1, wspace=0, hspace=0, subplot_spec=outer)
            sub_channel_axs = []
            for i in range(n_channels):
                ax = plt.Subplot(fig, inner_grid[i])

                ax.plot(nun.reshape(1, ts_length, n_channels)[:, :, i].flatten(), color='grey')
                ax.plot(x_cf.reshape(1, ts_length, n_channels)[:, :, i].flatten(), color='red')
                ax.plot(x_orig.reshape(1, ts_length, n_channels)[:, :, i].flatten(), color="#332288")
                # ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])

                # Add fill to changes
                starts = np.clip(np.where(diff_mask[0, :, i].flatten() == 1)[0] - 1, 0, ts_length)
                ends = np.where(diff_mask[0, :, i].flatten() == -1)[0]
                # Iterate over the mask segments and fill them with red color
                for i in range(0, len(starts)):
                    start_idx = starts[i]
                    end_idx = ends[i]
                    # plt.fill_between(comp_df.iloc[start_idx:end_idx].index, comp_df.min(), comp_df.max(), color='red', alpha=0.7)
                    ax.axvspan(start_idx, end_idx, alpha=0.2, color='red')

                """# Set column titles
                column_title = methods_trad_dict[method_name]
                # row_title = f'{dataset} \n instance {instance}'
                row_title = f'{dataset} [{instance}]'
                if m == 0:
                    ax.set_ylabel(row_title, fontsize=16, rotation=90)
                if i_dataset == 0:
                    ax.set_title(column_title, fontsize=18)"""

                # Add ax to list of subplots
                fig.add_subplot(ax)
                sub_channel_axs.append(ax)

        # Plot figure
        plt.tight_layout()
        if store_path is None:
            plt.show()
        else:
            plt.savefig(f'{store_path}/counterfactuals_{figure_number}{file_end}.png')
