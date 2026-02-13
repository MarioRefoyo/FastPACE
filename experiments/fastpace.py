import os
import pickle
import json
from collections import OrderedDict
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

from experiments.experiment_utils import load_parameters_from_json,generate_settings_combinations
from methods.FastPACECF import FastPACECF

from methods.nun_finders import GlobalNUNFinder
from experiments.experiment_utils import prepare_experiment, load_ae_outlier_calculator

"""DATASETS = [
    'CBF', 'CinCECGTorso', 'Coffee', "ECG200",
    "ECG5000", 'FacesUCR', 'FordA', 'GunPoint', 'HandOutlines',
    'ItalyPowerDemand',
    'NonInvasiveFatalECGThorax2', 'Plane',
    'ProximalPhalanxOutlineCorrect',
    'Strawberry', 'TwoPatterns'
]"""

DATASETS = [
    'BasicMotions', 'NATOPS',
    'UWaveGestureLibrary',
    # 'Cricket',
    'ArticularyWordRecognition', 'Epilepsy',
    'PenDigits',
    # 'PEMS-SF',
    'RacketSports',
    # 'SelfRegulationSCP1'
]

EXPERIMENT_FAMILY = 'hcem_final_plau'
PARAMS_PATH = f'experiments/params_cf/{EXPERIMENT_FAMILY}.json'
MODEL_TO_EXPLAIN_EXPERIMENT_NAME = "inceptiontime_pytorch"
OC_EXPERIMENT_NAME = 'pytorch_ae_basic_train_scaling'
SKIP_EXISTING_EXPERIMENTS = False

POOL_SIZE = 1


def worker_process(input_dict):
    experiment_dataset(**input_dict)


def generate_counterfactuals(X, nuns, cf_explainer):
    # Generate counterfactuals
    results = []
    for i in tqdm(range(len(X))):
        x_orig = X[i]
        nun = nuns[i]
        result = cf_explainer.generate_counterfactual(x_orig, nun_example=nun)
        results.append(result)
    return results


def create_metrics_dict_from_results(results_df, only_valids, not_nuns):
    results_df = results_df.drop("weights", axis=1)

    # Calculate valids
    valid_mean = results_df["valid"].mean()
    valid_std = results_df["valid"].std()
    is_nun_mean = results_df["is_nun"].mean()
    is_nun_std = results_df["is_nun"].std()

    # Select the solution set considered as valid
    if only_valids:
        results_df = results_df[results_df["valid"] == 1]
    if not_nuns:
        results_df = results_df[results_df["is_nun"] == 0]

    metrics_mean_dict = results_df.mean(axis=0).to_dict()
    metrics_mean_dict = {f"{k}_mean": v for k, v in metrics_mean_dict.items()}
    metrics_mean_dict["valid_mean"] = valid_mean
    metrics_mean_dict["is_nun_mean"] = is_nun_mean

    metrics_std_dict = results_df.std(axis=0).to_dict()
    metrics_std_dict = {f"{k}_std": v for k, v in metrics_std_dict.items()}
    metrics_std_dict["valid_std"] = valid_std
    metrics_std_dict["is_nun_std"] = is_nun_std

    # Create final dict
    tuple_order = []
    for col in results_df.columns:
        tuple_order.append((f'{col}_mean', metrics_mean_dict[f'{col}_mean']))
        tuple_order.append((f'{col}_std', metrics_std_dict[f'{col}_std']))
    metrics_dict = OrderedDict(tuple_order)
    return metrics_dict


def store_metric_dict_from_results(results_df, only_valids, not_nuns, result_path, results_name):
    # Store experiment average metrics
    metrics_dict = create_metrics_dict_from_results(results_df, only_valids=only_valids, not_nuns=not_nuns)

    final_name_part = ""
    if only_valids:
        final_name_part = final_name_part + "_only_valids"
    if not_nuns:
        final_name_part = final_name_part + "_not_nuns"
    # Store
    with open(f'{result_path}/metrics{results_name}{final_name_part}.json', 'w') as f:
        json.dump(metrics_dict, f, sort_keys=True)
    return metrics_dict, final_name_part


def store_results(results, result_path, results_name):
    # Store experiment counterfactuals
    with open(f'{result_path}/counterfactuals{results_name}.pickle', 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    # Store results excel
    results_df = pd.DataFrame.from_records(results)
    results_df = results_df.drop(["best_mask", "cfs", "x_orig", "nun", "best_mask"], axis=1)
    results_df.to_excel(f"{result_path}/results{results_name}.xlsx")

    # Store experiment average metrics
    _, _ = store_metric_dict_from_results(results_df, False, False, result_path, results_name)
    _, _ = store_metric_dict_from_results(results_df, True, False, result_path, results_name)
    _, _ = store_metric_dict_from_results(results_df, False, True, result_path, results_name)
    _, _ = store_metric_dict_from_results(results_df, True, True, result_path, results_name)


def get_experiment_name_prefix(params):
    include_params_as_experiment_name = params.get("include_params_as_experiment_name", None)
    if include_params_as_experiment_name is not None:
        prefix = ""
        for param in include_params_as_experiment_name:
            prefix += f"_{params[param]}"
    else:
        prefix = ""
    return prefix


def experiment_dataset(dataset, model_to_explain, experiment_family, exp_name, params):
    print(f"Starting experiment {exp_name} in {dataset}")
    X_train, y_train, X_test, y_test, subset_idx, model_wrapper, y_pred_train, y_pred_test, ts_length, n_channels, n_classes = prepare_experiment(
        dataset, params, MODEL_TO_EXPLAIN_EXPERIMENT_NAME)
    
    # Get outlier calculator
    outlier_calculator = load_ae_outlier_calculator(dataset, OC_EXPERIMENT_NAME, X_train, params["data_format"])

    # Create the result folder
    experiment_name_prefix = get_experiment_name_prefix(params)
    rel_experiment_path = f'{dataset}/{model_to_explain}/{experiment_family}/{experiment_name_prefix}_{exp_name}'
    result_path = f'./experiments/results/{rel_experiment_path}'
    os.makedirs(result_path, exist_ok=True)

    # Get the NUNs
    nun_finder = GlobalNUNFinder(
        X_train, y_train, y_pred_train, distance='euclidean',
        from_true_labels=False, data_format=params["data_format"]
    )

    # TRAIN AGENT
    # Get nuns
    nuns_train, _, _ = nun_finder.retrieve_nuns(X_train, y_pred_train)
    nuns, desired_classes, distances = nun_finder.retrieve_nuns(X_test, y_pred_test)

        # Define agent and train
    if params["tensorboard"]:
        tensorboard_path = f"./experiments/logs/RLCFE/{rel_experiment_path}"
    else:
        tensorboard_path = None

    algorithm_params = {k.replace("algorithm_", ""): v for k, v in params.items() if "algorithm_" in k}
    # Filter cemnn params to create preproc dataset name
    cemnn_params = {k.replace("cemnn_", ""): v for k, v in algorithm_params.items() if "cemnn_" in k}
    # Compute inference params upfront so hashes match inference settings
    inf_params = dict(
        train_num_simulations_ratio=cemnn_params["num_simulations_ratio"],
        train_planning_steps=cemnn_params["planning_steps"],
        train_cem_iters=cemnn_params["cem_iters"],
        train_plan_every=cemnn_params["plan_every"],
        inf_num_simulations_ratio=cemnn_params["num_simulations_ratio"],
        inf_planning_steps=cemnn_params["planning_steps"],
        inf_cem_iters=cemnn_params["cem_iters"],
        inf_plan_every=cemnn_params["plan_every"],

        search_target=cemnn_params["search_target"],
        train_target=None,  
        train_target_type=None,  
        train_pisoft_func=None,  
        model_blend_beta=None,
    )
    other_algorithm_params = dict(
        inf_model_blend_beta=None
    )
    # Remove cemnn params from algorithm params
    algorithm_params.update(inf_params)
    algorithm_params.update(other_algorithm_params)
    algorithm_params = {k.replace("cemnn_", ""): v for k, v in algorithm_params.items()}

    # Define pretrain params
    X_pretrain = X_train
    nun_pretrain = nuns_train

    # ----------- DEFINE EXPLAINER -----------
    standardize = True
    if params["scaling"] == "standard":
        standardize = False
    # Create the hierarchies from pcts
    ts_block_pcts = []
    ch_block_pcts = []
    for ts_ch_hierarchy_pcts in algorithm_params["ts_ch_block_pcts"]:
        ts_pct = ts_ch_hierarchy_pcts[0]
        ch_pct = ts_ch_hierarchy_pcts[1]
        ts_block_pcts.append(ts_pct)
        ch_block_pcts.append(ch_pct)
    # Get training pcts
    ts_train_pct = ch_train_pct = None

    # Get pruning
    if "pruning_ch_block_pct" in cemnn_params:
        pruning_ch_block_pct = cemnn_params["pruning_ch_block_pct"]
    else:
        pruning_ch_block_pct = None

    cf_explainer = FastPACECF(
        model_wrapper, outlier_calculator, X_pretrain, nun_pretrain,
        latent_ts_block_pcts=ts_block_pcts, latent_ts_block_train_pct=ts_train_pct,
        ch_block_pcts=ch_block_pcts, ch_block_train_pct=ch_train_pct,
        pruning_ch_block_pct=pruning_ch_block_pct,
        channel_groups=params["channel_groups"], standardize=standardize, ch_similarity=params["ch_similarity"],
        reward_type=params["reward_type"],
        non_valid_penalization=params["non_valid_penalization"],
        weight_losses=params["weight_losses"], mask_init=params["mask_init"],
        max_steps=params["max_steps"],
        algorithm_params=algorithm_params,
        device="cuda", tensorboard_path=tensorboard_path,
    )

    # Generate counterfactuals in train
    """results_train = generate_counterfactuals(X_train, nuns_train, cf_explainer,
                                             specific_train_steps=params["specific_train_steps"],
                                             specific_tensorboard_path=specific_tensorboard_path)
    store_results(results_train, result_path, results_name="_train")

    # Store experiment plots
    x_cfs_list_train = [result["cfs"] for result in results_train]
    x_cfs_train = np.concatenate(x_cfs_list_train, axis=0)
    plot_counterfactuals(X_train, nuns_train[:, 0, :, :], x_cfs_train, params["data_format"], store_path=result_path,
                         file_end="_train")"""

    # START COUNTERFACTUAL GENERATION
    results = generate_counterfactuals(X_test, nuns, cf_explainer)
    # Add training time
    results = [{**result} for result in results]
    store_results(results, result_path, results_name="")

    """# Store experiment plots
    x_cfs_list = [result["cfs"] for result in results]
    x_cfs = np.concatenate(x_cfs_list, axis=0)
    plot_counterfactuals(X_test, nuns[:, 0, :, :], x_cfs, params["data_format"], store_path=result_path,
                         file_end="")"""

    # Store experiment metadata
    params["X_test_indexes"] = subset_idx.tolist()
    with open(f'{result_path}/params.json', 'w') as fp:
        json.dump(params, fp, sort_keys=True)


def create_summary_table(dataset, model_to_explain, experiment_family):
    def save_concatenation_tables_from_excel(experiment_sub_dirs, file_end, only_valids, not_nuns):
        # Iterate through the combinations and retrieve the results file
        experiment_info_list = []
        params_set = set()
        for experiment_sub_dir in experiment_sub_dirs:
            results_path = f'{experiment_folder}/{experiment_sub_dir}'
            try:
                # Read the params file
                with open(f"{results_path}/params.json") as f:
                    params = json.load(f)
                    params = {**{"experiment_name": experiment_sub_dir}, **params}

                # Read the metrics file
                results_df = pd.read_excel(f"{results_path}/results{file_end}.xlsx", index_col=0)
                metrics_dict, final_name_part = store_metric_dict_from_results(results_df, only_valids, not_nuns, results_path, file_end)

                # Merge all info
                experiment_info = {**params, **metrics_dict}
                experiment_info_list.append(experiment_info)
                params_set.update(list(params.keys()))
            except FileNotFoundError:
                print(f"Experiment {experiment_sub_dir} not saved.")

        # Create the dataframe containing all info and store it
        all_results_df = pd.DataFrame.from_records(experiment_info_list)
        all_results_df = all_results_df.drop("X_test_indexes", axis=1)
        param_list = list(params)
        param_list.remove("seed")
        param_list.remove("experiment_name")
        for column in all_results_df.columns:
            if all_results_df[column].dtype == object:
                all_results_df[column] = all_results_df[column].astype(str)

        experiment_results_df = all_results_df.sort_values("improvement_over_nun_mean", ascending=False)
        experiment_results_df.to_excel(f"{experiment_folder}/concatenated_results{file_end}{final_name_part}.xlsx")


    experiment_folder = f'./experiments/results/{dataset}/{model_to_explain}/{experiment_family}'
    # Locate all experiment hashes for the given dataset by inspecting the folders
    experiment_sub_dirs = [f for f in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, f))]

    # Store results
    # save_concatenation_tables_from_excel(experiment_sub_dirs, file_end="_train", only_valids=False, not_nuns=False)
    save_concatenation_tables_from_excel(experiment_sub_dirs, file_end="", only_valids=False, not_nuns=False)

    # save_concatenation_tables_from_excel(experiment_sub_dirs, file_end="_train", only_valids=True, not_nuns=False)
    save_concatenation_tables_from_excel(experiment_sub_dirs, file_end="", only_valids=True, not_nuns=False)

    # save_concatenation_tables_from_excel(experiment_sub_dirs, file_end="_train", only_valids=False, not_nuns=True)
    save_concatenation_tables_from_excel(experiment_sub_dirs, file_end="", only_valids=False, not_nuns=True)

    # save_concatenation_tables_from_excel(experiment_sub_dirs, file_end="_train", only_valids=True, not_nuns=True)
    save_concatenation_tables_from_excel(experiment_sub_dirs, file_end="", only_valids=True, not_nuns=True)


if __name__ == "__main__":
    # Load parameters
    all_params = load_parameters_from_json(PARAMS_PATH)
    params_combinations = generate_settings_combinations(all_params)

    for dataset in DATASETS:
        # Calculate experiment inputs
        experiment_inputs_list = []
        for experiment_name, experiment_params in params_combinations.items():
            experiment_inputs = dict(
                dataset=dataset,
                model_to_explain=MODEL_TO_EXPLAIN_EXPERIMENT_NAME,
                experiment_family=EXPERIMENT_FAMILY,
                exp_name=experiment_name,
                params=experiment_params
            )
            if SKIP_EXISTING_EXPERIMENTS:
                # Create the result folder
                experiment_name_prefix = get_experiment_name_prefix(experiment_params)
                rel_experiment_path = f'{dataset}/{MODEL_TO_EXPLAIN_EXPERIMENT_NAME}/{EXPERIMENT_FAMILY}/{experiment_name_prefix}_{experiment_name}'
                result_path = f'./experiments/results/{rel_experiment_path}'
                if os.path.exists(f"{result_path}/counterfactuals.pickle"):
                    print(f"Skipping experiment {experiment_name} in {dataset} (already done).")
                    continue
                else:
                    experiment_inputs_list.append(experiment_inputs)
            else:
                experiment_inputs_list.append(experiment_inputs)

        # Execute experiments
        if POOL_SIZE > 1:
            with Pool(POOL_SIZE) as p:
                _ = list(tqdm(p.imap(worker_process, experiment_inputs_list), total=len(experiment_inputs_list)))
        else:
            for experiment_inputs in experiment_inputs_list:
                worker_process(experiment_inputs)

        # Get experiment results
        create_summary_table(dataset, MODEL_TO_EXPLAIN_EXPERIMENT_NAME, EXPERIMENT_FAMILY)

    # Revisit solutions and create a summary result table
    print('Finished')
