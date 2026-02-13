import os
import copy
import pickle
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from multiprocessing import Pool

from methods.outlier_calculators import AEOutlierCalculator, IFOutlierCalculator, LOFOutlierCalculator
from experiments.experiment_utils import local_data_loader, label_encoder, nun_retrieval, get_subsample
from methods.nun_finders import GlobalNUNFinder, IndependentNUNFinder
from methods.MultiSubSpaCE.FitnessFunctions import fitness_function_mo
from experiments.experiment_utils import load_model, load_ae_outlier_calculator


def get_start_end_subsequence_positions(orig_change_mask):
    # ----- Get potential extension locations
    ones_mask = np.in1d(orig_change_mask, 1).reshape(orig_change_mask.shape)
    # Get before and after ones masks
    before_ones_mask = np.roll(ones_mask, -1, axis=0)
    before_ones_mask[ones_mask.shape[0] - 1, :] = False
    after_ones_mask = np.roll(ones_mask, 1, axis=0)
    after_ones_mask[0, :] = False
    # Generate complete mask of after and before ones (and set to False the places where the original ones exist)
    before_after_ones_mask = before_ones_mask + after_ones_mask
    before_after_ones_mask[ones_mask] = False
    return before_after_ones_mask


def calculate_change_mask(x_orig, x_cf, x_nun=None, verbose=0):
    # Get original change mask (could contain points with common values between NUN, x_orig and x_cf)
    orig_change_mask = (x_orig != x_cf).astype(int)
    orig_change_mask = orig_change_mask.T.reshape(-1, 1)

    # Find common values
    if x_nun is not None:
        cv_xorig_nun = (x_orig == x_nun)
        cv_nun_cf = (x_nun == x_cf)
        cv_all = (cv_xorig_nun & cv_nun_cf).astype(int)
        cv_all = cv_all.T.reshape(-1, 1)

        # Check if those common values are at the start or end of a current subsequence
        start_end_mask = cv_all & get_start_end_subsequence_positions(orig_change_mask).astype(int)
        if verbose==1:
            print(orig_change_mask.flatten())
            print(get_start_end_subsequence_positions(orig_change_mask).flatten())
            print(cv_all.flatten())
            print(start_end_mask.flatten())

        # Add noise to those original points that are common to original, NUN and cf
        # are at the beginning or end of a subsequence on the change mask
        noise = np.random.normal(0, 1e-6, x_orig.shape)
        new_x_orig = x_orig + noise * start_end_mask.reshape(x_orig.shape, order='F')

        # Calculate adjusted change mask
        change_mask = (new_x_orig != x_cf).astype(int)
    else:
        change_mask = orig_change_mask.reshape(x_orig.shape, order='F')

    return change_mask


def load_dataset_for_eval(dataset, model_to_explain, osc_names, scaling="none"):
    data_format = "tf"
    X_train, y_train, X_test, y_test, ts_length, n_channels = local_data_loader(
        str(dataset), scaling=scaling, backend=data_format, data_path="./experiments/data")
    y_train, y_test = label_encoder(y_train, y_test)
    data_tuple = (X_train, y_train, X_test, y_test)
    classes = np.unique(y_train)
    n_classes = len(classes)

    # Load model
    model_folder = f'./experiments/models/{dataset}/{model_to_explain}'
    model_wrapper = load_model(model_folder, dataset, data_format, n_channels, ts_length, n_classes)

    # Predict
    y_pred_test_logits = model_wrapper.predict(X_test)
    y_pred_train_logits = model_wrapper.predict(X_train)
    y_pred_test = np.argmax(y_pred_test_logits, axis=1)
    y_pred_train = np.argmax(y_pred_train_logits, axis=1)

    # Load outlier calculators
    outlier_calculators = {}
    for osc_name, osc_exp_names in osc_names.items():
        if osc_name == "AE":
            outlier_calculator = load_ae_outlier_calculator(dataset, osc_exp_names, X_train, data_format)
            outlier_calculators[osc_name] = outlier_calculator
        elif osc_name == "IF":
            with open(f'./experiments/models/{dataset}/{osc_exp_names}/model.pickle', 'rb') as f:
                if_model = pickle.load(f)
            outlier_calculators[osc_name] = IFOutlierCalculator(if_model, X_train, data_format)
        elif osc_name == "LOF":
            with open(f'./experiments/models/{dataset}/{osc_exp_names}/model.pickle', 'rb') as f:
                lof_model = pickle.load(f)
            outlier_calculators[osc_name] = LOFOutlierCalculator(lof_model, X_train, data_format)
        else:
            raise ValueError("Not valid name in outlier calculator names.")

    # Get the NUNs
    possible_nuns = {}
    # Get nuns with global knn
    nun_finder = GlobalNUNFinder(
        X_train, y_train, y_pred_train, distance='euclidean',
        from_true_labels=False, data_format=data_format
    )
    gknn_nuns, desired_classes, distances = nun_finder.retrieve_nuns(X_test, y_pred_test)
    gknn_nuns = gknn_nuns[:, 0, :, :]
    possible_nuns['gknn'] = gknn_nuns
    """# Get nuns with individual knn for channels
    nun_finder = IndependentNUNFinder(
        X_train, y_train, y_pred_train, distance='euclidean', n_neighbors=1,
        from_true_labels=False, backend='tf', model=model
    )
    iknn_nuns, desired_classes, _ = nun_finder.retrieve_nuns(X_test, y_pred_test)
    iknn_nuns = iknn_nuns[:, 0, :, :]
    possible_nuns['iknn'] = iknn_nuns"""
    # NOTE: DESIRED CLASSES ARE ALWAYS THE SAME

    return data_tuple, y_pred_test, model_wrapper, outlier_calculators, possible_nuns, desired_classes


def process_method_dir(args):
    dataset, model_to_explain, method_dir_name, methods, model_wrapper, outlier_calculators, X_test, original_classes, possible_nuns, mo_weights, order = args
    results_df = pd.DataFrame()
    method_cfs_dataset = {}

    # Load solution cfs
    with open(f'./experiments/results/{dataset}/{model_to_explain}/{method_dir_name}/counterfactuals.pickle',
              'rb') as f:
        print(method_dir_name)
        method_cfs = pickle.load(f)

    # Load params
    with open(f'./experiments/results/{dataset}/{model_to_explain}/{method_dir_name}/params.json', 'r') as json_file:
        method_params = json.load(json_file)
        method_test_indexes = method_params["X_test_indexes"]

    # Get nuns used by the method depending on the name
    if "independent_channels_nun" in method_params:
        if method_params["independent_channels_nun"]:
            nuns = possible_nuns["iknn"]
        else:
            nuns = possible_nuns["gknn"]
    else:
        nuns = np.array([None] * len(X_test))

    # Calculate metrics
    method_name = methods[method_dir_name]
    method_metrics = calculate_method_metrics(
        model_wrapper, outlier_calculators, X_test[method_test_indexes],
        nuns[method_test_indexes], method_cfs, original_classes[method_test_indexes],
        method_name, mo_weights=mo_weights, order=order
    )
    method_metrics.insert(0, "ii", method_test_indexes)

    results_df = pd.concat([results_df, method_metrics])
    method_cfs_dataset[method_name] = method_cfs
    common_test_indexes = list(method_test_indexes)

    return results_df, method_cfs_dataset, common_test_indexes


def calculate_metrics_for_dataset_mp(dataset, methods, model_to_explain,
                                     data_tuple, original_classes, model_wrapper, outlier_calculators, possible_nuns,
                                     mo_weights=None):
    X_train, y_train, X_test, y_test = data_tuple

    cf_solution_dirs = [fname for fname in os.listdir(f'./experiments/results/{dataset}/{model_to_explain}') if
                        os.path.isdir(f'./experiments/results/{dataset}/{model_to_explain}/{fname}')]
    desired_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in cf_solution_dirs if cf_sol_dir in methods.keys()]
    valid_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in desired_cf_solution_dirs if os.path.isfile(
        f'./experiments/results/{dataset}/{model_to_explain}/{cf_sol_dir}/counterfactuals.pickle')]

    # Prepare arguments for parallel processing
    args = [
        (dataset, model_to_explain, method_dir_name, methods, model_wrapper, outlier_calculators, X_test,
         original_classes, possible_nuns, mo_weights, i + 1)
        for i, method_dir_name in enumerate(valid_cf_solution_dirs)
    ]

    # Use multiprocessing pool to parallelize the processing of each method directory
    with Pool(10) as pool:
        results = pool.map(process_method_dir, args)

    # Collect results
    results_df = pd.DataFrame()
    method_cfs_dataset = {}
    common_test_indexes = list(range(len(X_test)))

    for res_df, method_cfs, test_indexes in results:
        results_df = pd.concat([results_df, res_df])
        method_cfs_dataset.update(method_cfs)
        common_test_indexes = list(set(test_indexes).intersection(common_test_indexes))
        common_test_indexes.sort()

    # Calculate results table for the dataset
    means_df = results_df.groupby('method').mean()
    means_df = means_df.sort_values('order').drop('order', axis=1)
    stds_df = results_df.groupby('method').std()
    stds_df = stds_df.drop('order', axis=1)
    stds_df = stds_df.reindex(means_df.index)
    mean_std_df = means_df.round(2).astype(str) + " ± " + stds_df.round(2).astype(str)
    mean_std_df = mean_std_df.reset_index()
    results_df['dataset'] = dataset

    return mean_std_df, results_df, method_cfs_dataset, common_test_indexes


def calculate_metrics_for_dataset(dataset, methods, model_to_explain,
                                  data_tuple, original_classes, model, outlier_calculators, possible_nuns,
                                  mo_weights=None):
    X_train, y_train, X_test, y_test = data_tuple

    results_df = pd.DataFrame()
    root = f'./experiments/results/{dataset}/{model_to_explain}'

    # collect leaf dirs at depth 1 and 2 (root/<dir> and root/<dir>/<dir>)
    flat = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    lvl2 = []
    for d in flat:
        p = os.path.join(root, d)
        for dd in os.listdir(p):
            if os.path.isdir(os.path.join(p, dd)):
                lvl2.append(os.path.join(d, dd))  # keep relative subpath
    cf_solution_dirs = flat + lvl2

    # now match by leaf name, not the whole relative path
    desired_cf_solution_dirs = [
        rel for rel in cf_solution_dirs
        if os.path.basename(rel) in methods.keys()
    ]

    # require both files
    valid_cf_solution_dirs = [
        rel for rel in desired_cf_solution_dirs
        if os.path.isfile(os.path.join(root, rel, 'counterfactuals.pickle')) and
           os.path.isfile(os.path.join(root, rel, 'params.json'))
    ]

    common_test_indexes = list(range(len(X_test)))
    method_cfs_dataset = {}
    for i, rel in enumerate(valid_cf_solution_dirs):
        method_key = os.path.basename(rel)
        dir_path = os.path.join(root, rel)
        # Load solution cfs
        with open(os.path.join(dir_path, 'counterfactuals.pickle'), 'rb') as f:
            print(method_key)
            method_cfs = pickle.load(f)
        # Load params
        with open(os.path.join(dir_path, 'params.json'), 'r') as json_file:
            method_params = json.load(json_file)
            method_test_indexes = method_params["X_test_indexes"]
            data_format = method_params["data_format"]

        # Get nuns used by the method depending on the name
        if method_params.get("independent_channels_nun", False):
            nuns = possible_nuns["iknn"]
        else:
            nuns = possible_nuns["gknn"]

        # Calculate metrics
        method_name = methods[method_key]
        method_metrics = calculate_method_metrics(model, outlier_calculators, data_format,
                                                  X_test[method_test_indexes], nuns[method_test_indexes], method_cfs,
                                                  original_classes[method_test_indexes], method_name,
                                                  mo_weights=mo_weights, order=i + 1)
        method_metrics.insert(0, "ii", method_test_indexes)
        results_df = pd.concat([results_df, method_metrics])
        method_cfs_dataset[method_name] = method_cfs
        common_test_indexes = list(set(method_test_indexes).intersection(common_test_indexes))
        common_test_indexes.sort()

    # Calculate results table for the dataset
    means_df = results_df.groupby('method').mean()
    means_df = means_df.sort_values('order').drop('order', axis=1)
    stds_df = results_df.groupby('method').std()
    stds_df = stds_df.drop('order', axis=1)
    stds_df = stds_df.reindex(means_df.index)
    mean_std_df = means_df.round(2).astype(str) + " ± " + stds_df.round(2).astype(str)
    mean_std_df = mean_std_df.reset_index()
    results_df['dataset'] = dataset

    return mean_std_df, results_df, method_cfs_dataset, common_test_indexes


def get_method_objectives(model, outlier_calculator, X_test, nuns, solutions_in, original_classes):
    # Get the results and separate them in counterfactuals and execution times
    solutions = copy.deepcopy(solutions_in)
    # Check if the solutions are single or multiple solutions
    if 'cfs' in solutions[0]:
        counterfactuals = [solution['cfs'] for solution in solutions]
    else:
        counterfactuals = [solution['cf'] for solution in solutions]
    execution_times = [solution['time'] for solution in solutions]

    # Get size of the input
    length = X_test.shape[1]
    n_channels = X_test.shape[2]

    all_objectives_list = []
    for i in tqdm(range(len(X_test))):
        x_orig_i = X_test[i]
        counterfactuals_i = counterfactuals[i]

        # Calculate valids
        predicted_logits = model.predict(counterfactuals_i, verbose=0)
        predicted_classes = np.argmax(predicted_logits, axis=1)
        valids = (predicted_classes != original_classes[i]).astype(int)

        # Filter counterfactuals based on valids
        valid_idx = np.where(valids == 1)[0]
        if len(valid_idx) == 0:
            # Calculate objectives dict
            sample_objectives_dict = {
                "valids": [0],
                "desired_class_prob": [np.nan],
                "sparsity": [np.nan],
                "subsequences": [np.nan],
                "IoS": [np.nan],
                "execution_time": execution_times[i]
            }

        else:
            if len(valid_idx) < len(counterfactuals_i):
                raise ValueError("Not all cfs in front all valid")

            else:
                valid_counterfactuals_i = counterfactuals_i[valid_idx]
                n_counterfactuals_i = valid_counterfactuals_i.shape[0]

                # Calculate desired class based on NUN
                if nuns[i] is not None:
                    desired_class = np.argmax(model.predict(np.expand_dims(nuns[i], axis=0), verbose=0), axis=1)[0]
                else:
                    predicted_cf_classes = np.argmax(model.predict(valid_counterfactuals_i, verbose=0), axis=1)
                    vals, counts = np.unique(predicted_cf_classes, return_counts=True)
                    index = np.argmax(counts)
                    desired_class = vals[index]

                # Calculate predicted probabilities
                desired_predicted_probs = predicted_logits[:, desired_class]

                # Get change mask (L0)
                percentual_proximity = np.abs(x_orig_i - valid_counterfactuals_i) / np.abs(x_orig_i + 1e-6)
                change_masks = (percentual_proximity > 0.001).astype(int)
                sparsity = change_masks.sum(axis=(1, 2)) / (length * n_channels)

                # Subsequences
                subsequences = np.count_nonzero(np.diff(change_masks, prepend=0, axis=1) == 1, axis=(1, 2))
                feature_avg_subsequences = subsequences / n_channels
                subsequences_pct = feature_avg_subsequences / (length // 2)

                # Calculate outlier scores
                if outlier_calculator is not None:
                    aux_outlier_scores = outlier_calculator.get_outlier_scores(valid_counterfactuals_i)
                    aux_increase_outlier_score = aux_outlier_scores - outlier_calculator.get_outlier_scores(x_orig_i)[0]
                else:
                    aux_increase_outlier_score = np.zeros((n_counterfactuals_i, 1))
                aux_increase_outlier_score[aux_increase_outlier_score < 0] = 0

                # Calculate objectives dict
                sample_objectives_dict = {
                    "valids": valids,
                    "desired_class_prob": desired_predicted_probs,
                    "sparsity": sparsity,
                    "subsequences": subsequences_pct,
                    "IoS": aux_increase_outlier_score,
                    "execution_time": execution_times[i]
                }
        # Append to list
        all_objectives_list.append(sample_objectives_dict)
    return all_objectives_list


def obtain_cfs_objectives(dataset, methods, model_to_explain,
                          data_tuple, original_classes, model, outlier_calculator, possible_nuns):

    X_train, y_train, X_test, y_test = data_tuple

    cf_solution_dirs = [fname for fname in os.listdir(f'./experiments/results/{dataset}/{model_to_explain}') if os.path.isdir(f'./experiments/results/{dataset}/{model_to_explain}/{fname}')]
    desired_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in cf_solution_dirs if cf_sol_dir in methods.keys()]
    valid_cf_solution_dirs = [cf_sol_dir for cf_sol_dir in desired_cf_solution_dirs if os.path.isfile(f'./experiments/results/{dataset}/{model_to_explain}/{cf_sol_dir}/counterfactuals.pickle')]
    method_cfs_dataset_dict = {}
    method_objectives_dataset_dict = {}
    common_test_indexes = list(range(len(X_test)))
    for i, method_dir_name in enumerate(valid_cf_solution_dirs):
        # Load solution cfs
        with open(f'./experiments/results/{dataset}/{model_to_explain}/{method_dir_name}/counterfactuals.pickle', 'rb') as f:
            print(method_dir_name)
            method_cfs = pickle.load(f)
        # Load params
        with open(f'./experiments/results/{dataset}/{model_to_explain}/{method_dir_name}/params.json', 'r') as json_file:
            method_params = json.load(json_file)
            method_test_indexes = method_params["X_test_indexes"]

        # Get nuns used by the method depending on the name
        if "independent_channels_nun" in method_params:
            if method_params["independent_channels_nun"]:
                nuns = possible_nuns["iknn"]
            else:
                nuns = possible_nuns["gknn"]
        else:
            nuns = np.array([None]*len(X_test))

        # Calculate metrics
        method_name = methods[method_dir_name]
        method_objectives = get_method_objectives(
            model, outlier_calculator,
            X_test[method_test_indexes], nuns[method_test_indexes], method_cfs,
            original_classes[method_test_indexes]
        )
        method_objectives_dataset_dict[method_name] = method_objectives
        method_cfs_dataset_dict[method_name] = method_cfs

    return method_objectives_dataset_dict, method_cfs_dataset_dict, common_test_indexes


def calculate_method_valids(model, X_test, counterfactuals, original_classes):
    # Get size of the input
    length = X_test.shape[1]
    n_channels = X_test.shape[2]

    # Loop over counterfactuals
    valids = []
    for i in tqdm(range(len(X_test))):
        # Predict counterfactual class probability
        preds = model.predict(counterfactuals[i].reshape(-1, length, n_channels), verbose=0)
        pred_class = np.argmax(preds, axis=1)[0]

        # Valids
        if pred_class != original_classes[i]:
            valids.append(True)
        else:
            valids.append(False)

    return valids


def calculate_method_metrics(model_wrapper, outlier_calculators, data_format, X_test, nuns, solutions_in, original_classes,
                             method_name, mo_weights=None, order=None):
    # Get the results and separate them in counterfactuals and execution times
    solutions = copy.deepcopy(solutions_in)
    # Check if the solutions are single or multiple solutions
    if 'cfs' in solutions[0]:
        # If there are no mo_weights then there is no way to compare the counterfactuals
        if mo_weights is None:
            raise ValueError("There are multiple counterfactuals for a single input instance. "
                             "Weights for objectives must be passed to order the counterfactuals using a "
                             "specific utility function")
        counterfactuals = [solution['cfs'] for solution in solutions]
    else:
        counterfactuals = [solution['cf'] for solution in solutions]

    # Get time
    if "train_time" in solutions[0]:
        train_time = solutions[0]['train_time']
    else:
        train_time = 0
    execution_times = [solution['time'] for solution in solutions]

    if len(counterfactuals[0].shape) == 2:
        counterfactuals = [np.expand_dims(cf, axis=0) for cf in counterfactuals]

    # Get size of the input
    if data_format == "torch":
        counterfactuals = [cf.transpose(0, 2, 1) for cf in counterfactuals]

    length = X_test.shape[1]
    n_channels = X_test.shape[2]

    # Loop over counterfactuals
    nchanges = []
    l1s = []
    l2s = []
    pred_probas = []
    valids = []
    outlier_scores_dict = {}
    increase_outlier_scores_dict = {}
    n_subsequences = []
    best_cf_is = []

    outlier_scores_orig = outlier_calculators["AE"].get_outlier_scores(X_test)
    if nuns[0] is not None:
        nuns_os = outlier_calculators["AE"].get_outlier_scores(nuns)
        nun_ios = nuns_os - outlier_scores_orig
        nun_ios[nun_ios < 0] = 0
        nun_predicted_probs = model_wrapper.predict(nuns)
        desired_classes = np.argmax(nun_predicted_probs, axis=1)
        desired_classes_probs = nun_predicted_probs[np.arange(nun_predicted_probs.shape[0]), desired_classes]
        sparsity = 1
        subsequences_pct = n_channels / (n_channels * ( length // 2))
        nuns_fitness = (mo_weights[0] * desired_classes_probs - mo_weights[1] * sparsity -
                        mo_weights[2] * (subsequences_pct ** 0.25) - mo_weights[3] * nun_ios)
    else:
        desired_classes = None
    for i in tqdm(range(len(X_test))):
        counterfactuals_i = counterfactuals[i]
        x_orig_i = X_test[i]
        # If there are multiple counterfactuals apply mo_weights
        if counterfactuals_i.shape[0] > 1:
            desired_class = desired_classes[i]
            # Sort by objective weights and take the best
            predicted_probs = model_wrapper.predict(counterfactuals_i)
            # Get outlier scores from AE to get the best CF
            if outlier_calculators is not None:
                aux_outlier_scores = outlier_calculators["AE"].get_outlier_scores(counterfactuals_i)
            else:
                aux_outlier_scores = np.zeros((predicted_probs.shape[0], 1))
            # Get fitness scores
            change_masks = (counterfactuals_i != x_orig_i).astype(int)
            objective_fitness = fitness_function_mo(change_masks, predicted_probs, desired_class, aux_outlier_scores,
                                                    outlier_calculators["AE"].get_outlier_scores(x_orig_i)[0], 100)
            fitness = (objective_fitness * mo_weights).sum(axis=1)
            best_cf_i = np.argsort(fitness)[-1]
            counterfactual_i = counterfactuals_i[best_cf_i].reshape(length, n_channels)
            best_cf_is.append(best_cf_i)
        else:
            best_cf_is.append(0)
            counterfactual_i = counterfactuals_i[0].reshape(length, n_channels)

        # Predict counterfactual class probability
        preds = model_wrapper.predict(counterfactual_i)
        pred_class = np.argmax(preds, axis=1)[0]

        # Valids
        if (pred_class != original_classes[i]) and (~np.isnan(counterfactual_i).any()):
            valids.append(True)

            # Add class probability
            pred_proba = preds[0, pred_class]
            pred_probas.append(pred_proba)

            # Calculate l0
            # change_mask = (X_test[i] != counterfactuals[i]).astype(int)
            # print(X_test[i].shape, X_train[nuns_idx[i]].shape, counterfactuals[i].shape)
            change_mask = calculate_change_mask(x_orig_i, counterfactual_i, x_nun=nuns[i], verbose=0)
            nchanges.append(change_mask.sum())

            # Calculate l1
            l1 = np.linalg.norm((x_orig_i.flatten() - counterfactual_i.flatten()), ord=1)
            l1s.append(l1)

            # Calculate l2
            l2 = np.linalg.norm((x_orig_i.flatten() - counterfactual_i.flatten()), ord=2)
            l2s.append(l2)

            # Calculate outlier scores
            for oc_name, outlier_calculator in outlier_calculators.items():
                outlier_score_orig = outlier_calculator.get_outlier_scores(x_orig_i)[0]
                outlier_score = outlier_calculator.get_outlier_scores(counterfactual_i)[0]
                increase_outlier_score = outlier_score - outlier_score_orig
                if oc_name in outlier_scores_dict:
                    outlier_scores_dict[oc_name].append(outlier_score)
                else:
                    outlier_scores_dict[oc_name] = [outlier_score]
                if oc_name in increase_outlier_scores_dict:
                    increase_outlier_scores_dict[oc_name].append(increase_outlier_score)
                else:
                    increase_outlier_scores_dict[oc_name] = [increase_outlier_score]

            # Number of sub-sequences
            # print(change_mask.shape)
            subsequences = np.count_nonzero(np.diff(change_mask, prepend=0, axis=0) == 1, axis=(0,1))
            n_subsequences.append(subsequences)
        else:
            valids.append(False)
            # Append all NaNs to not being take into consideration
            pred_probas.append(np.nan)
            nchanges.append(np.nan)
            l1s.append(np.nan)
            l2s.append(np.nan)
            n_subsequences.append(np.nan)
            for oc_name, outlier_calculator in outlier_calculators.items():
                if oc_name in outlier_scores_dict:
                    outlier_scores_dict[oc_name].append(np.nan)
                else:
                    outlier_scores_dict[oc_name] = [np.nan]
                if oc_name in increase_outlier_scores_dict:
                    increase_outlier_scores_dict[oc_name].append(np.nan)
                else:
                    increase_outlier_scores_dict[oc_name] = [np.nan]

    # Valid NUN classes
    if nuns[0] is None:
        valid_nuns = [np.nan]*len(nuns)
    else:
        nun_preds = model_wrapper.predict(nuns)
        nun_pred_class = np.argmax(nun_preds, axis=1)
        valid_nuns = nun_pred_class != original_classes

    # Create dataframe
    results = pd.DataFrame()
    results["nchanges"] = nchanges
    results["sparsity"] = results["nchanges"] / (length * n_channels)
    results["L1"] = l1s
    results["L2"] = l2s
    results["proba"] = pred_probas
    results["valid"] = valids
    results["nuns_valid"] = valid_nuns
    # Create column for Outlier Scores for every calculator
    for oc_name, outlier_scores in outlier_scores_dict.items():
        outlier_scores = np.array(outlier_scores)
        results[f"{oc_name}_OS"] = outlier_scores
    for oc_name, increase_outlier_scores in increase_outlier_scores_dict.items():
        increase_os = np.array(increase_outlier_scores)
        increase_os[increase_os < 0] = 0
        results[f"{oc_name}_IOS"] = increase_os
    results['subsequences'] = n_subsequences
    results['subsequences %'] = np.array(n_subsequences) / (n_channels * (length // 2))
    results['train_time'] = train_time
    results['times'] = execution_times
    results['method'] = method_name
    results['best cf index'] = best_cf_is
    if order is not None:
        results['order'] = order
    fitness = (mo_weights[0]*results["proba"] - mo_weights[1]*results['sparsity'] -
               mo_weights[2]*(results['subsequences %'] ** 0.25) - mo_weights[3] * results[f"AE_IOS"])
    results['fitness'] = fitness
    results['nun_fitness'] = nuns_fitness
    results["ImprovementOverNUN"] = fitness - nuns_fitness
    return results
