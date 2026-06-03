import pickle
import os

# DATASET = 'UWaveGestureLibrary'
DATASET = 'ECG200'
# PREFIX_FILES_NAME = "fitness_evolutions_v6_mut2"
PREFIX_FILES_NAME = "subspace"


def infer_fragmentation_samples(dataset, model_to_explain_name, prefix_files_name, path='.'):
    files = [filename for filename in os.listdir(f'{path}/{dataset}/{model_to_explain_name}/{prefix_files_name}') if filename.startswith(prefix_files_name)]
    last_file_name = files[-1]
    last_file_name = last_file_name.replace(prefix_files_name, "")
    last_file_name = last_file_name.replace(".pickle", "")
    last_file_name = last_file_name.replace("_", "")
    last_file_samples_range = last_file_name.split("-")
    start_index = int(last_file_samples_range[0])
    end_index = int(last_file_samples_range[1])

    fragmentation = end_index - start_index + 1
    total = end_index + 1

    return fragmentation, total


def concatenate_and_store_partial_results(dataset, model_to_explain_name, prefix_files_name, suffixes_list, path='.'):
    all_files_list = []
    for suffix in suffixes_list:
        with open(f'{path}/{dataset}/{model_to_explain_name}/{prefix_files_name}/{prefix_files_name}_{suffix}.pickle', 'rb') as f:
            part_file = pickle.load(f)

        all_files_list = all_files_list + part_file

    # Store concatenated file
    with open(f'{path}/{dataset}/{model_to_explain_name}/{prefix_files_name}/counterfactuals.pickle', 'wb') as f:
        pickle.dump(all_files_list, f, pickle.HIGHEST_PROTOCOL)


def remove_partial_files(dataset, model_to_explain_name, prefix_files_name, path='.'):
    files = [filename for filename in os.listdir(f'{path}/{dataset}/{model_to_explain_name}/{prefix_files_name}') if filename.startswith(prefix_files_name)]
    partial_files = [filename for filename in files if '-' in filename]
    for partial_file in partial_files:
        os.remove(f'{path}/{dataset}/{model_to_explain_name}/{prefix_files_name}/{partial_file}')


def concatenate_result_files(dataset, model_to_explain_name, prefix_file_name):
    # Calculate suffixes to concatenate
    fragmentation_samples, total_samples = infer_fragmentation_samples(
        dataset,
        model_to_explain_name,
        prefix_file_name,
        path='./experiments/results'
    )
    suffixes_list = [f"{i:04d}-{i + fragmentation_samples - 1:04d}" for i in
                     range(0, total_samples, fragmentation_samples)]
    concatenate_and_store_partial_results(
        dataset,
        model_to_explain_name,
        prefix_file_name,
        suffixes_list,
        path='./experiments/results'
    )
    # Remove the temporal files
    remove_partial_files(
        dataset,
        model_to_explain_name,
        prefix_file_name,
        path='./experiments/results'
    )


if __name__ == "__main__":
    concatenate_result_files(DATASET, PREFIX_FILES_NAME)