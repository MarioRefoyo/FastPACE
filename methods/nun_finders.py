import copy
import itertools
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeries


class NUNFinder(ABC):
    def __init__(self, X_train, y_train, y_pred, distance, from_true_labels, data_format, n_neighbors):
        if data_format == 'tf':
            self.data_format = data_format
            self.X_train = X_train
        elif data_format == "torch":
            self.data_format = data_format
            self.X_train = X_train.transpose(0, 2, 1)
        else:
            raise ValueError('Backend not supported')

        self.ts_length = self.X_train.shape[1]
        self.n_channels = self.X_train.shape[2]
        self.data_shape = (self.ts_length, self.n_channels)

        self.y_train = y_train
        self.y_pred = y_pred
        self.distance = distance
        self.n_neighbors = n_neighbors
        self.feature_axis = 2

        # Get df with index from true training and predicted labels
        df_init = pd.DataFrame(y_train, columns=['true_label'])
        df_init["pred_label"] = y_pred
        df_init.index.name = 'index'
        if from_true_labels:
            label_name = 'true_label'
        else:
            label_name = 'pred_label'
        self.label_name = label_name
        self.df_index = df_init[[label_name]]

        # Train knn for get the NUNs of every possible output class
        classes = np.unique(y_train)
        diff_class_index_dict = {}
        classes_knn_dict = {}
        for c in classes:
            knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)

            diff_class_index = self.df_index[self.df_index[label_name] != c]
            knn.fit(self.X_train[diff_class_index.index.values])

            diff_class_index_dict[c] = diff_class_index
            classes_knn_dict[c] = knn
        self.diff_class_index_dict = diff_class_index_dict
        self.classes_knn_dict = classes_knn_dict

    def get_nns_indexes(self, knn, x_orig, knn_training_index_df):
        if self.data_format == "torch":
            x_orig = x_orig.transpose(1, 0)
        # Get the closests neighbors on the knn training data
        dist, ind = knn.kneighbors(
            np.expand_dims(x_orig, axis=0), return_distance=True
        )
        # Get only the first sample of the batch (batch_size=1 since we are only working with x_orig)
        dist, ind = dist[0], ind[0]
        # Transform the index to the original index in the complete X_train
        index = knn_training_index_df.index[ind]
        # Get label of the closest neighbor (even if nn is greater than 1)
        label = self.df_index[self.df_index.index.isin(index.tolist())].values[0][0]

        return index, label, dist

    @abstractmethod
    def retrieve_single_nun_specific(self, x_orig, original_label):
        pass

    def retrieve_nuns(self, x_origs, original_labels):
        # Check for shape errors
        if not len(x_origs.shape) == 3:
            if not ((x_origs.shape[0] == self.X_train.shape[1]) & (x_origs.shape[1] == self.X_train.shape[2])):
                raise ValueError(f"The input must have shape of (n_instances, {self.X_train.shape[1]}, {self.X_train.shape[2]}), "
                                 f"but input has shape of {x_origs.shape}")
            else:
                # Expand dimension and treat input as a single dataset sample
                x_origs = copy.deepcopy(x_origs)
                x_origs = np.expand_dims(x_origs, axis=0)

        # Find all the NUNs
        nuns = []
        nun_labels = []
        distances = []
        for instance_idx in range(len(x_origs)):
            nun, nun_label, distance = self.retrieve_single_nun_specific(x_origs[instance_idx], original_labels[instance_idx])
            nuns.append(nun)
            nun_labels.append(nun_label)
            distances.append(distance)

        if self.data_format == "tf":
            nuns = np.array(nuns)
        elif self.data_format == "torch":
            nuns = np.array(nuns)
            nuns = nuns.transpose(0, 1, 3, 2)

        return nuns, np.array(nun_labels), np.array(distances)


class GlobalNUNFinder(NUNFinder):
    def __init__(self, X_train, y_train, y_pred, distance, from_true_labels, data_format):
        # Force 1 n_neighbors
        n_neighbors = 1
        super().__init__(X_train, y_train, y_pred, distance, from_true_labels, data_format, n_neighbors)

    def retrieve_single_nun_specific(self, x_orig, original_label):
        global_nun_indexes, nn_label, global_dists = self.get_nns_indexes(
            self.classes_knn_dict[original_label], x_orig, self.diff_class_index_dict[original_label]
        )
        # global_nun_index, global_dist = global_nun_indexes[0], global_dists[0]

        # Retrieve NUN
        nun = self.X_train[global_nun_indexes]
        dist = global_dists

        return nun, nn_label, dist


class IndependentNUNFinder(NUNFinder):
    def __init__(self, X_train, y_train, y_pred, distance, from_true_labels, backend, n_neighbors, model):
        super().__init__(X_train, y_train, y_pred, distance, from_true_labels, backend, n_neighbors)
        self.model = model
        self.exhaustive = False

        # Train a knn per channel and class if independent channels flag is active
        classes = np.unique(y_train)

        same_class_index_dict = {}
        classes_feature_knn_dict = {}
        for c in classes:
            feature_knn_dict = {}
            for feature in range(self.n_channels):
                knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)

                same_class_index = self.df_index[self.df_index[self.label_name] == c]
                if len(same_class_index) > 0:
                    knn.fit(self.X_train[same_class_index.index.values, :, feature])

                    same_class_index_dict[c] = same_class_index
                    feature_knn_dict[feature] = knn
            classes_feature_knn_dict[c] = feature_knn_dict
        self.same_class_index_dict = same_class_index_dict
        self.classes_feature_knn_dict = classes_feature_knn_dict

        # Pre-load possible combinations of channels
        if self.exhaustive:
            # Generate all possible combinations
            range_channel_indexes = [list(range(self.n_neighbors)) for _ in range(self.n_channels)]
            combinations = list(itertools.product(*range_channel_indexes))
            combinations = np.array(combinations)
        else:
            n_samples = 10000
            # Sample randomly from possibilities
            combinations = np.random.choice(self.n_neighbors, (n_samples, self.n_channels))
            combinations = np.unique(combinations, axis=0)
        self.combinations = combinations

    def generate_nuns_from_indexes(self, combinations):
        nuns = np.zeros((len(combinations),) + self.data_shape)

        for feature in range(self.n_channels):
            feature_indexes = combinations[:, feature]
            nuns[:, :, feature] = self.X_train[feature_indexes, :, feature]
        return nuns

    def retrieve_single_nun_specific(self, x_orig, original_label):
        global_nun_indexes, nn_label, global_dists = self.get_nns_indexes(
            self.classes_knn_dict[original_label], x_orig, self.diff_class_index_dict[original_label]
        )

        # Retrieve NUN
        if self.n_neighbors > 1:
            # ToDo: Generate NUN from multiple neighbors
            channel_indexes = []
            for feature in range(self.n_channels):
                feature_nun_idxes, feature_label, _ = self.get_nns_indexes(
                    self.classes_feature_knn_dict[nn_label][feature],
                    x_orig[:, feature], self.same_class_index_dict[nn_label]
                )
                channel_indexes.append(feature_nun_idxes)

            # Replace range indexes to true indexes
            traduced_combinations = copy.deepcopy(self.combinations)
            for feature in range(self.n_channels):
                for i_nn in range(self.n_neighbors):
                    traduced_combinations[traduced_combinations[:, feature] == i_nn, feature] = channel_indexes[feature][i_nn]

            possible_nuns = self.generate_nuns_from_indexes(traduced_combinations)
            # Get validity and distances of the permutations
            nun_logits = self.model.predict(possible_nuns, verbose=0)
            nun_classes = np.argmax(nun_logits, axis=1)
            valids = nun_classes == nn_label
            nun_distances = np.linalg.norm(x_orig - possible_nuns, axis=(1, 2))

            # Order by proximity and get the nun
            possible_nuns_df = pd.DataFrame()
            possible_nuns_df["perm_id"] = list(range(len(traduced_combinations)))
            possible_nuns_df["y_pred"] = nun_classes
            possible_nuns_df["valid"] = valids
            possible_nuns_df["distance"] = nun_distances
            possible_nuns_df = possible_nuns_df.sort_values(by="distance")
            possible_nuns_df = possible_nuns_df[possible_nuns_df["valid"] == True]
            if len(possible_nuns_df) == 0:
                print("NUN could not be found. Returning the global NUN.")
                nuns = self.X_train[global_nun_indexes]
                return nuns, nn_label, global_dists

            # First n_neighbors
            perm_indexes = possible_nuns_df["perm_id"].values[:self.n_neighbors]
            nuns = possible_nuns[perm_indexes]
            dist = nun_distances[perm_indexes]

        else:
            nuns = np.zeros((self.n_neighbors,) + self.data_shape)
            feature_labels = []
            for feature in range(self.n_channels):
                feature_nun_idxes, feature_label, _ = self.get_nns_indexes(
                    self.classes_feature_knn_dict[nn_label][feature],
                    x_orig[:, feature], self.same_class_index_dict[nn_label]
                )
                feature_nun_idx = feature_nun_idxes[0]
                nuns[:, :, feature] = self.X_train[feature_nun_idx, :, feature]
                feature_labels.append(feature_label)
            dist = np.linalg.norm(x_orig - nuns)
            if len(set(feature_labels)) != 1:
                print(f"Channel labels correspond to instances of classes: {feature_labels}. Desired class is {nn_label}")

        return nuns, nn_label, dist

