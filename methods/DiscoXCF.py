import copy
import pandas as pd
import numpy as np

from .COMTE.Optimization import BruteForceSearch, OptimizedSearch
from .counterfactual_common import CounterfactualMethod
from .DiscoX.mutils import cal_tags_mps, fast_find_anomalies, idx_to_str, get_score, get_discord, get_custom_data


class DiscoXCF(CounterfactualMethod):
    def __init__(self, model_wrapper, X_train, y_train, window_pct=0.1):
        super().__init__(model_wrapper)
        self.X_train = X_train
        self.y_train = y_train
        self.window = int(window_pct * X_train.shape[1])
        self.steps = np.concatenate([np.arange(0.01, 0.5, 0.01), np.arange(0.5, 1.01, 0.02)])

    @staticmethod
    def predict(X, model_wrapper):
        y_pred = model_wrapper.predict(X)
        return np.argmax(y_pred, axis=1)

    @staticmethod
    def ind_list_to_intervals(lst):
        # split non-mapped indices into separate intervals
        intervals = []
        intervals_inds = []
        curr_interval = []
        curr_interval_inds = []

        prev_i = lst[0]
        curr_interval.append(prev_i)
        curr_interval_inds.append(0)

        for ind, i in enumerate(lst[1:]):
            if i == prev_i + 1:
                curr_interval.append(i)
                curr_interval_inds.append(ind + 1)
            else:
                intervals.append(curr_interval)
                intervals_inds.append(curr_interval_inds)
                curr_interval = [i]
                curr_interval_inds = [ind + 1]
            prev_i = i

        intervals.append(curr_interval)
        intervals_inds.append(curr_interval_inds)

        return intervals, intervals_inds

    # wrapper for multidimensional time series
    def mfill_short_intervals(self, mapping, target_ts, X_target_c_sep, d_dims, thres=3):
        new_target_ts = target_ts.copy()

        for dim in d_dims:
            target_ts_d = target_ts[:, dim]
            X_target_c_sep_d = X_target_c_sep[:, dim]

            new_mapping, new_target_ts[:, dim] = \
                self.fill_short_intervals(mapping, target_ts_d, X_target_c_sep_d, thres=thres)

        return new_mapping, new_target_ts

    # fill short non-mapped subsequences by extending their longest adjacent mapped subsequence
    def fill_short_intervals(self, mapping, target_ts, X_target_c_sep, thres=3):
        # non-mapped indices
        non_map = np.where(np.isnan(mapping))[0]
        if len(non_map) > 0:
            non_map_intervals, _ = self.ind_list_to_intervals(non_map)

            # find position of current non-mapped subsequence and idx of subsequence to extend
            for interval in non_map_intervals:
                mapped_intervals, mapped_intervals_inds = self.ind_list_to_intervals(mapping)
                if len(interval) < thres:
                    for mii_i, mii in enumerate(mapped_intervals_inds):
                        if interval[0] == mii[0]:
                            if mii_i == 0:
                                # the current non-mapped subsequence is at the beginning of TS
                                fill_from = mii_i + len(interval)
                                backup_fill_from = -1
                                plus = 1
                            elif mii_i + len(interval) == len(mapped_intervals_inds):
                                # the current non-mapped subsequence is at the end of TS
                                fill_from = mii_i - 1
                                backup_fill_from = -1
                                plus = -1
                            else:
                                # the current non-mapped subsequence is in the middle of TS
                                # so, find the lengths of adjacent subsequences
                                if len(mapped_intervals_inds[mii_i - 1]) > \
                                        len(mapped_intervals_inds[mii_i + len(interval)]):
                                    fill_from = mii_i - 1
                                    backup_fill_from = mii_i + 1
                                    plus = -1
                                else:
                                    fill_from = mii_i + len(interval)
                                    backup_fill_from = mii_i - 1
                                    plus = 1
                            break

                    # fill non-mapped subsequence
                    # first check if there are no nans in it (from instances borders)
                    if plus == 1:
                        temp_mapped = np.array(mapping[mapped_intervals_inds[fill_from]])[-len(interval):] - \
                                      len(mapped_intervals_inds[fill_from])
                    else:
                        temp_mapped = np.array(mapping[mapped_intervals_inds[fill_from]])[:len(interval)] + \
                                      len(mapped_intervals_inds[fill_from])

                    # if trying to expand beyond length of X_target_c_sep (and
                    # the current non-mapped subsequence is at the end of TS), don't
                    # will interpolate later in main code
                    beyond = np.where(temp_mapped >= len(X_target_c_sep))[0]

                    if len(beyond):
                        if backup_fill_from == -1:
                            temp_mapped = temp_mapped[:min(beyond)]
                            temp_target_ts = X_target_c_sep[list(map(int, temp_mapped))]

                            mapping[interval][:min(beyond)] = temp_mapped
                            target_ts[interval][:min(beyond)] = temp_target_ts

                            return mapping, target_ts
                        else:
                            fill_from == backup_fill_from
                            backup_fill_from = -1

                    # if trying to expand before idx 0 of X_target_c_sep (and
                    # the current non-mapped subsequence is at the beginning of TS), don't
                    # will interpolate later in main code
                    beyond = np.where(temp_mapped < 0)[0]

                    if len(beyond):
                        if backup_fill_from == -1:
                            temp_mapped = temp_mapped[min(beyond):]
                            temp_target_ts = X_target_c_sep[list(map(int, temp_mapped))]

                            mapping[interval][min(beyond):] = temp_mapped
                            target_ts[interval][min(beyond):] = temp_target_ts

                            return mapping, target_ts
                        else:
                            fill_from == backup_fill_from
                            backup_fill_from = -1

                    temp_target_ts = X_target_c_sep[list(map(int, temp_mapped))]

                    nan_temp = np.where(np.isnan(temp_target_ts))[0]

                    # if there are nans, replace starting from first nan with backup
                    if len(nan_temp) and backup_fill_from != -1:
                        print('nans in fill from')
                        if plus == 1:
                            backup_mapped = np.array(mapping[mapped_intervals_inds[backup_fill_from]])[:len(interval)] + \
                                            len(mapped_intervals_inds[backup_fill_from])
                        else:
                            backup_mapped = np.array(mapping[mapped_intervals_inds[backup_fill_from]])[
                                            -len(interval):] - \
                                            len(mapped_intervals_inds[backup_fill_from])

                        temp_mapped[nan_temp[0]:] = backup_mapped[nan_temp[0]:]

                    nan_temp = np.where(np.isnan(temp_target_ts))[0]

                    # if there are still nans, leave them and interpolate
                    if len(nan_temp):
                        print('nans in backup fill from')
                        # inds in between nans should be interpolated too
                        temp_target_ts[min(nan_temp):max(nan_temp) + 1] = np.nan
                        temp_mapped[min(nan_temp):max(nan_temp) + 1] = np.inf

                    mapping[interval] = temp_mapped
                    target_ts[interval] = temp_target_ts

        return mapping, target_ts

    # wrapper for multidimensional time series
    def minterpolate_short(self, X_cf, mapping, d_dims, thres=3):
        new_cf = X_cf.copy()

        for dim in d_dims:
            cf_d = X_cf[:, dim]
            new_cf[:, dim] = self.interpolate_short(cf_d, mapping, thres=thres)

        return new_cf

    # interpolate intervals shorter than tresh
    def interpolate_short(self, X_cf, mapping, thres=3):
        # non-mapped indices
        mapping[mapping == np.inf] = np.nan
        non_map = np.where(np.isnan(mapping))[0]
        if len(non_map) > 0:
            non_map_intervals, _ = self.ind_list_to_intervals(non_map)

            for interval in non_map_intervals:
                if len(interval) < thres:
                    X_cf[interval] = np.nan

            # interpolate short nan intervals
            X_cf = pd.DataFrame(X_cf)
            mask = X_cf.copy()
            grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
            grp['ones'] = 1
            for i in [0]:
                mask[i] = (grp.groupby(i)['ones'].transform('count') < thres) | X_cf[i].notnull()
            X_cf = X_cf.interpolate().bfill()[mask]
            X_cf = X_cf.to_numpy().reshape(-1)

        return X_cf

    def generate_counterfactual_specific(self, x_orig, desired_target=None, nun_example=None):
        x_cf_list = []

        orig_y = self.predict(x_orig, self.model_wrapper)[0]
        cf_found = False
        for target_c in np.unique(self.y_train):
            if cf_found:
                break

            # print('target_c, orig_y', target_c, orig_y)

            if target_c == orig_y:
                print('same, continue')
                continue

            target_c_inds = np.where(self.y_train == target_c)[0]
            X_target_c = self.X_train[target_c_inds]
            X_target_c_sep = np.concatenate((X_target_c[0],
                                             np.full((1, x_orig.shape[1]), fill_value=np.nan)))
            for X_ in X_target_c[1:]:
                X_target_c_sep = np.concatenate((X_target_c_sep, X_))
                X_target_c_sep = np.concatenate((X_target_c_sep,
                                                 np.full((1, x_orig.shape[1]), fill_value=np.nan)))

            X_target_c_sep = X_target_c_sep[:-1, :]

            mps, _ = cal_tags_mps(x_orig, win=self.window, query=X_target_c_sep)

            kdps, kdps_idx = fast_find_anomalies(mps)

            str_name = idx_to_str(kdps_idx)
            scoring_method = 'mean'
            scores = get_score(kdps, scoring_method)

            all_scores = []
            all_sets = []

            for k in range(kdps_idx.shape[1]):
                customdata = get_custom_data(scores, str_name, k)

                scores_ = np.array([c[0][0] for c in customdata])
                all_scores.append(scores_)

                sets_ = np.array([c[1] for c in customdata])
                all_sets.append(sets_)

            all_scores, all_sets = np.array(all_scores), np.array(all_sets)

            mapped = np.full(self.X_train.shape[1], fill_value=np.nan)
            when_mapped = np.full(self.X_train.shape[1], fill_value=np.nan)
            target_ts = x_orig.copy()

            when = 1

            # the number of non-mapped intervals
            n_non_mapped = np.inf
            # the length of the longest non-mapped intervals
            max_len_non_mapped = np.inf

            # until all time steps are mapped or only intervals shorter than 3 time steps remain unmapped
            while n_non_mapped and max_len_non_mapped >= self.window and not cf_found:
                # if all indices have been mapped
                if not np.any(np.isnan(mapped)):
                    break

                if cf_found:
                    break

                d_X_start, d_dims, d_nun_start, all_scores = get_discord(x_orig, self.window,
                                                                         X_target_c_sep, all_scores, kdps_idx, all_sets)

                d_X_end = d_X_start + self.window
                d_nun_end = d_nun_start + self.window

                # mapped indices
                already_mapped = np.delete(np.arange(len(mapped)),
                                           np.where(np.isnan(mapped))[0])

                inter_ids = np.in1d(np.arange(d_X_start, d_X_end),
                                    already_mapped)
                X_inter = np.arange(d_X_start, d_X_end)[inter_ids]

                # if any index of the current subsequence has been mapped (but not all)
                if len(X_inter):
                    if len(X_inter) == self.window:
                        continue

                    # if only intersects one already mapped subsequence
                    if len(self.ind_list_to_intervals(X_inter)[0]) == 1:
                        # to the right of already mapped subsequence
                        if X_inter[-1] + 1 in np.arange(d_X_start, d_X_end):
                            # last mapped index (to start expanding from)
                            exp_from = X_inter[-1]

                            # remaining nb of time steps until end of target_c_sep
                            to_tcs_end = len(X_target_c_sep) - mapped[exp_from]
                            # remaining nb of time steps until end of current instance
                            # (until NaN index)
                            to_curr_end = len(x_orig) - 1 - mapped[exp_from] % (len(x_orig) + 1)

                            to_curr_end = min(to_curr_end, to_tcs_end)

                            # expand as far as possible (stopping at end of current subsequence)
                            exp_len = int(min(to_curr_end, self.window - len(X_inter)))

                            mapped[exp_from + 1:exp_from + 1 + exp_len] = np.arange(
                                mapped[exp_from] + 1, mapped[exp_from] + 1 + exp_len)

                            nun_subs = X_target_c_sep[int(mapped[exp_from]) + 1: \
                                                      int(mapped[exp_from]) + 1 + exp_len, d_dims]
                            target_ts[exp_from + 1:exp_from + 1 + exp_len, d_dims] = nun_subs

                            when_mapped[exp_from + 1:exp_from + 1 + exp_len] = np.full(
                                nun_subs.shape[0], fill_value=when_mapped[exp_from])
                        else:
                            # to the left of already mapped subsequence
                            # first mapped index (to start expanding from)
                            exp_from = X_inter[0]

                            # remaining nb of time steps until start of target_c_sep
                            # (until 0 index)
                            to_tcs_start = mapped[exp_from]
                            # remaining nb of time steps until start of current instance
                            # (until NaN index)
                            to_curr_start = mapped[exp_from] % (len(x_orig) + 1)

                            to_curr_start = min(to_curr_start, to_tcs_start)

                            # expand as far as possible (stopping at start of current subsequence)
                            exp_len = int(min(to_curr_start, self.window - len(X_inter)))

                            mapped[d_X_start:d_X_start + exp_len] = np.arange(
                                mapped[exp_from] - exp_len, mapped[exp_from])

                            nun_subs = X_target_c_sep[int(mapped[exp_from]) - exp_len: \
                                                      int(mapped[exp_from]), d_dims]
                            target_ts[d_X_start:d_X_start + exp_len, d_dims] = nun_subs

                            when_mapped[d_X_start:d_X_start + exp_len] = np.full_like(
                                nun_subs.shape[0], fill_value=when_mapped[exp_from])
                    elif len(self.ind_list_to_intervals(X_inter)[0]) == 2:
                        # if overlaps two, expand from first[when]
                        # if there's enough room to cover second[when]
                        # (until end of same interval, i.e. successive
                        # consequetive indices), cover it all; otherwise,
                        # only cover empty indices

                        when_left = when_mapped[X_inter[0]]
                        when_right = when_mapped[X_inter[-1]]

                        # expand from left to right
                        if when_left < when_right:
                            # last mapped index (to start expanding from)
                            exp_from = self.ind_list_to_intervals(X_inter)[0][0][-1]

                            # remaining nb of time steps until end of target_c_sep
                            to_tcs_end = len(X_target_c_sep) - 1 - mapped[exp_from]
                            # remaining nb of time steps until end of current instance
                            # (until NaN index)
                            to_curr_end = len(x_orig) - 1 - mapped[exp_from] % (len(x_orig) + 1)

                            to_curr_end = min(to_curr_end, to_tcs_end)

                            # index of first element of adjacent subsequence to the right
                            adj_start = self.ind_list_to_intervals(X_inter)[0][1][0]

                            # find length of adjacent subsequence to the right
                            adj_len = 0
                            while True:
                                if adj_start + adj_len + 1 == len(mapped):
                                    break

                                if mapped[adj_start + adj_len] + 1 == \
                                        mapped[adj_start + adj_len + 1]:
                                    adj_len += 1
                                else:
                                    break

                            adj_len += 1

                            if to_curr_end >= self.window - len(X_inter) + adj_len:
                                exp_len = self.window - len(X_inter) + adj_len
                            else:
                                exp_len = int(min(to_curr_end, self.window - len(X_inter)))

                            mapped[exp_from + 1:exp_from + 1 + exp_len] = np.arange(
                                mapped[exp_from] + 1, mapped[exp_from] + 1 + exp_len)

                            nun_subs = X_target_c_sep[int(mapped[exp_from]) + 1: \
                                                      int(mapped[exp_from]) + 1 + exp_len, d_dims]
                            target_ts[exp_from + 1:exp_from + 1 + exp_len, d_dims] = nun_subs

                            when_mapped[exp_from + 1:exp_from + 1 + exp_len] = np.full_like(
                                nun_subs.shape[0], fill_value=when_mapped[exp_from])
                        # expand from right to left
                        elif when_left > when_right:
                            # last mapped index (to start expanding from)
                            exp_from = self.ind_list_to_intervals(X_inter)[0][1][0]

                            # remaining nb of time steps until start of target_c_sep
                            # (until 0 index)
                            to_tcs_start = mapped[exp_from]
                            # remaining nb of time steps until start of current instance
                            # (until NaN index)
                            to_curr_start = mapped[exp_from] % (len(x_orig) + 1)

                            to_curr_start = min(to_curr_start, to_tcs_start)

                            # index of last element of adjacent subsequence to the left
                            adj_end = self.ind_list_to_intervals(X_inter)[0][0][-1]

                            # find length of adjacent subsequence to the left
                            adj_len = 0
                            while True:
                                if adj_end - adj_len == 0:
                                    break

                                if mapped[adj_end - adj_len] - 1 == \
                                        mapped[adj_end - adj_len - 1]:
                                    adj_len += 1
                                else:
                                    break

                            adj_len += 1

                            if to_curr_start >= self.window + adj_len:
                                exp_len = self.window - len(X_inter) + adj_len
                            else:
                                exp_len = int(min(to_curr_start, self.window - len(X_inter)))

                            mapped[exp_from - exp_len:exp_from] = np.arange(
                                mapped[exp_from] - exp_len, mapped[exp_from])

                            nun_subs = X_target_c_sep[int(mapped[exp_from]) - exp_len: \
                                                      int(mapped[exp_from]), d_dims]
                            target_ts[exp_from - exp_len:exp_from, d_dims] = nun_subs

                            when_mapped[exp_from - exp_len:exp_from] = np.full_like(
                                nun_subs.shape[0], fill_value=when_mapped[exp_from])
                        else:
                            print('something wrong, when_left==when_right')

                else:
                    # if we're here, the indices have never been mapped before
                    # or they extend previously mapped ones

                    # introduce target subseqence in target timeseries
                    nun_subs = X_target_c_sep[d_nun_start:d_nun_end, d_dims]

                    # print('else, d_X_start, d_X_end, d_nun_start, d_nun_end', d_X_start, d_X_end, d_nun_start, d_nun_end)

                    target_ts[d_X_start:d_X_end, d_dims] = nun_subs
                    mapped[d_X_start:d_X_end] = np.arange(d_nun_start, d_nun_end)
                    when_mapped[d_X_start:d_X_end] = np.full_like(when_mapped[d_X_start:d_X_end], fill_value=when)
                    when += 1

                try:
                    _, target_ts = self.mfill_short_intervals(mapped, target_ts, X_target_c_sep,
                                                         d_dims, thres=3)
                except Exception as ex:
                    print('Could not fill short', ex)

                for w_i_, w in enumerate(self.steps):
                    X_cf = (1 - w) * x_orig + w * target_ts
                    X_cf = self.minterpolate_short(X_cf, mapped, d_dims, thres=3)

                    y_cf = self.predict(X_cf, self.model_wrapper)[0]

                    if y_cf == target_c:
                        cf_found = True
                        print('CF found!, From', orig_y, 'to', y_cf, 'target_c', target_c)
                        x_cf_list.append(X_cf)
                        break

                # non-mapped indices
                nan_map = np.where(np.isnan(mapped))[0]
                if len(nan_map) > 0:
                    # split non-mapped indices into separate intervals
                    nan_map_intervals, _ = self.ind_list_to_intervals(nan_map)
                    max_len_non_mapped = max([len(_) for _ in nan_map_intervals])
                else:
                    n_non_mapped = 0

            if not cf_found:
                # Last try by filling up non-mapped intervals 3<len<min_w
                try:
                    _, target_ts = self.mfill_short_intervals(mapped, target_ts, X_target_c_sep,
                                                         d_dims, thres=3)
                except Exception as ex:
                    print('Could not fill short', ex)

                for w_i_, w in enumerate(self.steps):
                    X_cf = (1 - w) * x_orig + w * target_ts
                    X_cf = self.minterpolate_short(X_cf, mapped, d_dims, thres=3)

                    y_cf = self.predict(X_cf, self.model_wrapper)[0]

                    if y_cf == target_c:
                        cf_found = True
                        print('CF found! (last), From', orig_y, 'to', y_cf, 'target_c', target_c)
                        x_cf_list.append(X_cf)
                        break

                # print('last mapped')

            if not cf_found:
                print('CF not found... orig_y, y_cf, target_c, w', orig_y, y_cf, target_c, w)

        # Simplest choice: choose the first counterfactual that appears
        if not x_cf_list:
            x_cf = x_orig.copy()
        else:
            x_cf = x_cf_list[0]
        result = {'cf': np.expand_dims(x_cf, axis=0)}

        return result
