import numpy as np
import os
import random
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sys
from classifiers import resnet_val as resnet
from sktime_convert import from_nested_to_3d_numpy, load_from_tsfile_to_dataframe, from_nested_to_multi_index
from mutils import cal_tags_mps, fast_find_anomalies, idx_to_str, get_score, get_discord, get_custom_data

random.seed(42)
np.random.seed(42)

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

def load_classifier(x_train, y_train, x_test, y_test,
                output_directory=None, nb_epochs=1500, weights_directory=None):
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier = resnet.Classifier_RESNET(output_directory,
                                          input_shape,
                                          nb_classes, verbose=True,
                                          load_weights= True,
                                          weights_directory=weights_directory)

    return classifier.model

def predict(X, model):
    if len(X.shape)==2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X = X.reshape((1, X.shape[0], X.shape[1]))
    elif len(X.shape)==1:
        X = X.reshape((1,-1,1))

    y_pred = model.predict(X)
    return np.argmax(y_pred, axis=1)

def ind_list_to_intervals(lst):
    #split non-mapped indices into separate intervals
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
            curr_interval_inds.append(ind+1)
        else:
            intervals.append(curr_interval)
            intervals_inds.append(curr_interval_inds)
            curr_interval = [i]
            curr_interval_inds = [ind+1]
        prev_i = i

    intervals.append(curr_interval)
    intervals_inds.append(curr_interval_inds)

    return intervals, intervals_inds

#wrapper for multidimensional time series
def mfill_short_intervals(mapping, target_ts, X_target_c_sep, d_dims, thres=3):
    new_target_ts = target_ts.copy()
    
    for dim in d_dims:
        target_ts_d = target_ts[:,dim]
        X_target_c_sep_d = X_target_c_sep[:,dim]
        
        new_mapping, new_target_ts[:,dim] =\
            fill_short_intervals(mapping, target_ts_d, X_target_c_sep_d, thres=thres)
                
    return new_mapping, new_target_ts

#fill short non-mapped subsequences by extending their longest adjacent mapped subsequence
def fill_short_intervals(mapping, target_ts, X_target_c_sep, thres=3):
    #non-mapped indices
    non_map = np.where(np.isnan(mapping))[0]
    if len(non_map)>0:
        non_map_intervals, _ = ind_list_to_intervals(non_map)
    
        #find position of current non-mapped subsequence and idx of subsequence to extend
        for interval in non_map_intervals:
            mapped_intervals, mapped_intervals_inds = ind_list_to_intervals(mapping)
            if len(interval) < thres:
                for mii_i, mii in enumerate(mapped_intervals_inds):
                    if interval[0] == mii[0]:
                        if mii_i == 0:
                            #the current non-mapped subsequence is at the beginning of TS
                            fill_from = mii_i + len(interval)
                            backup_fill_from = -1
                            plus = 1
                        elif mii_i + len(interval) == len(mapped_intervals_inds):
                            #the current non-mapped subsequence is at the end of TS
                            fill_from = mii_i - 1
                            backup_fill_from = -1
                            plus = -1
                        else:
                            #the current non-mapped subsequence is in the middle of TS
                            #so, find the lengths of adjacent subsequences
                            if len(mapped_intervals_inds[mii_i-1])>\
                                len(mapped_intervals_inds[mii_i+len(interval)]):
                                fill_from = mii_i - 1
                                backup_fill_from = mii_i + 1
                                plus = -1
                            else:
                                fill_from = mii_i + len(interval)
                                backup_fill_from = mii_i - 1
                                plus = 1
                        break
                    
                #fill non-mapped subsequence
                #first check if there are no nans in it (from instances borders)
                if plus == 1:
                    temp_mapped = np.array(mapped[mapped_intervals_inds[fill_from]])[-len(interval):] -\
                                        len(mapped_intervals_inds[fill_from])
                else:
                    temp_mapped = np.array(mapped[mapped_intervals_inds[fill_from]])[:len(interval)] +\
                                        len(mapped_intervals_inds[fill_from])
                
                #if trying to expand beyond length of X_target_c_sep (and 
                #the current non-mapped subsequence is at the end of TS), don't
                #will interpolate later in main code
                beyond = np.where(temp_mapped>=len(X_target_c_sep))[0]
                
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
                
                #if trying to expand before idx 0 of X_target_c_sep (and 
                #the current non-mapped subsequence is at the beginning of TS), don't
                #will interpolate later in main code
                beyond = np.where(temp_mapped<0)[0]
                
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
                
                #if there are nans, replace starting from first nan with backup
                if len(nan_temp) and backup_fill_from!=-1:
                    print('nans in fill from')
                    if plus == 1:
                        backup_mapped = np.array(mapping[mapped_intervals_inds[backup_fill_from]])[:len(interval)] +\
                                    len(mapped_intervals_inds[backup_fill_from])
                    else:
                        backup_mapped = np.array(mapping[mapped_intervals_inds[backup_fill_from]])[-len(interval):] -\
                                    len(mapped_intervals_inds[backup_fill_from])
                    
                    temp_mapped[nan_temp[0]:] = backup_mapped[nan_temp[0]:]
                    
                nan_temp = np.where(np.isnan(temp_target_ts))[0]
                
                #if there are still nans, leave them and interpolate
                if len(nan_temp):
                    print('nans in backup fill from')
                    #inds in between nans should be interpolated too
                    temp_target_ts[min(nan_temp):max(nan_temp)+1] = np.nan
                    temp_mapped[min(nan_temp):max(nan_temp)+1] = np.inf
                   
                mapping[interval] = temp_mapped
                target_ts[interval] = temp_target_ts
                
    return mapping, target_ts

#wrapper for multidimensional time series
def minterpolate_short(X_cf, mapping, d_dims, thres=3):
    new_cf = X_cf.copy()
    
    for dim in d_dims:
        cf_d = X_cf[:,dim]
        new_cf[:,dim] = interpolate_short(cf_d, mapping, thres=thres)
                
    return new_cf

#interpolate intervals shorter than tresh
def interpolate_short(X_cf, mapping, thres=3):
    #non-mapped indices
    mapping[mapping == np.inf] = np.nan
    non_map = np.where(np.isnan(mapping))[0]
    if len(non_map)>0:
        non_map_intervals, _ = ind_list_to_intervals(non_map)
        
        for interval in non_map_intervals:
            if len(interval) < thres:
                X_cf[interval] = np.nan
            
        #interpolate short nan intervals
        X_cf = pd.DataFrame(X_cf)
        mask = X_cf.copy()
        grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
        grp['ones'] = 1
        for i in [0]:
            mask[i] = (grp.groupby(i)['ones'].transform('count') < thres) | X_cf[i].notnull()
        X_cf = X_cf.interpolate().bfill()[mask]
        X_cf = X_cf.to_numpy().reshape(-1)
    
    return X_cf

name = sys.argv[1]

steps = np.concatenate([ np.arange(0.01, 0.5, 0.01), np.arange(0.5, 1.01, 0.02)])

X_train, y_train = load_from_tsfile_to_dataframe(
    os.path.join(data_path, name, name + "_TRAIN.ts")
)

X_test, y_test = load_from_tsfile_to_dataframe(
    os.path.join(data_path, name, name + "_TEST.ts")
)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

multi_index_X_train = from_nested_to_multi_index(X_train)
multi_index_X_test = from_nested_to_multi_index(X_test)

X_train = from_nested_to_3d_numpy(X_train)
X_test = from_nested_to_3d_numpy(X_test)

X_train = X_train.reshape((X_train.shape[0],X_train.shape[2],-1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[2],-1))

seed = 42
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models',
                          name, 'models_1500_' + str(seed)))
            
model = load_classifier(X_train, y_train, X_test, y_test,
                        output_directory='', weights_directory=model_path)

threshold = 0.90
sample_pct = 1

window = int(0.1*X_train.shape[1])
    
if os.path.exists(os.path.join(os.sep, results_path, name, 'X_test_cfs_discox.npy')):
    print(os.path.join(os.sep, results_path, name, 'X_test_cfs_discox.npy'))
    print('Already there.')
    sys.exit()

X_cfs, y_cfs = [], []
no_cf_cnt = 0
ws = []

for X_i, X in enumerate(X_test):
    print('X_i', X_i)
    orig_y = predict(X, model)[0]
    for target_c in np.unique(y_train):
        cf_found = False

        print('target_c, orig_y', target_c, orig_y)

        if target_c == orig_y:
            print('same, continue')
            continue

        target_c_inds = np.where(y_train == target_c)[0]
        X_target_c = X_train[target_c_inds]
        X_target_c_sep = np.concatenate((X_target_c[0],
                             np.full((1,X.shape[1]), fill_value=np.nan)))
        for X_ in X_target_c[1:]:
            X_target_c_sep = np.concatenate((X_target_c_sep, X_))
            X_target_c_sep = np.concatenate((X_target_c_sep,
                                 np.full((1,X.shape[1]), fill_value=np.nan)))
        
        X_target_c_sep = X_target_c_sep[:-1,:]

        print('X_i, target_c', X_i, target_c)

        mps, _ = cal_tags_mps(X, win=window, query=X_target_c_sep)
        
        kdps , kdps_idx = fast_find_anomalies(mps)
        
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

        mapped = np.full(X_train.shape[1], fill_value=np.nan)
        when_mapped = np.full(X_train.shape[1], fill_value=np.nan)
        target_ts = X.copy()

        when = 1

        # the number of non-mapped intervals
        n_non_mapped = np.inf
        # the length of the longest non-mapped intervals
        max_len_non_mapped = np.inf

        # until all time steps are mapped or only intervals shorter than 3 time steps remain unmapped
        while n_non_mapped and max_len_non_mapped >= window and not cf_found:
            # if all indices have been mapped
            if not np.any(np.isnan(mapped)):
                break

            if cf_found:
                break

            d_X_start, d_dims, d_nun_start, all_scores = get_discord(X, window,
                                  X_target_c_sep, all_scores, kdps_idx, all_sets)
            
            d_X_end = d_X_start + window
            d_nun_end = d_nun_start + window

            # mapped indices
            already_mapped = np.delete(np.arange(len(mapped)),
                                       np.where(np.isnan(mapped))[0])

            inter_ids = np.in1d(np.arange(d_X_start, d_X_end),
                                already_mapped)
            X_inter = np.arange(d_X_start, d_X_end)[inter_ids]
              
            # if any index of the current subsequence has been mapped (but not all)
            if len(X_inter):                    
                if len(X_inter) == window:
                    continue
                
                # if only intersects one already mapped subsequence
                if len(ind_list_to_intervals(X_inter)[0]) == 1:
                    # to the right of already mapped subsequence
                    if X_inter[-1]+1 in np.arange(d_X_start, d_X_end):
                        # last mapped index (to start expanding from)
                        exp_from = X_inter[-1]

                        # remaining nb of time steps until end of target_c_sep
                        to_tcs_end = len(X_target_c_sep) - mapped[exp_from]
                        # remaining nb of time steps until end of current instance
                        # (until NaN index)
                        to_curr_end = len(X) - 1 - mapped[exp_from] % (len(X) + 1)

                        to_curr_end = min(to_curr_end, to_tcs_end)

                        # expand as far as possible (stopping at end of current subsequence)
                        exp_len = int(min(to_curr_end, window-len(X_inter)))
                        
                        mapped[exp_from + 1:exp_from + 1 + exp_len] = np.arange(
                            mapped[exp_from] + 1, mapped[exp_from] + 1 + exp_len)
                        
                        nun_subs = X_target_c_sep[int(mapped[exp_from]) + 1:\
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
                        to_curr_start = mapped[exp_from] % (len(X) + 1)

                        to_curr_start = min(to_curr_start, to_tcs_start)

                        # expand as far as possible (stopping at start of current subsequence)
                        exp_len = int(min(to_curr_start, window-len(X_inter)))
                                  
                        mapped[d_X_start:d_X_start + exp_len] = np.arange(
                            mapped[exp_from] - exp_len, mapped[exp_from])
                                                        
                        nun_subs = X_target_c_sep[int(mapped[exp_from]) - exp_len:\
                                                  int(mapped[exp_from]), d_dims]
                        target_ts[d_X_start:d_X_start + exp_len, d_dims] = nun_subs
                        
                        when_mapped[d_X_start:d_X_start + exp_len] = np.full_like(
                            nun_subs.shape[0], fill_value=when_mapped[exp_from])
                elif len(ind_list_to_intervals(X_inter)[0])==2:
                    # if overlaps two, expand from first[when]
                    # if there's enough room to cover second[when]
                    # (until end of same interval, i.e. successive
                    # consequetive indices), cover it all; otherwise,
                    # only cover empty indices
                    
                    when_left = when_mapped[X_inter[0]]
                    when_right = when_mapped[X_inter[-1]]
                    
                    #expand from left to right
                    if when_left < when_right:
                        # last mapped index (to start expanding from)
                        exp_from = ind_list_to_intervals(X_inter)[0][0][-1]
                       
                        # remaining nb of time steps until end of target_c_sep
                        to_tcs_end = len(X_target_c_sep) - 1 - mapped[exp_from]
                        # remaining nb of time steps until end of current instance
                        # (until NaN index)
                        to_curr_end = len(X) - 1 - mapped[exp_from] % (len(X) + 1)

                        to_curr_end = min(to_curr_end, to_tcs_end)

                        #index of first element of adjacent subsequence to the right 
                        adj_start = ind_list_to_intervals(X_inter)[0][1][0]
                                                        
                        #find length of adjacent subsequence to the right 
                        adj_len = 0
                        while True:
                            if adj_start + adj_len + 1 == len(mapped):
                                break
                            
                            if mapped[adj_start + adj_len] + 1 ==\
                                mapped[adj_start + adj_len + 1]:
                                    adj_len += 1
                            else:
                                break
                            
                        adj_len += 1
                        
                        if to_curr_end >= window - len(X_inter) + adj_len:
                            exp_len = window - len(X_inter) + adj_len
                        else:
                            exp_len = int(min(to_curr_end, window-len(X_inter)))
                        
                        mapped[exp_from + 1:exp_from + 1 + exp_len] = np.arange(
                                mapped[exp_from] + 1, mapped[exp_from] + 1 + exp_len)
                        
                        nun_subs = X_target_c_sep[int(mapped[exp_from]) + 1:\
                                                  int(mapped[exp_from]) + 1 + exp_len, d_dims]
                        target_ts[exp_from + 1:exp_from + 1 + exp_len, d_dims] = nun_subs                                
                        
                        when_mapped[exp_from + 1:exp_from + 1 + exp_len] = np.full_like(
                            nun_subs.shape[0], fill_value=when_mapped[exp_from])
                    #expand from right to left
                    elif when_left > when_right:
                        # last mapped index (to start expanding from)
                        exp_from = ind_list_to_intervals(X_inter)[0][1][0]
                        
                        # remaining nb of time steps until start of target_c_sep
                        # (until 0 index)
                        to_tcs_start = mapped[exp_from]
                        # remaining nb of time steps until start of current instance
                        # (until NaN index)
                        to_curr_start = mapped[exp_from] % (len(X) + 1)

                        to_curr_start = min(to_curr_start, to_tcs_start)
                        
                        #index of last element of adjacent subsequence to the left 
                        adj_end = ind_list_to_intervals(X_inter)[0][0][-1]
                        
                        #find length of adjacent subsequence to the left 
                        adj_len = 0
                        while True:
                            if adj_end - adj_len == 0:
                                break
                            
                            if mapped[adj_end - adj_len] - 1 ==\
                                mapped[adj_end - adj_len - 1]:
                                   adj_len += 1
                            else:
                                break
                        
                        adj_len += 1
                        
                        if to_curr_start >= window + adj_len:
                            exp_len = window - len(X_inter) + adj_len
                        else:
                            exp_len = int(min(to_curr_start, window-len(X_inter)))
                         
                        mapped[exp_from - exp_len:exp_from] = np.arange(
                            mapped[exp_from] - exp_len, mapped[exp_from])

                        nun_subs = X_target_c_sep[int(mapped[exp_from]) - exp_len:\
                                                  int(mapped[exp_from]), d_dims]
                        target_ts[exp_from - exp_len:exp_from, d_dims] = nun_subs                                
                        
                        when_mapped[exp_from - exp_len:exp_from] = np.full_like(
                            nun_subs.shape[0], fill_value=when_mapped[exp_from])
                    else:
                        print('something wrong, when_left==when_right')
                        
            else:
                #if we're here, the indices have never been mapped before
                #or they extend previously mapped ones
             
                #introduce target subseqence in target timeseries
                nun_subs = X_target_c_sep[d_nun_start:d_nun_end, d_dims]
                
                print('else, d_X_start, d_X_end, d_nun_start, d_nun_end',
                      d_X_start, d_X_end, d_nun_start, d_nun_end)
                
                target_ts[d_X_start:d_X_end, d_dims] = nun_subs
                mapped[d_X_start:d_X_end] = np.arange(d_nun_start, d_nun_end)          
                when_mapped[d_X_start:d_X_end] = np.full_like(when_mapped[d_X_start:d_X_end], fill_value=when)
                when += 1
                        
            try:
                _, target_ts = mfill_short_intervals(mapped, target_ts, X_target_c_sep,
                                             d_dims, thres=3)
            except Exception as ex:
                print('Could not fill short', ex)
        
            for w_i_, w in enumerate(steps): 
                X_cf = (1-w)*X + w*target_ts
                X_cf = minterpolate_short(X_cf, mapped, d_dims, thres=3)
                
                y_cf = predict(X_cf, model)[0]
                
                if y_cf == target_c:
                    cf_found = True
                    print('CF found!, From', orig_y, 'to', y_cf, 'target_c', target_c)
                    ws.append(w)
                    X_cfs.append(X_cf)
                    y_cfs.append(y_cf)
                    break
                            
            #non-mapped indices
            nan_map = np.where(np.isnan(mapped))[0]
            if len(nan_map)>0:
                #split non-mapped indices into separate intervals
                nan_map_intervals, _ = ind_list_to_intervals(nan_map)
                max_len_non_mapped = max([len(_) for _ in nan_map_intervals])
            else: 
                n_non_mapped = 0
        
        if not cf_found:
            #Last try by filling up non-mapped intervals 3<len<min_w
            try:
                _, target_ts = mfill_short_intervals(mapped, target_ts, X_target_c_sep,
                                             d_dims, thres=3)
            except Exception as ex:
                print('Could not fill short', ex)
                
            for w_i_, w in enumerate(steps): 
                X_cf = (1-w)*X + w*target_ts
                X_cf = minterpolate_short(X_cf, mapped, d_dims, thres=3)
                
                y_cf = predict(X_cf, model)[0]
                
                if y_cf == target_c:
                    cf_found = True
                    print('CF found! (last), From', orig_y, 'to', y_cf, 'target_c', target_c)
                    ws.append(w)
                    X_cfs.append(X_cf)
                    y_cfs.append(y_cf)
                    break
                        
            print('last mapped')
                    
        if not cf_found:
            print('CF not found... orig_y, y_cf, target_c, w', orig_y, y_cf,
                  target_c, w)
            X_cfs.append(np.full_like(X, fill_value=-1))
            y_cfs.append(-1)
            no_cf_cnt += 1
    
if not os.path.exists(os.path.join(os.sep, results_path, name)):
    os.makedirs(os.path.join(os.sep, results_path, name))

X_cfs = np.asarray(X_cfs)
y_cfs = np.asarray(y_cfs)

print('no_cf_cnt', no_cf_cnt)
    
np.save(os.path.join(os.sep, results_path, name, 'X_test_cfs_discox.npy'), X_cfs)
np.save(os.path.join(os.sep, results_path, name, 'y_test_cfs_discox.npy'), y_cfs)

print('mean w', np.mean(ws))

