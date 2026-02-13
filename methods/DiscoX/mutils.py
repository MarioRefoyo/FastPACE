'''The bulk of this file was written by the authors of TSADIS. We made some modifications to allow
outer-joins (the of the paramater 'query') and added the 'get_discord()' function for our use'''


import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
import os
import stumpy

def get_win(x):
    this_tag =np.squeeze(MinMaxScaler((0,np.abs(2*np.average(x)))).fit_transform(x.reshape(-1,1)))
    peaks = find_peaks(this_tag, prominence=np.abs(np.average(x)/2))[0]
    return np.median(np.diff(peaks, axis=-1))

def cal_tags_mps(ts, win=None, query=None, train_range=None):
    all_tags_profiles=[]
    all_mpis=[]
    win_size =win
    if np.ndim(ts)==1:
        if win_size is None:
            win_size=get_win(ts)
            print('win:',win_size)
        # profile = mpx.compute(ts,win_size,query=query)['mp']
        mp_ = stumpy.stump(ts, win_size, query, ignore_trivial = False, normalize=False)
        profile = mp_[:, 0].astype(float)
        mpi = mp_[:, 1].astype(int)
        return np.asarray([profile]).T, np.asarray([mpi]).T
    
    for i in np.arange(ts.shape[1]):
        this_tag = ts[:,i]
        this_query = query[:,i]
        if win is None:
            win_size=get_win(this_tag)
            print('win:',win_size)
        # profile = mpx.compute(this_tag,win_size,query=this_query)['mp']
        mp_  = stumpy.stump(this_tag, win_size, this_query, ignore_trivial = False, normalize=False)
        profile = mp_[:, 0].astype(float)
        mpi = mp_[:, 1].astype(int)
        pad_size =len(this_tag) -len(profile)
        profile = np.insert(profile, len(profile),[np.min(profile)]*pad_size)
        mpi = np.insert(mpi, len(mpi),-1*pad_size)
        all_tags_profiles.append(profile)
        all_mpis.append(mpi)
    all_tags_profiles=np.asarray(all_tags_profiles).T
    all_mpis=np.asarray(all_mpis).T
    return all_tags_profiles, all_mpis

def fast_find_anomalies(mps):
    #get all-kdp-profils (def. 14)
    KDPs = np.sort(mps, axis = 1)
    KDPs_idx = np.argsort(mps, axis = 1)
    return np.flip(KDPs,  axis=1), np.flip(KDPs_idx, axis=1) # flip the rows odered just for visualization, otherwise does not impact the result

def idx_to_str(idxs):
  idxs = idxs.astype(int)
  all_ = []
  for k in np.arange(idxs.shape[1]):
      this_p = idxs[:,:k+1]
      string_p = []
      for i in np.arange(len(this_p)):
          string_p.append(str(set(this_p[i,:])))
      all_.append(string_p)
  return np.asarray(all_)

def get_score(val, method='sum'):
  if method =='sum':
    all_ = np.zeros_like(val)
    for i in np.arange(val.shape[1]):
        all_[:,i]= np.sum(val[:,:i+1], axis=1)
    return all_
  elif  method =='min':
    return val
  elif method == 'mean':
      all_ = np.zeros_like(val)
      for i in np.arange(val.shape[1]):
          all_[:,i]= np.mean(val[:,:i+1], axis=1)
      return all_

def get_custom_data(s, str_name, k):
    # my_custom_data = np.empty((len(s),2,1), dtype='object')
    my_custom_data = np.empty((len(s),3,1), dtype='object')
    this_s = s[:,k].reshape(-1,1)
    this_set = str_name[k].reshape(-1,1)
    all_set = str_name[-1].reshape(-1,1)
    # my_custom_data = np.asarray([[i,j] for i, j in zip(this_s, this_set)])
    my_custom_data[:,0] = this_s
    my_custom_data[:,1] = this_set
    my_custom_data[:,2] = all_set
    return my_custom_data

def get_discord(ts, win, query, all_scores, kdps_idx, all_sets):
    d_idx = np.unravel_index(np.argmax(all_scores, axis=None), all_scores.shape)    
    
    #check if the anomaly score doesn't increase when adding dimensions
    for k in np.arange(d_idx[0], kdps_idx.shape[1]):
        if all_scores[k,d_idx[1]] > all_scores[d_idx] + 1e-5:
            d_idx = (k,d_idx[1])
            
    #check if the anomaly score stays the same when removing dimensions
    for k in np.arange(0, d_idx[0]):
        if np.abs(all_scores[k,d_idx[1]] - all_scores[d_idx]) <= 1e-5:
            d_idx = (k,d_idx[1]) 
    
    d_sets_ = all_sets[d_idx][0].split(', ')
    
    d_sets = []
    for str_ in d_sets_:
        d_sets.append(''.join(c for c in str_ if c.isdigit()))
    
    #the natural dimensions of the discord    
    d_sets = np.array(d_sets, dtype=np.int8)
        
    #compute and aggregate the DPs of each dimension of the discord
    d_dps = []
    for dim in d_sets:
        d_dim = ts[d_idx[1]:d_idx[1] + win, dim]    
        d_dps.append(stumpy.mass(d_dim, query[:,dim]))
        
    d_dps = np.array(d_dps)
    d_dps = np.sum(d_dps, axis=0)
    
    #get nearest neighbor of discord
    d_nn = np.argmin(d_dps)
    
    # print(d_dps[d_nn], d_dps[255:270])
    #exclude current discord from future queries
    all_scores[:,d_idx[1]] = np.full(all_scores.shape[0], fill_value=-np.inf)
    
    return d_idx[1], d_sets, d_nn, all_scores