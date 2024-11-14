from copy import deepcopy as copy
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d as smooth
from scipy import stats
from sklearn import linear_model
import sys

from aux import Generic, load_npy, split
from disp import set_plot
from my_stats import get_r2

cc = np.concatenate


def skl_fit_ridge(
        pfxs, cols_x, targs, itr_all, ntrain, nsplit, mask_pfx=None,
        return_y=None, return_nrl=None, alpha=10, verbose=True, seed=0, **kwargs):
    """
    Use scikit-learn to fit a linear model using ridge regression.
    :param pfxs: data file prefixes (original neural/behav file & extended behavior file)
    :param cols_x: which columns to use to fit
    :param targs: which targets to fit
    :param itr_all: idxs of all trials to sample training/test data from
    :param ntrain: number of training trials
    :param nsplit: number of training/test splits
    """
    if return_y is None:
        return_y = []
    if return_nrl is None:
        return_nrl = []
        
    # load all data
    if verbose is True:  print('Loading...')
    
    # main data frames with surrogate neural recordings and basic behav quantities
    dfs_0 = {itr: np.load(f'{pfxs[0]}_tr_{itr}.npy', allow_pickle=True)[0]['df'] for itr in itr_all}
    # corresponding data frames with extended behavioral quantities
    dfs_1 = {itr: pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_all}
    
    if mask_pfx is not None:
        masks = {itr: np.load(f'{mask_pfx}_tr_{itr}.npy', allow_pickle=True)[0]['mask'] for itr in itr_all}
    
    # loop over splits
    np.random.seed(seed)
    
    rslts = []
    itr_train_all = []
    itr_test_all = []
    
    for csplit in range(nsplit):
        if verbose is True:
            sys.stdout.write(f'\nSplit {csplit}')
        elif verbose == 'dots':
            sys.stdout.write('.')
        
        irnd = np.random.permutation(len(itr_all))
        
        # training trials
        itr_train = itr_all[irnd[:ntrain]]
        
        dfs_train_0 = [dfs_0[itr] for itr in itr_train]
        dfs_train_1 = [dfs_1[itr] for itr in itr_train]
        
        nts_train = [len(df_train) for df_train in dfs_train_0]  # length (num timesteps) of training trials
        its_start_train = cc([[0], np.cumsum(nts_train)[:-1]])  # training trial start time indexes
        its_end_train = np.cumsum(nts_train)  # training trial end time indexes
        
        # test trials
        itr_test = itr_all[irnd[ntrain:]]
        
        dfs_test_0 = [dfs_0[itr] for itr in itr_test]
        dfs_test_1 = [dfs_1[itr] for itr in itr_test]
        
        nts_test = [len(df_test) for df_test in dfs_test_0]
        its_start_test = cc([[0], np.cumsum(nts_test)[:-1]])
        its_end_test = np.cumsum(nts_test)
    
        cols_0 = dfs_train_0[0].columns
        cols_1 = dfs_train_1[0].columns
        
        xs_train_ = [np.array(df_train[cols_x]) for df_train in dfs_train_0]
        xs_test_ = [np.array(df_test[cols_x]) for df_test in dfs_test_0]
        
        if mask_pfx is not None:
            
            masks_train = [masks[itr] for itr in itr_train]
            masks_test = [masks[itr] for itr in itr_test]
            
            for ctr_train, mask_train in enumerate(masks_train):
                xs_train_[ctr_train][~mask_train, :] = np.nan
                
            for ctr_test, mask_test in enumerate(masks_test):
                xs_test_[ctr_test][~mask_test, :] = np.nan
                
        xs_train = cc(xs_train_)
        xs_test = cc(xs_test_)
        
        mvalid_train_x = ~np.any(np.isnan(xs_train), axis=1)
        mvalid_test_x = ~np.any(np.isnan(xs_test), axis=1)
        
        rslt = Generic(
            alpha=alpha, targs=targs,
            w={}, bias={},
            ys_train={}, y_hats_train={}, rms_err_train={}, r2_train={}, 
            ys_test={}, y_hats_test={}, rms_err_test={}, r2_test={},
            songs_train=[], songs_test=[], nrl_train=[], nrl_test=[])

        ys_train = []
        ys_test = []
        
        for targ in targs:
            
            if targ in cols_0:
                ys_train.append(cc([np.array(df_train[targ]) for df_train in dfs_train_0]))
                ys_test.append(cc([np.array(df_test[targ]) for df_test in dfs_test_0]))
                
            elif targ in cols_1:
                ys_train.append(cc([np.array(df_train[targ]) for df_train in dfs_train_1]))
                ys_test.append(cc([np.array(df_test[targ]) for df_test in dfs_test_1]))
                
            else:
                raise KeyError(f'Column with label "{targ}" not found.')
                
        # create array of targs
        ys_train = np.transpose(ys_train)
        ys_test = np.transpose(ys_test)
        
        mvalid_train_y = ~np.any(np.isnan(ys_train), axis=1)
        mvalid_test_y = ~np.any(np.isnan(ys_test), axis=1)
        
        mvalid_train = (mvalid_train_x & mvalid_train_y)
        mvalid_test = (mvalid_test_x & mvalid_test_y)
        
        rgr = linear_model.Ridge(alpha=alpha).fit(xs_train[mvalid_train, :], ys_train[mvalid_train, :])
        
        y_hats_train = np.nan*np.zeros(ys_train.shape)
        y_hats_test = np.nan*np.zeros(ys_test.shape)
        
        y_hats_train[mvalid_train, :] = rgr.predict(xs_train[mvalid_train, :])
        y_hats_test[mvalid_test, :] = rgr.predict(xs_test[mvalid_test, :])
        
        for ctarg, targ in enumerate(targs):

            # store basic vars
            rslt.w[targ] = rgr.coef_[ctarg, :]
            rslt.bias[targ] = rgr.intercept_[ctarg]

            rslt.r2_train[targ] = get_r2(ys_train[:, ctarg], y_hats_train[:, ctarg])
            rslt.r2_test[targ] = get_r2(ys_test[:, ctarg], y_hats_test[:, ctarg])
            
            rslt.rms_err_train[targ] = np.sqrt(np.mean((ys_train[:, ctarg] - y_hats_train[:, ctarg])**2))
            rslt.rms_err_test[targ] = np.sqrt(np.mean((ys_test[:, ctarg] - y_hats_test[:, ctarg])**2))

            # store targs and predictions
            if csplit in return_y:
                rslt.ys_train[targ] = split(ys_train[:, ctarg], its_start_train, its_end_train)
                rslt.ys_test[targ] = split(ys_test[:, ctarg], its_start_test, its_end_test)

                rslt.y_hats_train[targ] = split(y_hats_train[:, ctarg], its_start_train, its_end_train)
                rslt.y_hats_test[targ] = split(y_hats_test[:, ctarg], its_start_test, its_end_test)
            
        rslt.itr_train = copy(itr_train)
        rslt.itr_test = copy(itr_test)
        
        # store songs (making for easy plotting directly from rslt object)
        if csplit in return_y:
            
            # train
            rslt.songs_train = []
            rslt.ts_train = []
            for df_train_0 in dfs_train_0:
                song = np.zeros(len(df_train_0), dtype=int)
                song[np.array(df_train_0['S']) == 1] = 1
                song[np.array(df_train_0['P']) == 1] = 2
                song[np.array(df_train_0['F']) == 1] = 3
                
                rslt.songs_train.append(song.copy())
                rslt.ts_train.append(np.array(df_train_0['T']))
            
            # test
            rslt.songs_test = []
            rslt.ts_test = []
            for df_test_0 in dfs_test_0:
                song = np.zeros(len(df_test_0), dtype=int)
                song[np.array(df_test_0['S']) == 1] = 1
                song[np.array(df_test_0['P']) == 1] = 2
                song[np.array(df_test_0['F']) == 1] = 3
                
                rslt.songs_test.append(song.copy())
                rslt.ts_test.append(np.array(df_test_0['T']))
                
        # store neural predictors
        if csplit in return_nrl:
            rslt.xs_train = copy(xs_train_)
            rslt.xs_test = copy(xs_test_)
                
        rslts.append(rslt)
        
    return rslts


def skl_fit_lin_single(pfxs, cols_x, targs, itr_all, ntrain, nsplit, mask_pfx=None, seed=0, verbose=False, **kwargs):
    """Use scikit-learn to fit a linear model, but looping over single columns as predictors."""
    ncol_x = len(cols_x)
    
    # load all data
    if verbose is True:  print('\nLoading...')
    
    # main data frames with surrogate neural recordings and basic behav quantities
    dfs_0 = {itr: np.load(f'{pfxs[0]}_tr_{itr}.npy', allow_pickle=True)[0]['df'] for itr in itr_all}
    # corresponding data frames with extended behavioral quantities
    dfs_1 = {itr: pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_all}
    
    if mask_pfx is not None:
        masks = {itr: np.load(f'{mask_pfx}_tr_{itr}.npy', allow_pickle=True)[0]['mask'] for itr in itr_all}
        
    # loop over splits
    np.random.seed(seed)
    
    rslt = Generic(
        r2_train={targ: np.nan*np.zeros((nsplit, ncol_x)) for targ in targs},
        r2_test={targ: np.nan*np.zeros((nsplit, ncol_x)) for targ in targs},
        w={targ: np.nan*np.zeros((nsplit, ncol_x)) for targ in targs},
        bias={targ: np.nan*np.zeros((nsplit, ncol_x)) for targ in targs})
    
    sys.stdout.write('Splits:\n')
    for csplit in range(nsplit):
        if verbose:
            sys.stdout.write('>')
        
        irnd = np.random.permutation(len(itr_all))
        
        # training trials
        itr_train = itr_all[irnd[:ntrain]]
        
        dfs_train_0 = [dfs_0[itr] for itr in itr_train]
        dfs_train_1 = [dfs_1[itr] for itr in itr_train]
    
        # test trials
        itr_test = itr_all[irnd[ntrain:]]
        
        dfs_test_0 = [dfs_0[itr] for itr in itr_test]
        dfs_test_1 = [dfs_1[itr] for itr in itr_test]
        
        cols_0 = dfs_train_0[0].columns
        cols_1 = dfs_train_1[0].columns
    
        xs_train_ = [np.array(df_train[cols_x]) for df_train in dfs_train_0]
        xs_test_ = [np.array(df_test[cols_x]) for df_test in dfs_test_0]
        
#         if ignore_pre_song:
#             # set to nan all times before song starts, which is same as time before neural activity starts
#             for ctr_train, x_train_ in enumerate(xs_train_):
#                 quiet = np.all(x_train_ == 0, axis=1)
#                 t_first_song = np.nonzero(~quiet)[0][0]
#                 xs_train_[ctr_train][:t_first_song, :] = np.nan
                
#             for ctr_test, x_test_ in enumerate(xs_test_):
#                 quiet = np.all(x_test_ == 0, axis=1)
#                 t_first_song = np.nonzero(~quiet)[0][0]
#                 xs_test_[ctr_test][:t_first_song, :] = np.nan

        if mask_pfx is not None:
            
            masks_train = [masks[itr] for itr in itr_train]
            masks_test = [masks[itr] for itr in itr_test]
            
            for ctr_train, mask_train in enumerate(masks_train):
                xs_train_[ctr_train][~mask_train, :] = np.nan
                
            for ctr_test, mask_test in enumerate(masks_test):
                xs_test_[ctr_test][~mask_test, :] = np.nan
                
        xs_train = cc(xs_train_)
        xs_test = cc(xs_test_)
        
        ys_train = []
        ys_test = []
    
        for targ in targs:
            
            if targ in cols_0:
                ys_train.append(cc([np.array(df_train[targ]) for df_train in dfs_train_0]))
                ys_test.append(cc([np.array(df_test[targ]) for df_test in dfs_test_0]))
                
            elif targ in cols_1:
                ys_train.append(cc([np.array(df_train[targ]) for df_train in dfs_train_1]))
                ys_test.append(cc([np.array(df_test[targ]) for df_test in dfs_test_1]))
                
            else:
                raise KeyError(f'Column with label "{targ}" not found.')
        
        # create array of targs
        ys_train = np.transpose(ys_train)
        ys_test = np.transpose(ys_test)
        
        mvalid_train_y = ~np.any(np.isnan(ys_train), axis=1)
        mvalid_test_y = ~np.any(np.isnan(ys_test), axis=1)
        
        # loop over neurons
        for ccol_x, col_x in enumerate(cols_x):
            if verbose and ((ccol_x % 15) == 0):  sys.stdout.write('.')
                
            mvalid_train_x = ~np.isnan(xs_train[:, ccol_x])
            mvalid_test_x = ~np.isnan(xs_test[:, ccol_x])
            
            mvalid_train = (mvalid_train_x & mvalid_train_y)
            mvalid_test = (mvalid_test_x & mvalid_test_y)
            
            rgr = linear_model.LinearRegression().fit(xs_train[:, [ccol_x]][mvalid_train, :], ys_train[mvalid_train, :])

            y_hats_train = np.nan*np.zeros(ys_train.shape)
            y_hats_test = np.nan*np.zeros(ys_test.shape)
            
            y_hats_train[mvalid_train, :] = rgr.predict(xs_train[:, [ccol_x]][mvalid_train, :])
            y_hats_test[mvalid_test, :] = rgr.predict(xs_test[:, [ccol_x]][mvalid_test, :])

            for ctarg, targ in enumerate(targs):
                rslt.r2_train[targ][csplit, ccol_x] = get_r2(ys_train[:, ctarg], y_hats_train[:, ctarg])
                rslt.r2_test[targ][csplit, ccol_x] = get_r2(ys_test[:, ctarg], y_hats_test[:, ctarg])

                rslt.w[targ][csplit, ccol_x] = rgr.coef_[ctarg, 0]
                rslt.bias[targ][csplit, ccol_x] = rgr.intercept_[ctarg]
                
    return rslt


def skl_fit_ridge_add_col(pfxs, cols_x, cols_x_fixed, targs, itr_all, ntrain, nsplit, mask_pfx=None, alpha=10, seed=0, verbose=False, **kwargs):
    """Use scikit-learn to fit a linear model, but looping over new single columns as predictors given a fixed set of columns as pre-existing predictors."""
    ncol_x = len(cols_x)
    
    # load all data
    print('\nLoading...')
    
    # main data frames with surrogate neural recordings and basic behav quantities
    dfs_0 = {itr: np.load(f'{pfxs[0]}_tr_{itr}.npy', allow_pickle=True)[0]['df'] for itr in itr_all}
    # corresponding data frames with extended behavioral quantities
    dfs_1 = {itr: pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_all}
    
    if mask_pfx is not None:
        masks = {itr: np.load(f'{mask_pfx}_tr_{itr}.npy', allow_pickle=True)[0]['mask'] for itr in itr_all}
    
    # loop over splits
    np.random.seed(seed)
    
    rslt = Generic(
        cols_x_fixed=cols_x_fixed,
        r2_train={targ: np.nan*np.zeros((nsplit, ncol_x)) for targ in targs},
        r2_test={targ: np.nan*np.zeros((nsplit, ncol_x)) for targ in targs},
        w={targ: np.nan*np.zeros((nsplit, ncol_x)) for targ in targs},
        bias={targ: np.nan*np.zeros((nsplit, ncol_x)) for targ in targs})
    
    sys.stdout.write('Splits:')
    for csplit in range(nsplit):
        sys.stdout.write('X')
        
        irnd = np.random.permutation(len(itr_all))
        
        # training trials
        itr_train = itr_all[irnd[:ntrain]]
        
        dfs_train_0 = [dfs_0[itr] for itr in itr_train]
        dfs_train_1 = [dfs_1[itr] for itr in itr_train]
    
        # test trials
        itr_test = itr_all[irnd[ntrain:]]
        
        dfs_test_0 = [dfs_0[itr] for itr in itr_test]
        dfs_test_1 = [dfs_1[itr] for itr in itr_test]
        
        cols_0 = dfs_train_0[0].columns
        cols_1 = dfs_train_1[0].columns
    
        xs_train_ = [np.array(df_train[cols_x]) for df_train in dfs_train_0]
        xs_test_ = [np.array(df_test[cols_x]) for df_test in dfs_test_0]
        
        if mask_pfx is not None:
            masks_train = [masks[itr] for itr in itr_train]
            masks_test = [masks[itr] for itr in itr_test]
            
            for ctr_train, mask_train in enumerate(masks_train):
                xs_train_[ctr_train][~mask_train, :] = np.nan
            
            for ctr_test, mask_test in enumerate(masks_test):
                xs_test_[ctr_test][~mask_test, :] = np.nan
                
        xs_train = cc(xs_train_)
        xs_test = cc(xs_test_)
        
        mvalid_train_x = ~np.any(np.isnan(xs_train), axis=1)
        mvalid_test_x = ~np.any(np.isnan(xs_test), axis=1)
        
        for targ in targs:
            
            if targ in cols_0:
                ys_train = cc([np.array(df_train[targ]) for df_train in dfs_train_0])
                ys_test = cc([np.array(df_test[targ]) for df_test in dfs_test_0])
                
            elif targ in cols_1:
                ys_train = cc([np.array(df_train[targ]) for df_train in dfs_train_1])
                ys_test = cc([np.array(df_test[targ]) for df_test in dfs_test_1])
                
            else:
                raise KeyError(f'Column with label "{targ}" not found.')
        
            mvalid_train_y = ~np.isnan(ys_train)
            mvalid_test_y = ~np.isnan(ys_test)
            
            mvalid_train = (mvalid_train_x & mvalid_train_y)
            mvalid_test = (mvalid_test_x & mvalid_test_y)
            
            cols_x_fixed_int = [int(col_x[2:]) for col_x in cols_x_fixed[targ]]  # since originally written e.g. 'R_23'
        
            # loop over neurons
            for ccol_x, col_x in enumerate(cols_x):
                if verbose and ((ccol_x % 15) == 0):  sys.stdout.write('.')

                fit_cols_x = cols_x_fixed_int + [ccol_x]
                rgr = linear_model.Ridge(alpha=alpha).fit(xs_train[:, fit_cols_x][mvalid_train, :], ys_train[mvalid_train])

                y_hats_train = np.nan*np.zeros(len(ys_train))
                y_hats_test = np.nan*np.zeros(len(ys_test))

                y_hats_train[mvalid_train] = rgr.predict(xs_train[:, fit_cols_x][mvalid_train, :])
                y_hats_test[mvalid_test] = rgr.predict(xs_test[:, fit_cols_x][mvalid_test, :])

                rslt.r2_train[targ][csplit, ccol_x] = get_r2(ys_train, y_hats_train)
                rslt.r2_test[targ][csplit, ccol_x] = get_r2(ys_test, y_hats_test)

                rslt.w[targ][csplit, ccol_x] = rgr.coef_[-1]
                rslt.bias[targ][csplit, ccol_x] = rgr.intercept_
                
    return rslt
