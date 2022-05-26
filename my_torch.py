import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d as smooth
from scipy import stats
from sklearn import linear_model
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from aux import Generic, load_npy, split
from disp import set_plot
from my_stats import get_r2

cc = np.concatenate


def skl_fit_ridge(pfxs, cols_x, targs, itr_all, ntrain, nsplit, return_y=None, alpha=10, verbose=True, seed=0, **kwargs):
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
        
    # load all data
    if verbose:  print('Loading...')
    
    # main data frames with surrogate neural recordings and basic behav quantities
    dfs_0 = {itr: np.load(f'{pfxs[0]}_tr_{itr}.npy', allow_pickle=True)[0]['df'] for itr in itr_all}
    # corresponding data frames with extended behavioral quantities
    dfs_1 = {itr: pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_all}
    
    # loop over splits
    np.random.seed(seed)
    
    rslts = []
    for csplit in range(nsplit):
        if verbose:  sys.stdout.write(f'\nSplit {csplit}')
        
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

        xs_train = cc([np.array(df_train[cols_x]) for df_train in dfs_train_0])
        xs_test = cc([np.array(df_test[cols_x]) for df_test in dfs_test_0])
        
        rslt = Generic(
            alpha=alpha, targs=targs,
            w={}, bias={},
            ys_train={}, y_hats_train={}, rms_err_train={}, r2_train={}, 
            ys_test={}, y_hats_test={}, rms_err_test={}, r2_test={})

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
        
        mvalid_train = ~np.any(np.isnan(ys_train), axis=1)
        mvalid_test = ~np.any(np.isnan(ys_test), axis=1)
        
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

            # store extra vars (if specified)
            if csplit in return_y:
                rslt.ys_train[targ] = split(ys_train[:, ctarg], its_start_train, its_end_train)
                rslt.ys_test[targ] = split(ys_test[:, ctarg], its_start_test, its_end_test)

                rslt.y_hats_train[targ] = split(y_hats_train[:, ctarg], its_start_train, its_end_train)
                rslt.y_hats_test[targ] = split(y_hats_test[:, ctarg], its_start_test, its_end_test)
                
        rslts.append(rslt)
        
    return rslts


def skl_fit_lin_single(pfxs, cols_x, targs, itr_all, ntrain, nsplit, seed=0, verbose=False, **kwargs):
    """Use scikit-learn to fit a linear model, but looping over single columns as predictors."""
    ncol_x = len(cols_x)
    
    # load all data
    print('\nLoading...')
    
    # main data frames with surrogate neural recordings and basic behav quantities
    dfs_0 = {itr: np.load(f'{pfxs[0]}_tr_{itr}.npy', allow_pickle=True)[0]['df'] for itr in itr_all}
    # corresponding data frames with extended behavioral quantities
    dfs_1 = {itr: pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_all}
    
    # loop over splits
    np.random.seed(seed)
    
    rslt = Generic(
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
    
        xs_train = cc([np.array(df_train[cols_x]) for df_train in dfs_train_0])
        xs_test = cc([np.array(df_test[cols_x]) for df_test in dfs_test_0])
        
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
        
        mvalid_train = ~np.any(np.isnan(ys_train), axis=1)
        mvalid_test = ~np.any(np.isnan(ys_test), axis=1)
        
        # loop over neurons
        for ccol_x, col_x in enumerate(cols_x):
            if verbose and ((ccol_x % 15) == 0):  sys.stdout.write('.')
                
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


def skl_fit_ridge_add_col(pfxs, cols_x, cols_x_fixed, targs, itr_all, ntrain, nsplit, alpha=10, seed=0, verbose=False, **kwargs):
    """Use scikit-learn to fit a linear model, but looping over new single columns as predictors given a fixed set of columns as pre-existing predictors."""
    ncol_x = len(cols_x)
    
    # load all data
    print('\nLoading...')
    
    # main data frames with surrogate neural recordings and basic behav quantities
    dfs_0 = {itr: np.load(f'{pfxs[0]}_tr_{itr}.npy', allow_pickle=True)[0]['df'] for itr in itr_all}
    # corresponding data frames with extended behavioral quantities
    dfs_1 = {itr: pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_all}
    
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
    
        xs_train = cc([np.array(df_train[cols_x]) for df_train in dfs_train_0])
        xs_test = cc([np.array(df_test[cols_x]) for df_test in dfs_test_0])
        
        for targ in targs:
            
            if targ in cols_0:
                ys_train = cc([np.array(df_train[targ]) for df_train in dfs_train_0])
                ys_test = cc([np.array(df_test[targ]) for df_test in dfs_test_0])
                
            elif targ in cols_1:
                ys_train = cc([np.array(df_train[targ]) for df_train in dfs_train_1])
                ys_test = cc([np.array(df_test[targ]) for df_test in dfs_test_1])
                
            else:
                raise KeyError(f'Column with label "{targ}" not found.')
        
            mvalid_train = ~np.isnan(ys_train)
            mvalid_test = ~np.isnan(ys_test)
            
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


# # OLD VERSION
# def skl_fit_ridge_(pfxs, cols_x, targs, itr_train, itr_test, alpha=10, return_y=False, verbose=True, **kwargs):
#     """Use scikit-learn to fit a linear model using ridge regression."""
#     # load all data
#     print('Loading...')
#     dfs_train_0 = [np.load(f'{pfxs[0]}_tr_{itr}.npy', allow_pickle=True)[0]['df'] for itr in itr_train]
#     dfs_test_0 = [np.load(f'{pfxs[0]}_tr_{itr}.npy', allow_pickle=True)[0]['df'] for itr in itr_test]
    
#     dfs_train_1 = [pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_train]
#     dfs_test_1 = [pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_test]
    
#     cols_0 = dfs_train_0[0].columns
#     cols_1 = dfs_train_1[0].columns
    
#     xs_train = [np.array(df_train[cols_x]) for df_train in dfs_train_0]
#     xs_test = [np.array(df_test[cols_x]) for df_test in dfs_test_0]
    
#     rslt = Generic(
#         alpha=alpha,
#         r2_train={}, r2_test={}, w={}, bias={}, rms_err_train={}, rms_err_test={},
#         ys_train={}, ys_test={}, y_hats_train={}, y_hats_test={}, mvalids_train={}, mvalids_test={})
    
#     print('Fitting...') 
#     for targ in targs:
#         if verbose:  sys.stdout.write('>')
        
#         if targ in cols_0:
#             ys_train = [np.array(df_train[targ]) for df_train in dfs_train_0]
#             ys_test = [np.array(df_test[targ]) for df_test in dfs_test_0]
#         elif targ in cols_1:
#             ys_train = [np.array(df_train[targ]) for df_train in dfs_train_1]
#             ys_test = [np.array(df_test[targ]) for df_test in dfs_test_1]
#         else:
#             raise KeyError(f'Column with label "{targ}" not found.')

#         mvalids_train = [~np.isnan(y_train) for y_train in ys_train]
#         mvalids_test = [~np.isnan(y_test) for y_test in ys_test]
    
#         xs_fit = cc(xs_train)[cc(mvalids_train), :]
#         y_fit = cc(ys_train)[cc(mvalids_train)]
        
#         # regress
#         rgr = linear_model.Ridge(alpha=alpha).fit(xs_fit, y_fit)
        
#         # store basic vars
#         rslt.w[targ] = rgr.coef_
#         rslt.bias[targ] = rgr.intercept_
        
#         rslt.r2_train[targ] = rgr.score(cc(xs_train)[cc(mvalids_train), :], cc(ys_train)[cc(mvalids_train)])
#         rslt.r2_test[targ] = rgr.score(cc(xs_test)[cc(mvalids_test), :], cc(ys_test)[cc(mvalids_test)])
        
#         # store extra vars (if specified)
#         if return_y:
#             rslt.ys_train[targ] = ys_train
#             rslt.ys_test[targ] = ys_test

#             rslt.mvalids_train[targ] = mvalids_train
#             rslt.mvalids_test[targ] = mvalids_test

#             y_hats_train = [rgr.predict(x_train[mvalid_train, :]) for x_train, mvalid_train in zip(xs_train, mvalids_train)]
#             y_hats_test = [rgr.predict(x_test[mvalid_test, :]) for x_test, mvalid_test in zip(xs_test, mvalids_test)]
            
#             rslt.y_hats_train[targ] = y_hats_train
#             rslt.y_hats_test[targ] = y_hats_test

#             rslt.rms_err_train[targ] = np.mean((cc(ys_train)[cc(mvalids_train)] - cc(y_hats_train))**2)
#             rslt.rms_err_test[targ] = np.mean((cc(ys_test)[cc(mvalids_test)] - cc(y_hats_test))**2)
       
#     return rslt


# def skl_fit_lin_single_(pfxs, cols_x, targs, itr_train, itr_test, **kwargs):
#     """Use scikit-learn to fit a linear model, but looping over single columns as predictors."""
#     # load all data
#     print('Loading...')
#     dfs_train_0 = [load_npy(f'{pfxs[0]}_tr_{itr}.npy')['df'] for itr in itr_train]
#     dfs_test_0 = [load_npy(f'{pfxs[0]}_tr_{itr}.npy')['df'] for itr in itr_test]
    
#     dfs_train_1 = [pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_train]
#     dfs_test_1 = [pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_test]
    
#     cols_0 = dfs_train_0[0].columns
#     cols_1 = dfs_train_1[0].columns
    
#     ncol_x = len(cols_x)
    
#     xs_train = [np.array(df_train[cols_x]) for df_train in dfs_train_0]
#     xs_test = [np.array(df_train[cols_x]) for df_train in dfs_test_0]
    
#     rslt = Generic(r2_train={}, r2_test={}, w={}, bias={})
    
#     for targ in targs:
#         print(f'Fitting {targ}')
#         if targ in cols_0:
#             ys_train = [np.array(df_train[targ]) for df_train in dfs_train_0]
#             ys_test = [np.array(df_test[targ]) for df_test in dfs_test_0]
#         elif targ in cols_1:
#             ys_train = [np.array(df_train[targ]) for df_train in dfs_train_1]
#             ys_test = [np.array(df_test[targ]) for df_test in dfs_test_1]
#         else:
#             raise KeyError(f'Column with label "{targ}" not found.')

#         mvalids_train = [~np.isnan(y_train) for y_train in ys_train]
#         mvalids_test = [~np.isnan(y_test) for y_test in ys_test]
    
#         r2_train = np.nan*np.zeros(ncol_x)
#         r2_test = np.nan*np.zeros(ncol_x)
#         w = np.nan*np.zeros(ncol_x)
#         bias = np.nan*np.zeros(ncol_x)
        
#         for ccol_x, col_x in enumerate(cols_x):
#             sys.stdout.write('.')
#             xs_fit = cc(xs_train)[:, [ccol_x]][cc(mvalids_train), :]
#             y_fit = cc(ys_train)[cc(mvalids_train)]
            
#             rgr = linear_model.LinearRegression().fit(xs_fit, y_fit)

#             r2_train[ccol_x] = rgr.score(xs_fit, y_fit)
#             r2_test[ccol_x] = rgr.score(cc(xs_test)[:, [ccol_x]][cc(mvalids_test), :], cc(ys_test)[cc(mvalids_test)])
            
#             w[ccol_x] = rgr.coef_[0]
#             bias[ccol_x] = rgr.intercept_
            
#         rslt.r2_train[targ] = r2_train.copy()
#         rslt.r2_test[targ] = r2_test.copy()
#         rslt.w[targ] = w.copy()
#         rslt.bias[targ] = bias.copy()

#     return rslt


# def skl_fit_lin_(pfxs, cols_x, targs, itr_train, itr_test, return_y=False, verbose=True, **kwargs):
#     """Use scikit-learn to fit a linear model."""
        
#     # load all data
#     print('Loading...')
#     dfs_train_0 = [load_npy(f'{pfxs[0]}_tr_{itr}.npy')['df'] for itr in itr_train]
#     dfs_test_0 = [load_npy(f'{pfxs[0]}_tr_{itr}.npy')['df'] for itr in itr_test]
    
#     dfs_train_1 = [pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_train]
#     dfs_test_1 = [pd.read_csv(f'{pfxs[1]}_tr_{itr}.csv') for itr in itr_test]
    
#     xs_train = [np.array(df_train[cols_x]) for df_train in dfs_train_0]
#     xs_test = [np.array(df_test[cols_x]) for df_test in dfs_test_0]
    
#     cols_0 = dfs_train_0[0].columns
#     cols_1 = dfs_train_1[0].columns
    
#     rslt = Generic(
#         r2_train={}, r2_test={}, w={}, bias={}, rms_err_train={}, rms_err_test={},
#         ys_train={}, ys_test={}, y_hats_train={}, y_hats_test={}, mvalids_train={}, mvalids_test={})
    
#     print('Fitting...')
#     for targ in targs:
#         if verbose:  sys.stdout.write('>')
        
#         if targ in cols_0:
#             ys_train = [np.array(df_train[targ]) for df_train in dfs_train_0]
#             ys_test = [np.array(df_test[targ]) for df_test in dfs_test_0]
#         elif targ in cols_1:
#             ys_train = [np.array(df_train[targ]) for df_train in dfs_train_1]
#             ys_test = [np.array(df_test[targ]) for df_test in dfs_test_1]
#         else:
#             raise KeyError(f'Column with label "{targ}" not found.')

#         mvalids_train = [~np.isnan(y_train) for y_train in ys_train]
#         mvalids_test = [~np.isnan(y_test) for y_test in ys_test]
        
#         xs_fit = cc(xs_train)[cc(mvalids_train), :]
#         y_fit = cc(ys_train)[cc(mvalids_train)]
        
#         # regress
#         rgr = linear_model.LinearRegression().fit(xs_fit, y_fit)

#         # store basic vars
#         rslt.w[targ] = rgr.coef_
#         rslt.bias[targ] = rgr.intercept_
        
#         rslt.r2_train[targ] = rgr.score(cc(xs_train)[cc(mvalids_train), :], cc(ys_train)[cc(mvalids_train)])
#         rslt.r2_test[targ] = rgr.score(cc(xs_test)[cc(mvalids_test), :], cc(ys_test)[cc(mvalids_test)])
        
#         # store extra vars (if specified)
#         if return_y:
#             rslt.ys_train[targ] = ys_train
#             rslt.ys_test[targ] = ys_test

#             rslt.mvalids_train[targ] = mvalids_train
#             rslt.mvalids_test[targ] = mvalids_test

#             y_hats_train = [rgr.predict(x_train[mvalid_train, :]) for x_train, mvalid_train in zip(xs_train, mvalids_train)]
#             y_hats_test = [rgr.predict(x_test[mvalid_test, :]) for x_test, mvalid_test in zip(xs_test, mvalids_test)]
            
#             rslt.y_hats_train[targ] = y_hats_train
#             rslt.y_hats_test[targ] = y_hats_test

#             rslt.rms_err_train[targ] = np.mean((cc(ys_train)[cc(mvalids_train)] - cc(y_hats_train))**2)
#             rslt.rms_err_test[targ] = np.mean((cc(ys_test)[cc(mvalids_test)] - cc(y_hats_test))**2)

#     return rslt


class NeuralBehavDataset(Dataset):
        
    def __init__(self, pfx, itrs, cols_x, col_y):
        self.pfx = pfx
        self.itrs = itrs
        self.cols_x = cols_x
        self.col_y = col_y

    def __len__(self):
        return len(self.itrs)

    def __getitem__(self, idx):
        # return all data from one trial
        df = np.load(f'{self.pfx}_tr_{self.itrs[idx]}.npy', allow_pickle=True)[0]['df']
        return np.array(df[self.cols_x]), np.array(df[self.col_y])


class NeuralNetwork(nn.Module):
    
    def __init__(self, cols_x):
        super(NeuralNetwork, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(len(cols_x), 1),
        )
        
    def forward(self, x):
        return self.stack(x)

    
def torch_fit_lin(pfx, cols_x, col_y, itrs_train, itrs_test, **kwargs):
    lr = kwargs.get('lr', 1e-5)
    nepoch = kwargs.get('nepoch', 1500)
    print_every = kwargs.get('print_every', 50)
    
    dataset_train = NeuralBehavDataset(pfx, itrs_train, cols_x, col_y)
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    dataset_test = NeuralBehavDataset(pfx, itrs_test, cols_x, col_y)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    for x, y in dataloader_train:
        print(f'First training data shape: {x.shape}, {y.shape}')
        break

    for x, y in dataloader_test:
        print(f'First test data shape: {x.shape}, {y.shape}')
        break
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeuralNetwork(cols_x).to(device)
    
    print(f'Using {device} device')
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    def train(dataloader, model, loss_fn, optimizer):
        model.train()
        size = len(dataloader.dataset)
        for ctr, (x, y) in enumerate(dataloader):
            x, y = x[0, :, :].to(device), y.T.to(device)

            # compute prediction error
            y_hat = model(x.float())
            loss = loss_fn(y_hat, y.float())

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss

    def test(dataloader, model, ctrs_return=None):
        model.eval()
        size = len(dataloader.dataset)

        if ctrs_return is None:
            ctrs_return = []
        elif ctrs_return == 'all':
            ctrs_return = range(size)

        ys = []
        y_hats = []

        with torch.no_grad():
            for ctr, (x, y) in enumerate(dataloader):
                x, y = x[0, :, :].to(device), y.T.to(device)

                # compute predictions
                y_hat = model(x.float())

                if ctr in ctrs_return:
                    ys.append(y.numpy())
                    y_hats.append(y_hat.numpy())

        return ys, y_hats
    
    sys.stdout.write('Loss: ')
    for cepoch in range(nepoch):
        loss = train(dataloader_train, model, loss_fn, optimizer)
        if cepoch == 0 or cepoch % print_every == 0:
            sys.stdout.write(f'{loss:.6f} (E{cepoch+1}), ')
            
    ys_train, y_hats_train = test(dataloader_train, model, ctrs_return='all')
    ys_test, y_hats_test = test(dataloader_test, model, ctrs_return='all')
    
    var_train = np.var(cc(ys_train))
    err_train = np.mean((cc(ys_train) - cc(y_hats_train))**2)
    
    var_test = np.var(cc(ys_test))
    err_test = np.mean((cc(ys_test) - cc(y_hats_test))**2)
    
    rgr = Generic(
        model=model,
        w=list(model.parameters())[0][0].detach().numpy(),
        bias=list(model.parameters())[1].detach().numpy()[0],
        rms_err_train=err_train,
        rms_err_test=err_test,
        r2_train=(var_train - err_train)/var_train,
        r2_test=(var_test - err_test)/var_test,
        ys_train=ys_train,
        y_hats_train=y_hats_train,
        ys_test=ys_test,
        y_hats_test=y_hats_test)
    
    return rgr
