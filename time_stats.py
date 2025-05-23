from copy import deepcopy
import numpy as np
from scipy import signal, stats
from sklearn.linear_model import LinearRegression

from aux import get_seg


def f_test(rss_reduced, rss_full, df_reduced, df_full, n):
    """
    Calculate the F-statistic between a full and nested model.
    :param rss_reduced: residual sum of squares (RSS) for reduced model
    :param rss_full: RSS for full model
    :param df_reduced: degrees of freedom for reduced model
    :param df_full: degrees of freedom for full model
    :param n: number of data points
    :return: F, p-val
    """

    num = (rss_reduced - rss_full) / (df_full - df_reduced)
    denom = rss_full / (n - df_full)

    f = num / denom

    p_val = 1 - stats.f.cdf(f, df_full - df_reduced, n - df_full)

    return f, p_val


def pearsonr_with_confidence(x, y, confidence=0.95):
    """
    Calculate the pearson correlation coefficient, its p-value, and upper and
        lower 95% confidence bound.
    :param x: one array
    :param y: other array
    :param confidence: how confident the confidence interval
    :return: correlation, p-value, lower confidence bound, upper confidence bound
    """

    rho, p = stats.pearsonr(x, y)
    n = len(x)

    # calculate confidence interval on correlation
    # how confident do we want to be?
    n_sds = stats.norm.ppf(1 - (1 - confidence) / 2)
    z = 0.5 * np.log((1 + rho) / (1 - rho)) if rho != 1 else np.nan  # convert to z-space
    sd = np.sqrt(1. / (n - 3))
    lb_z = z - n_sds * sd
    ub_z = z + n_sds * sd
    # convert back to rho-space
    lb = (np.exp(2*lb_z) - 1) / (np.exp(2*lb_z) + 1)
    ub = (np.exp(2*ub_z) - 1) / (np.exp(2*ub_z) + 1)

    return rho, p, lb, ub


def cov_with_confidence(x, y, confidence=0.95):
    """
    Calculate the covariance of two variables, its p-value, and upper and
        lower 95% confidence bound.
    :param x: one array
    :param y: other array
    :param confidence: how confident the confidence interval
    """

    # ignore nans
    nan_mask = np.any(np.isnan(np.array([x, y])), axis=0)
    x_no_nans = x[~nan_mask]
    y_no_nans = y[~nan_mask]

    cov = np.cov(x_no_nans, y_no_nans)[0, 1]
    corr, pv, lb, ub = pearsonr_with_confidence(x_no_nans, y_no_nans, confidence)

    scale_factor = cov / corr

    return cov, pv, lb * scale_factor, ub * scale_factor


def xcov_multi_with_confidence(
        xs, ys, lag_backward, lag_forward,
        confidence=0.95, pre_norm=True, scale=False):
    """
    Calculate cross-covariance between x and y for multiple time-series.
    Positive lags correspond to covariances between x at an earlier time
    and y at a later time.
    :param xs: list of input time-series
    :param ys: list of output time-series
    :param lag_forward: number of lags to look forward (causal (x yields y))
    :param lag_backward: number of lags to look back (acausal (y yields x))
    :param confidence: confidence of confidence interval desired
    :param pre_norm: if True, xs and ys will be initially scaled to have mean 0 and std 1
    :param scale: if True, results will be scaled by geometric mean of x's & y's variances
    :return: cross-covariance, p-value, lower bound, upper bound
    """

    xs = deepcopy(xs)
    ys = deepcopy(ys)

    if not np.all([len(x) == len(y) for x, y in zip(xs, ys)]):

        raise ValueError('Arrays within xs and ys must all be of the same size!')

    if pre_norm:

        x_mean = np.nanmean(np.concatenate(xs))
        xs = [x - x_mean for x in xs]
        x_std = np.nanstd(np.concatenate(xs))
        xs = [x / x_std for x in xs]

        y_mean = np.nanmean(np.concatenate(ys))
        ys = [y - y_mean for y in ys]
        y_std = np.nanstd(np.concatenate(ys))
        ys = [y / y_std for y in ys]

    covs = []
    p_values = []
    lbs = []
    ubs = []

    for lag in range(-lag_backward, lag_forward):
        try:
            if lag == 0:

                x_rel = xs
                y_rel = ys

            elif lag < 0:

                x_rel = [x[-lag:] for x in xs if len(x) > -lag]
                y_rel = [y[:lag] for y in ys if len(y) > -lag]

            elif lag > 0:

                x_rel = [x[:-lag] for x in xs if len(x) > lag]
                y_rel = [y[lag:] for y in ys if len(y) > lag]

            all_xs = np.concatenate(x_rel)
            all_ys = np.concatenate(y_rel)

            cov, p_value, lb, ub = cov_with_confidence(all_xs, all_ys, confidence)
        except:
            cov, p_value, lb, ub = np.nan, np.nan, np.nan, np.nan

        covs.append(cov)
        p_values.append(p_value)
        lbs.append(lb)
        ubs.append(ub)

    covs = np.array(covs)
    p_values = np.array(p_values)
    lbs = np.array(lbs)
    ubs = np.array(ubs)

    if scale:

        # scale by average variance of signals
        var_x = np.nanvar(np.concatenate(xs))
        var_y = np.nanvar(np.concatenate(ys))
        norm_factor = np.sqrt(var_x * var_y)
        covs /= norm_factor
        lbs /= norm_factor
        ubs /= norm_factor

    return covs, p_values, lbs, ubs


def xcov_with_confidence(
        x, y, lag_backward, lag_forward, confidence=0.95, pre_norm=True, scale=False):
    """
    Calculate the time-lagged cross-covariance function between x and y.
    """

    return xcov_multi_with_confidence(
        xs=[x], ys=[y], lag_backward=lag_backward, lag_forward=lag_forward,
        confidence=confidence, pre_norm=pre_norm, scale=scale)


def partial_corr(x, y, controls):
    """
    Calculate the partial correlation of x and y conditioned on a set of
    control variables. This is calculated by first determining
    :param x: 1D array
    :param y: 1D array
    :param controls: list of 1D arrays
    :return: partial correlation coefficient, p-value
    """

    for control in controls: assert len(control) == len(x) == len(y)

    control_array = np.array(controls).T
    residuals = []

    for v in [x, y]:

        lrg = LinearRegression()
        lrg.fit(control_array, v)
        residuals.append(v - lrg.predict(control_array))

    return stats.pearsonr(residuals[0], residuals[1])


def nan_detrend(x):
    t = np.arange(len(x))
    mvalid = ~np.isnan(x)
    slp, icpt, r, pv, stderr = stats.linregress(t[mvalid], x[mvalid])
    return x - (slp*t + icpt)


def nan_cov(x, y):
    """Compute covariance of two vectors ignoring nans."""
    mvalid = (~np.isnan(x)) & (~np.isnan(y))
    
    return np.cov(np.array([x[mvalid], y[mvalid]]))[0, 1]


def xcov_conv_tri(x, y):
    """
    Estimate cross-covariance of two signals using scipy.signal.correlate and then
    subsequently dividing by triangle function to account for time-lag-dependent
    differences in # samples going into cov estimate. This is useful if you expect
    slow timescales in the autocovariance function, since it is a more unbiased
    estimate of those timescales, but it can also lead to singular covariance matrices.
    """
    x_mn = np.nanmean(x)
    y_mn = np.nanmean(y)
    
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        # fft no work if nans
        xcov_temp = signal.correlate(x - x_mn, y - y_mn, mode='full', method='direct')
    else:
        xcov_temp = signal.correlate(x - x_mn, y - y_mn, mode='full')
        
    x_1 = np.ones(len(x))
    x_1[np.isnan(x)] = 0
    
    y_1 = np.ones(len(y))
    y_1[np.isnan(y)] = 0
    
    tri = signal.correlate(x_1, y_1, mode='full')
    tcov = np.arange(-len(x)+1, len(x))
    
    return xcov_temp/tri, tcov


def xcov_conv_standard(x, y):
    """
    Estimate cross-covariance of two signals using scipy.signal.correlate .
    """
    x_mn = np.nanmean(x)
    y_mn = np.nanmean(y)
    
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        # fft no work if nans
        xcov_temp = signal.correlate(x - x_mn, y - y_mn, mode='full', method='direct')
    else:
        xcov_temp = signal.correlate(x - x_mn, y - y_mn, mode='full')
        
    tcov = np.arange(-len(x)+1, len(x))
    
    return xcov_temp/len(x), tcov


def csd_multi(xs, ys, dt, df, min_dur=0):
    """Compute csd from multiple, variable length samples of x, y time-series."""
    f_out = np.arange(0, 1/(2*dt), df)  # output freq vector
    
    # get signal chunks w no nans
    xs_valid = []
    ys_valid = []

    imin_dur = int(round(min_dur/dt))
    
    for x, y in zip(xs, ys):
        mvalid = ( (~np.isnan(x)) & (~np.isnan(y)) )
        segs, bds = get_seg(mvalid, min_gap=0)
        
        for lb, ub in bds:
            if (ub - lb) >= imin_dur:
                xs_valid.append(x[lb:ub])
                ys_valid.append(y[lb:ub])
                
    p_xys = []
    
    p_xxs = []
    p_yys = []
    
    for x, y in zip(xs_valid, ys_valid):
        f, p_xy = signal.csd(x, y, 1/dt, nperseg=len(x))
        f, p_xx = signal.csd(x, x, 1/dt, nperseg=len(x))
        f, p_yy = signal.csd(y, y, 1/dt, nperseg=len(x))
    
        p_xys.append(np.interp(f_out, f, p_xy))
        p_xxs.append(np.interp(f_out, f, p_xx))
        p_yys.append(np.interp(f_out, f, p_yy))
        
    w = np.array([len(x) for x in xs_valid])
    w = w/w.sum()
    
    p_xy_hat = w @ np.array(p_xys)
    p_xx_hat = w @ np.array(p_xxs)
    p_yy_hat = w @ np.array(p_yys)
    
    return f_out, np.abs(p_xy_hat)**2


def coh_multi(xs, ys, dt, df, min_dur=0):
    """Compute coherence from multiple, variable length samples of x, y time-series."""
    f_out = np.arange(0, 1/(2*dt), df)  # output freq vector
    
    # get signal chunks w no nans
    xs_valid = []
    ys_valid = []

    imin_dur = int(round(min_dur/dt))
    
    for x, y in zip(xs, ys):
        mvalid = ( (~np.isnan(x)) & (~np.isnan(y)) )
        segs, bds = get_seg(mvalid, min_gap=0)
        
        for lb, ub in bds:
            if (ub - lb) >= imin_dur:
                xs_valid.append(x[lb:ub])
                ys_valid.append(y[lb:ub])
                
    p_xys = []
    
    p_xxs = []
    p_yys = []
    
    for x, y in zip(xs_valid, ys_valid):
        f, p_xy = signal.csd(x, y, 1/dt, nperseg=len(x))
        f, p_xx = signal.csd(x, x, 1/dt, nperseg=len(x))
        f, p_yy = signal.csd(y, y, 1/dt, nperseg=len(x))
    
        p_xys.append(np.interp(f_out, f, p_xy))
        p_xxs.append(np.interp(f_out, f, p_xx))
        p_yys.append(np.interp(f_out, f, p_yy))
        
    w = np.array([len(x) for x in xs_valid])
    w = w/w.sum()
    
    p_xy_hat = w @ np.array(p_xys)
    p_xx_hat = w @ np.array(p_xxs)
    p_yy_hat = w @ np.array(p_yys)
    
    return f_out, np.abs(p_xy_hat)**2/(p_xx_hat*p_yy_hat)
