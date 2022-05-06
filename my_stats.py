from copy import deepcopy
import numpy as np
from scipy import stats


def nanpearsonr(x, y):
    """Return pearson correlation after removing nans."""
    valid = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
    return stats.pearsonr(x[valid], y[valid])


def get_r2(y, y_hat):
    return 1 - np.nanmean((y_hat - y)**2)/np.nanvar(y, ddof=0)


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
    z = 0.5 * (np.log(1 + rho) - np.log(1 - rho))  # convert to z-space
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
        xs, ys, lag_b, lag_f,
        confidence=0.95, pre_norm=True, scale=False):
    """
    Calculate cross-covariance between x and y for multiple time-series.
    Positive lags correspond to covariances between x at an earlier time
    and y at a later time.
    :param xs: list of input time-series
    :param ys: list of output time-series
    :param lag_f: number of lags to look forward (causal (x yields y))
    :param lag_b: number of lags to look back (acausal (y yields x))
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

    for lag in range(-lag_b, lag_f):

        if lag == 0:

            x_rel = xs
            y_rel = ys

        elif lag < 0:

            x_rel = [x[-lag:] for x in xs if len(x) > -lag]
            y_rel = [y[:lag] for y in ys if len(y) > -lag]

        else:

            # calculate the cross covariance between x and y with a specific lag
            # first get the relevant xs and ys from each time-series

            x_rel = [x[:-lag] for x in xs if len(x) > lag]
            y_rel = [y[lag:] for y in ys if len(y) > lag]

        all_xs = np.concatenate(x_rel)
        all_ys = np.concatenate(y_rel)

        cov, p_value, lb, ub = cov_with_confidence(all_xs, all_ys, confidence)

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
        x, y, lag_b, lag_f, confidence=0.95, pre_norm=True, scale=False):
    """
    Calculate the time-lagged cross-covariance function between x and y.
    """

    return xcov_multi_with_confidence(
        xs=[x], ys=[y], lag_b=lag_b, lag_f=lag_f,
        confidence=confidence, pre_norm=pre_norm, scale=scale)