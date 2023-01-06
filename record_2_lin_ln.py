"""Functions for recording surrogate female neural activity in response to song."""
import numpy as np
from scipy import signal

cc = np.concatenate


def smlt_lin(i_s, i_p, params, dt):
    """Linear filter."""
    h_ss = params['H_SS']  # time points X neurons
    h_ps = params['H_PS']
    
    t_h = np.arange(len(h_ss))
    n = h_ss.shape[1]
    
    # convolve filters with input currents
    rs = \
        + dt*signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + dt*signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :]
    
    return rs


# LN: linear-nonlinear neurons
def smlt_ln(i_s, i_p, params, dt):
    """Linear-nonlinear neuron with sigmoidal nonlin."""
    h_ss = params['H_SS']  # time points X neurons
    h_ps = params['H_PS']
    
    t_h = np.arange(len(h_ss))
    n = h_ss.shape[1]
    
    # nonlinearity params
    r_mins = params['R_MIN']
    r_maxs = params['R_MAX']
    z_0s = params['Z_0']
    betas = params['BETA']
    
    # convolve filters with input currents
    zs = \
        + dt*signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + dt*signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :]
    
    rs = r_mins + (r_maxs-r_mins)*(np.tanh(betas*(zs-z_0s)) + 1)/2
    
    return rs


def smlt_ln_relu(i_s, i_p, params, dt):
    """Linear-nonlinear neuron with signed relu nonlin."""
    h_ss = params['H_SS']  # time points X neurons
    h_ps = params['H_PS']
    
    t_h = np.arange(len(h_ss))
    n = h_ss.shape[1]
    
    # nonlinearity params
    sgns = params['SGNS']
    
    # convolve filters with input currents
    zs = \
        + dt*signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + dt*signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :]
    
    rs = zs.copy()
    rs[:, sgns > 0] = np.clip(rs[:, sgns > 0], 0, np.inf)
    rs[:, sgns < 0] = np.clip(rs[:, sgns < 0], -np.inf, 0)
    
    return rs


def smlt_ln_relu_flex(i_s, i_p, params, dt):
    """Linear-nonlinear neuron with piece-wise rectification."""
    h_ss = params['H_SS']
    h_ps = params['H_PS']
    
    t_h = np.arange(len(h_ss))
    n = h_ss.shape[1]
    
    # nonlin params
    b_ms = params['B_M']
    b_ps = params['B_P']
    
    # convolve filters with input currents
    zs = \
        + dt*signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + dt*signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :]
    
    rs = zs.copy()
    
    # multiply all neg responses by b_minus vals for each nrn
    mneg = rs < 0
    rs[mneg] *= np.tile(b_ms[None, :], (len(rs), 1))[mneg]
    
    # multiply all pos responses by b_plus vals for each nrn
    mpos = rs >= 0
    rs[mpos] *= np.tile(b_ps[None, :], (len(rs), 1))[mpos]
    
    return rs
    

# lin and ln simulations with params fit using ridge regression
## (same funcs, since only the fitting procedure is different)
smlt_linr = smlt_lin
smlt_lnr = smlt_ln
smlt_lnr_relu = smlt_ln_relu
smlt_lnr_relu_flex = smlt_ln_relu_flex
smlt_lnma_tweaked = smlt_ln_relu


# LN: linear-nonlinear neurons where each filter is sum of two exponentials derived from MA fit
def smlt_lnma_tweaked(i_s, i_p, params, dt):
    """Linear-nonlinear neuron with double exponential filter and sigmoid nonlin."""
    x_s_0s = np.array(params['X_S_0'])
    tau_s_0s = np.array(params['TAU_S_0'])
    x_s_1s = np.array(params['X_S_1'])
    tau_s_1s = np.array(params['TAU_S_1'])
    x_p_0s = np.array(params['X_P_0'])
    tau_p_0s = np.array(params['TAU_P_0'])
    x_p_1s = np.array(params['X_P_1'])
    tau_p_1s = np.array(params['TAU_P_1'])
    sgns = np.array(params['SGN'])
    
    # nonlinearity params
    t_h = np.arange(0, 600, dt)
    
    h_ss = np.array([
        x_s_0*np.exp(-t_h/tau_s_0) + x_s_1*np.exp(-t_h/tau_s_1) 
        for x_s_0, tau_s_0, x_s_1, tau_s_1 in zip(x_s_0s, tau_s_0s, x_s_1s, tau_s_1s)
    ]).T
    
    h_ps = np.array([
        x_p_0*np.exp(-t_h/tau_p_0) + x_p_1*np.exp(-t_h/tau_p_1)
        for x_p_0, tau_p_0, x_p_1, tau_p_1 in zip(x_p_0s, tau_p_0s, x_p_1s, tau_p_1s)
    ]).T
    
    # convolve filters with input currents
    zs = \
        + dt*signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + dt*signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :]
    
    # pass through nonlin
    rs = zs.copy()
    rs[:, sgns > 0] = np.clip(rs[:, sgns > 0], 0, np.inf)
    rs[:, sgns < 0] = np.clip(rs[:, sgns < 0], -np.inf, 0)
    
    return rs
