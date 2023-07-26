"""Functions for recording surrogate female neural activity in response to song."""
import numpy as np
from scipy import signal


# auxiliary functions
def get_ma_matched_filter(tau_r, tau_a, x_s, x_p, dt):
    """Given a set of MA params, get the equivalent linear/lin-nonlin filter."""
    th = np.arange(0, 600, dt)  # filter time vec
    gam = (1/tau_a) - (1/tau_r)
    
    if np.abs(gam) > 1e-8:
        h_s = (x_s/(tau_r/tau_a - 1))*((-1/tau_r)*np.exp(-th/tau_r) + (1/tau_a)*np.exp(-th/tau_a))
        h_p = (x_p/(tau_r/tau_a - 1))*((-1/tau_r)*np.exp(-th/tau_r) + (1/tau_a)*np.exp(-th/tau_a))
    else:
        h_s = (x_s/tau_r)*(np.exp(-th/tau_a) - (th/tau_a)*np.exp(-th/tau_a))
        h_p = (x_p/tau_r)*(np.exp(-th/tau_a) - (th/tau_a)*np.exp(-th/tau_a))
    
    return h_s, h_p, th


# simulation functions
## based on MA fit
def smlt_ma(i_s, i_p, params, dt):
    """MA: Multiplicative adaptive neuron."""
    tau_rs = params['TAU_R']
    tau_as = params['TAU_A']
    x_ss = params['X_S']
    x_ps = params['X_P']
    
    n = len(tau_rs)
    
    t = np.arange(len(i_s))*dt
    rs = np.nan*np.zeros((len(t), n))
    
    rs[0, :] = 0
    a_s = np.zeros(n)
    a_p = np.zeros(n)
    
    for ct, t_ in enumerate(t[1:], 1):
        a_s += ((dt/tau_as) * (-a_s + i_s[ct]))
        a_p += ((dt/tau_as) * (-a_p + i_p[ct]))
        dr = (dt/tau_rs) * (-rs[ct-1, :] + (1 - a_s)*x_ss*i_s[ct] + (1 - a_p)*x_ps*i_p[ct])
        rs[ct, :] = rs[ct-1, :] + dr
    
    return rs


def smlt_linma(i_s, i_p, params, dt):
    """LINMA: Linear neuron with MA-derived filter."""
    tau_rs = params['TAU_R']
    tau_as = params['TAU_A']
    x_ss = params['X_S']
    x_ps = params['X_P']
    
    n = len(tau_rs)
    
    ## convert MA params to filters
    h_ss_temp = []
    h_ps_temp = []

    for tau_r, tau_a, x_s, x_p in zip(tau_rs, tau_as, x_ss, x_ps):
        h_s, h_p = get_ma_matched_filter(tau_r, tau_a, x_s, x_p, dt)[:2]
        h_ss_temp.append(h_s)
        h_ps_temp.append(h_p)
        
    nt = np.max([len(h_s) for h_s in h_ss_temp])
    
    h_ss = np.zeros((nt, n))
    h_ps = np.zeros((nt, n))

    for cnrn, (h_s, h_p) in enumerate(zip(h_ss_temp, h_ps_temp)):
        h_ss[:len(h_s), cnrn] = h_s
        h_ps[:len(h_p), cnrn] = h_p
        
    # convolve filters with input currents
    rs = \
        + dt*signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + dt*signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :]
    
    return rs
        
    
def smlt_lnma(i_s, i_p, params, dt):
    """LNMA: Linear-nonlinear neuron with MA-derived filter and RELU nonlin."""
    tau_rs = params['TAU_R']
    tau_as = params['TAU_A']
    x_ss = params['X_S']
    x_ps = params['X_P']
    
    n = len(tau_rs)
    
    ## convert MA params to filters
    h_ss_temp = []
    h_ps_temp = []
    signs = []

    for tau_r, tau_a, x_s, x_p in zip(tau_rs, tau_as, x_ss, x_ps):
        h_s, h_p = get_ma_matched_filter(tau_r, tau_a, x_s, x_p, dt)[:2]
        h_ss_temp.append(h_s)
        h_ps_temp.append(h_p)
        
        if np.abs(x_s) > np.abs(x_p):
            signs.append(np.sign(x_s))
        else:
            signs.append(np.sign(x_p))
            
    signs = np.array(signs)
        
    nt = np.max([len(h_s) for h_s in h_ss_temp])
    
    h_ss = np.zeros((nt, n))
    h_ps = np.zeros((nt, n))

    for cnrn, (h_s, h_p) in enumerate(zip(h_ss_temp, h_ps_temp)):
        h_ss[:len(h_s), cnrn] = h_s
        h_ps[:len(h_p), cnrn] = h_p
        
    # convolve filters with input currents
    zs = \
        + dt*signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + dt*signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :]
    
    # rectify
    rs = zs.copy()
    rs[:, signs > 0] = np.clip(rs[:, signs > 0], 0, np.inf)
    rs[:, signs < 0] = np.clip(rs[:, signs < 0], -np.inf, 0)
    
    return rs
