"""Functions for recording surrogate female neural activity in response to song."""
import numpy as np
from scipy import signal

from aux import get_sine_off_cur


# MA: multiplicative adaptation neuron
def smlt_ppln_ma(i_s, i_p, tau_rs, tau_as, x_ss, x_ps, dt):
    n = len(tau_rs)
    
    t = np.arange(len(i_s))*dt
    rs = np.nan*np.zeros((len(t), n))
    
    rs[0, :] = 0
    a_s = np.zeros(n)
    a_p = np.zeros(n)
    
    for ct, t_ in enumerate(t[1:], 1):
        a_s += ((dt/tau_as) * (-a_s + i_s[ct]))
        a_p += ((dt/tau_as) * (-a_p + i_p[ct]))
        rs[ct, :] = rs[ct-1, :] + (dt/tau_rs) * (-rs[ct-1, :] + (1 - a_s)*x_ss*i_s[ct] + (1 - a_p)*x_ps*i_p[ct])
    
    return rs


def smlt_ppln_sia(i_s, i_p, tau_rs, tau_as, x_ss, x_ps, dt):
    n = len(tau_rs)
    
    t = np.arange(len(i_s))*dt
    rs = np.nan*np.zeros((len(t), n))
    
    rs[0, :] = 0
    a = np.zeros(n)
    
    for ct, t_ in enumerate(t[1:], 1):
        a += ((dt/tau_as) * (-a + i_s[ct] + i_p[ct]))
        rs[ct, :] = rs[ct-1, :] + (dt/tau_rs) * (-rs[ct-1, :] + (1 - a)*x_ss*i_s[ct] + (1 - a)*x_ps*i_p[ct])
    
    return rs


# LIN: linear filter neurons
def smlt_ppln_lin(i_s, i_p, h_ss, h_ps, dt):
    """Each column of h_ss or h_ps is one neuron's filter"""
    rs = dt*(signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :])
    
    return rs


# LIN2e: linear filter neurons where each filter is sum of two exponentials
def smlt_ppln_lin2e(i_s, i_p, t_h, x_s_0s, tau_s_0s, x_s_1s, tau_s_1s, x_p_0s, tau_p_0s, x_p_1s, tau_p_1s, dt):
    h_ss = np.array([
        x_s_0*np.exp(-t_h/tau_s_0) + x_s_1*np.exp(-t_h/tau_s_1) 
        for x_s_0, tau_s_0, x_s_1, tau_s_1 in zip(x_s_0s, tau_s_0s, x_s_1s, tau_s_1s)
    ]).T
    
    h_ps = np.array([
        x_p_0*np.exp(-t_h/tau_p_0) + x_p_1*np.exp(-t_h/tau_p_1)
        for x_p_0, tau_p_0, x_p_1, tau_p_1 in zip(x_p_0s, tau_p_0s, x_p_1s, tau_p_1s)
    ]).T
    
    return smlt_ppln_lin(i_s, i_p, h_ss, h_ps, dt)


# LN: linear-nonlinear neurons
def smlt_ppln_ln(i_s, i_p, h_ss, h_ps, r_mins, r_maxs, z_0s, betas, dt):
    """Each column of h_ss or h_ps is one neuron's filter"""
    zs = dt*(signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :])
    
    return r_mins + (r_maxs-r_mins)*(np.tanh(betas*(zs-z_0s)) + 1)/2


# LN: linear-nonlinear neurons where each filter is sum of two exponentials
def smlt_ppln_ln2e(i_s, i_p, t_h, x_s_0s, tau_s_0s, x_s_1s, tau_s_1s, x_p_0s, tau_p_0s, x_p_1s, tau_p_1s, r_mins, r_maxs, z_0s, betas, dt):
    h_ss = np.array([
        x_s_0*np.exp(-t_h/tau_s_0) + x_s_1*np.exp(-t_h/tau_s_1) 
        for x_s_0, tau_s_0, x_s_1, tau_s_1 in zip(x_s_0s, tau_s_0s, x_s_1s, tau_s_1s)
    ]).T
    
    h_ps = np.array([
        x_p_0*np.exp(-t_h/tau_p_0) + x_p_1*np.exp(-t_h/tau_p_1)
        for x_p_0, tau_p_0, x_p_1, tau_p_1 in zip(x_p_0s, tau_p_0s, x_p_1s, tau_p_1s)
    ]).T
    
    return smlt_ppln_ln(i_s, i_p, h_ss, h_ps, r_mins, r_maxs, z_0s, betas, dt)


# LN-RELU
def smlt_ppln_lnrelu(i_s, i_p, h_ss, h_ps, dt, sign):
    zs = dt*(signal.fftconvolve(i_s[:, None], h_ss, mode='full', axes=0)[:len(i_s), :] \
        + signal.fftconvolve(i_p[:, None], h_ps, mode='full', axes=0)[:len(i_p), :])
    
    if sign > 0:
        rs = np.clip(zs, 0, np.inf)
    else:
        rs = np.clip(zs, -np.inf, 0)
    
    return rs


# MA-S-OFF: multiplicative adaptation neurons with sine-offset responses
def smlt_ppln_masoff(i_s, i_p, tau_rs, tau_as, x_s_all, x_p_all, x_qs_all, x_ps_all, dt):
    n = len(tau_rs)
    
    i_s, i_p, i_qs, i_ps = get_sine_off_cur(i_s, i_p)[:4]
    t = np.arange(len(i_s))*dt
    rs = np.nan*np.zeros((len(t), n))
    
    rs[0, :] = 0
    a_s = np.zeros(n)
    a_p = np.zeros(n)
    a_qs = np.zeros(n)
    a_ps = np.zeros(n)
    
    for ct, t_ in enumerate(t[1:], 1):
        a_s += ((dt/tau_as) * (-a_s + i_s[ct]))
        a_p += ((dt/tau_as) * (-a_p + i_p[ct]))
        a_qs += ((dt/tau_as) * (-a_qs + i_qs[ct]))
        a_ps += ((dt/tau_as) * (-a_ps + i_ps[ct]))
        
        rs[ct, :] = rs[ct-1, :] + (dt/tau_rs) * (-rs[ct-1, :] + (1-a_s)*x_s_all*i_s[ct] + (1-a_p)*x_p_all*i_p[ct] + (1-a_qs)*x_qs_all*i_qs[ct] + (1-a_ps)*x_ps_all*i_ps[ct])
        
    return rs
    
    
# MA-S-RB: multiplicative adaptation with "sine-rebound" responses
def smlt_ppln_masrb(i_s, i_p, tau_rs, tau_as, x_s_all, x_p_all, x_qs_all, x_ps_all, dt):
    n = len(tau_rs)
    
    i_s, i_p, i_qs, i_ps = get_sine_off_cur(i_s, i_p)[:4]
    t = np.arange(len(i_s))*dt
    rs = np.nan*np.zeros((len(t), n))
    
    rs[0, :] = 0
    a_s = np.zeros(n)
    a_p = np.zeros(n)
    
    for ct, t_ in enumerate(t[1:], 1):
        a_s += ((dt/tau_as) * (-a_s + i_s[ct]))
        a_p += ((dt/tau_as) * (-a_p + i_p[ct]))
        rs[ct, :] = rs[ct-1, :] + (dt/tau_rs) * (-rs[ct-1, :] + (1 - a_s)*x_s_all*i_s[ct] + (1 - a_p)*x_p_all*i_p[ct] + a_s*x_qs_all*i_qs[ct] + a_s*x_ps_all*i_ps[ct])
        
    return rs
    