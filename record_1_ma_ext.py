"""Functions for recording surrogate female neural activity in response to song."""
import numpy as np
from scipy import signal

from aux import get_seg

cc = np.concatenate


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


def get_sine_off_cur(i_s, i_p):
    
    bds_q = get_seg((i_s==0) & (i_p==0), min_gap=1)[1].astype(int)
    bds_s = get_seg(i_s==1, min_gap=1)[1].astype(int)
    bds_p = get_seg(i_p==1, min_gap=1)[1].astype(int)
    
    if bds_q.size == 0:
        bds_q = bds_q.reshape((0, 2))
    if bds_s.size == 0:
        bds_s = bds_s.reshape((0, 2))
    if bds_p.size == 0:
        bds_p = bds_p.reshape((0, 2))
    
    bds = cc([bds_q, bds_s, bds_p])
    modes = cc([np.repeat(0, len(bds_q)), np.repeat(1, len(bds_s)), np.repeat(2, len(bds_p))])
    
    idx_sorted = np.argsort(bds[:, 0])
    bds = bds[idx_sorted, :]
    modes = modes[idx_sorted]
    
    prev_modes = -1*np.ones(len(modes), dtype=int)
    prev_modes[1:] = modes[:-1]
    
    modes[(prev_modes == 1) & (modes == 0)] = 4  # quiet post-sine
    modes[(prev_modes == 1) & (modes == 2)] = 5  # pulse post-sine
    
    i_qs = np.zeros(len(i_s))
    i_ps = np.zeros(len(i_p))
    
    for mode, (lb, ub) in zip(modes, bds):
        if mode == 4:  # quiet post-sine
            i_qs[lb:ub] = 1
        elif mode == 5:  # pulse post-sine
            i_ps[lb:ub] = 1
            
    return i_s, i_p, i_qs, i_ps, modes


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


## MA_SIA: multiplicative adaptation neuron but with stimulus-invariant adaptation
def smlt_ma_sia(i_s, i_p, params, dt):
    tau_rs = params['TAU_R']
    tau_as = params['TAU_A']
    x_ss = params['X_S']
    x_ps = params['X_P']
    
    n = len(tau_rs)
    
    t = np.arange(len(i_s))*dt
    rs = np.nan*np.zeros((len(t), n))
    
    rs[0, :] = 0
    a = np.zeros(n)
    
    for ct, t_ in enumerate(t[1:], 1):
        a += ((dt/tau_as) * (-a + i_s[ct] + i_p[ct]))
        dr = (dt/tau_rs) * (-rs[ct-1, :] + (1 - a)*x_ss*i_s[ct] + (1 - a)*x_ps*i_p[ct])
        rs[ct, :] = rs[ct-1, :] + dr
    
    return rs


## MA_IND_TA: multiplicative adaptation neuron with independent adaptation times
def smlt_ma_ind_ta(i_s, i_p, params, dt):
    tau_rs = params['TAU_R']
    tau_a_ss = params['TAU_A_S']
    tau_a_ps = params['TAU_A_P']
    x_ss = params['X_S']
    x_ps = params['X_P']
    
    n = len(tau_rs)
    
    t = np.arange(len(i_s))*dt
    rs = np.nan*np.zeros((len(t), n))
    
    rs[0, :] = 0
    a_s = np.zeros(n)
    a_p = np.zeros(n)
    
    for ct, t_ in enumerate(t[1:], 1):
        a_s += ((dt/tau_a_ss) * (-a_s + i_s[ct]))
        a_p += ((dt/tau_a_ps) * (-a_p + i_p[ct]))
        dr = (dt/tau_rs) * (-rs[ct-1, :] + (1 - a_s)*x_ss*i_s[ct] + (1 - a_p)*x_ps*i_p[ct])
        rs[ct, :] = rs[ct-1, :] + dr
    
    return rs
    

    
# MASRB: multiplicative adaptation with "sine-rebound" responses
def smlt_masrb(i_s, i_p, params, dt):
    tau_rs = params['TAU_R']
    tau_as = params['TAU_A']
    x_ss = params['X_S']
    x_ps = params['X_P']
    x_qs_all = params['X_QS']
    x_ps_all = params['X_PS']
    
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
        
        dr = (dt/tau_rs) * (-rs[ct-1, :] \
            + (1 - a_s)*x_ss*i_s[ct] + (1 - a_p)*x_ps*i_p[ct] \
            + a_s*x_qs_all*i_qs[ct] + a_s*x_ps_all*i_ps[ct])
        
        rs[ct, :] = rs[ct-1, :] + dr
        
    return rs
    
    
    
# # MA-S-OFF: multiplicative adaptation neurons with sine-offset responses
# def smlt_masoff(i_s, i_p, params, dt):
#     tau_rs = params['TAU_R']
#     tau_as = params['TAU_A']
#     x_ss = params['X_S']
#     x_ps = params['X_P']
#     x_qs_all = params['X_QS']
#     x_ps_all = params['X_PS']
    
#     n = len(tau_rs)
    
#     i_s, i_p, i_qs, i_ps = get_sine_off_cur(i_s, i_p)[:4]
#     t = np.arange(len(i_s))*dt
#     rs = np.nan*np.zeros((len(t), n))
    
#     rs[0, :] = 0
#     a_s = np.zeros(n)
#     a_p = np.zeros(n)
#     a_qs = np.zeros(n)
#     a_ps = np.zeros(n)
    
#     for ct, t_ in enumerate(t[1:], 1):
#         a_s += ((dt/tau_as) * (-a_s + i_s[ct]))
#         a_p += ((dt/tau_as) * (-a_p + i_p[ct]))
#         a_qs += ((dt/tau_as) * (-a_qs + i_qs[ct]))
#         a_ps += ((dt/tau_as) * (-a_ps + i_ps[ct]))
        
#         dr = (dt/tau_rs) * (-rs[ct-1, :] \
#             + (1-a_s)*x_ss*i_s[ct] + (1-a_p)*x_ps*i_p[ct] \
#             + (1-a_qs)*x_qs_all*i_qs[ct] + (1-a_ps)*x_ps_all*i_ps[ct])
        
#         rs[ct, :] = rs[ct-1, :] + dr
        
#     return rs
    