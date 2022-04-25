"""Functions for recording surrogate female neural activity in response to song."""
import numpy as np
from scipy import signal

from aux import get_sine_off_cur


# MA: multiplicative adaptation neuron
def smlt_ma(i_s, i_p, tau_r, tau_a, x_s, x_p, dt):
    """Simulate response to song inputs."""
    t = np.arange(len(i_s))*dt
    r = np.nan*np.zeros(len(t))
    
    r[0] = 0
    a_s, a_p = 0, 0
    
    for ct, t_ in enumerate(t[1:], 1):
        a_s += ((dt/tau_a) * (-a_s + x_s*i_s[ct]))
        a_p += ((dt/tau_a) * (-a_p + x_p*i_p[ct]))
        r[ct] = r[ct-1] + (dt/tau_r) * (-r[ct-1] + (x_s - a_s)*i_s[ct] + (x_p - a_p)*i_p[ct])
    
    return r


# LIN: linear filter neurons
def smlt_lin(i_s, i_p, h_s, h_p, dt):
    r = dt*(signal.fftconvolve(i_s, h_s, mode='full')[:len(i_s)] \
    + signal.fftconvolve(i_p, h_p, mode='full')[:len(i_p)])
    
    return r


# LIN2E: linear filter neurons where each filter is sum of two exponentials
def smlt_lin2e(i_s, i_p, t_h, x_s_0, tau_s_0, x_s_1, tau_s_1, x_p_0, tau_p_0, x_p_1, tau_p_1, dt):
    h_s = x_s_0*np.exp(-t_h/tau_s_0) + x_s_1*np.exp(-t_h/tau_s_1)
    h_p = x_p_0*np.exp(-t_h/tau_p_0) + x_p_1*np.exp(-t_h/tau_p_1)
    
    return smlt_lin(i_s, i_p, h_s, h_p, dt)


# LN: linear-nonlinear neurons
def smlt_ln(i_s, i_p, h_s, h_p, r_min, r_max, z_0, beta, dt):
    z = dt*(signal.fftconvolve(i_s, h_s, mode='full')[:len(i_s)] + signal.fftconvolve(i_p, h_p, mode='full')[:len(i_p)])
    
    return r_min + (r_max-r_min)*(np.tanh(beta*(z-z_0)) + 1)/2


# LN2E: linear-nonlinear neurons where each filter is sum of two exponentials
def smlt_ln2e(i_s, i_p, t_h, x_s_0, tau_s_0, x_s_1, tau_s_1, x_p_0, tau_p_0, x_p_1, tau_p_1, r_min, r_max, z_0, beta, dt):
    h_s = x_s_0*np.exp(-t_h/tau_s_0) + x_s_1*np.exp(-t_h/tau_s_1)
    h_p = x_p_0*np.exp(-t_h/tau_p_0) + x_p_1*np.exp(-t_h/tau_p_1)
    
    return smlt_ln(i_s, i_p, h_s, h_p, r_min, r_max, z_0, beta, dt)


# MA-S-OFF: multiplicative adaptation neuron with sine-offset responses
def smlt_masoff(i_s, i_p, tau_r, tau_a, x_s, x_p, x_qs, x_ps, dt):
    
    i_s, i_p, i_qs, i_ps = get_sine_off_cur(i_s, i_p)[:4]
    
    t = np.arange(len(i_s))*dt
    r = np.nan*np.zeros(len(t))
    
    r[0] = 0
    a_s, a_p, a_qs, a_ps = 0, 0, 0, 0
    
    for ct, t_ in enumerate(t[1:], 1):
        a_s += ((dt/tau_a) * (-a_s + i_s[ct]))
        a_p += ((dt/tau_a) * (-a_p + i_p[ct]))
        a_qs += ((dt/tau_a) * (-a_qs + i_qs[ct]))
        a_ps += ((dt/tau_a) * (-a_ps + i_ps[ct]))
        r[ct] = r[ct-1] + (dt/tau_r) * (-r[ct-1] + (1 - a_s)*x_s*i_s[ct] + (1 - a_p)*x_p*i_p[ct] + (1 - a_qs)*x_qs*i_qs[ct] + (1 - a_ps)*x_ps*i_ps[ct])
    
    return r


# MA-S-RB: multiplicative adaptation with "sine-rebound" responses
def smlt_masrb(i_s, i_p, tau_r, tau_a, x_s, x_p, x_qs, x_ps, dt):
    
    i_s, i_p, i_qs, i_ps = get_sine_off_cur(i_s, i_p)[:4]
    
    t = np.arange(len(i_s))*dt
    r = np.nan*np.zeros(len(t))
    
    r[0] = 0
    a_s, a_p = 0, 0
    
    for ct, t_ in enumerate(t[1:], 1):
        a_s += ((dt/tau_a) * (-a_s + i_s[ct]))
        a_p += ((dt/tau_a) * (-a_p + i_p[ct]))
        r[ct] = r[ct-1] + (dt/tau_r) * (-r[ct-1] + (1 - a_s)*x_s*i_s[ct] + (1 - a_p)*x_p*i_p[ct] + a_s*x_qs*i_qs[ct] + a_s*x_ps*i_ps[ct])
    
    return r
