"""Functions for recording surrogate female neural activity in response to song."""
import numpy as np
from scipy import signal


def smlt_rsvr(i_s, i_p, params, dt):
    w_rec = params['w_rec']
    w_in = params['w_in']
    tau = params['tau']
    
    hs = np.nan*np.zeros((len(i_s), w_rec.shape[0]))
    hs[0, :] = 0
    
    rs = np.nan*np.zeros((len(i_s), w_rec.shape[0]))
    rs[0, :] = np.tanh(hs[0, :])
    
    for t_ in range(1, len(i_s)):
        h = hs[t_-1, :]
        u = np.array([i_s[t_], i_p[t_]])
#         dh = (dt/tau)*(-h + np.tanh(w_rec@h + w_in@u))
        dh = (dt/tau)*(-h + w_rec@np.tanh(h) + w_in@u)
        hs[t_] = h + dh
        rs[t_] = np.tanh(hs[t_])
        
    return rs