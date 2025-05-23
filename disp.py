from copy import deepcopy
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from aux import get_seg

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def set_font_size(ax, font_size, legend_font_size=None):
    """Set font_size of all axis text objects to specified value."""
    
    axs = ax
    try:
        iter(axs)
    except TypeError:
        axs = np.array(ax)
    for ax in axs.flatten():
        texts = [ax.title, ax.xaxis.label, ax.yaxis.label] + \
            ax.get_xticklabels() + ax.get_yticklabels()
        
        try:
            texts.extend([ax.zaxis.label] + ax.get_zticklabels())
        except:
            pass

        for text in texts:
            text.set_fontsize(font_size)

        if ax.get_legend():
            if not legend_font_size:
                legend_font_size = font_size
            for text in ax.get_legend().get_texts():
                text.set_fontsize(legend_font_size)
                
                
def set_plot(ax, x_lim=None, y_lim=None, x_ticks=None, y_ticks=None, x_tick_labels=None, y_tick_labels=None,
        x_label=None, y_label= None, title=None, font_size=12):
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels is not None:
        ax.set_yticklabels(y_tick_labels)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    if font_size is not None:
        set_font_size(ax, font_size)
    

def get_line(x, y):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        
    nnan_mask = (~np.isnan(x)) & (~np.isnan(y))
    slp, icpt, r, p, stderr = stats.linregress(x[nnan_mask], y[nnan_mask])
    
    x_ln = np.array([np.nanmin(x), np.nanmax(x)])
    y_ln = slp*x_ln + icpt
    
    return x_ln, y_ln, (slp, icpt, r, p, stderr)
    
    
def get_line_log(x, y, n=None):
    log_x = np.log10(x)
    log_y = np.log10(y)
    
    if n is None:
        n = len(x)
    
    log_x_resampled = np.linspace(log_x[0], log_x[-1], n)
    log_y_resampled = np.interp(log_x_resampled, log_x, log_y)
    
    return get_line(log_x_resampled, log_y_resampled)

    
def set_n_x_ticks(ax, n, x_min=None, x_max=None):
    x_ticks = ax.get_xticks()
    
    x_min = np.min(x_ticks) if x_min is None else x_min
    x_max = np.max(x_ticks) if x_max is None else x_max
    
    ax.set_xticks(np.linspace(x_min, x_max, n))
    
    
def set_n_y_ticks(ax, n, y_min=None, y_max=None):
    y_ticks = ax.get_yticks()
    
    y_min = np.min(y_ticks) if y_min is None else y_min
    y_max = np.max(y_ticks) if y_max is None else y_max
    
    ax.set_yticks(np.linspace(y_min, y_max, n))

    
def set_color(ax, color, box=False):
    """Set colors on all parts of axis."""

    if box:
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['left'].set_color(color)
    ax.spines['right'].set_color(color)

    ax.tick_params(axis='x', color=color)
    ax.tick_params(axis='y', color=color)

    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color(color)

    ax.title.set_color(color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    
    
def get_spaced_colors(cmap, n, step):
    """step from 0 to 1"""
    cmap = cm.get_cmap(cmap)
    return cmap((np.arange(n, dtype=float)*step)%1)


def get_ordered_colors(cmap, n, lb=0, ub=1):
    cmap = cm.get_cmap(cmap)
    return cmap(np.linspace(lb, ub, n))

    
def fast_fig(n_ax, ax_size, fig_w=15):
    """Quickly make figure and axes objects from number of axes and ax size (h, w)."""
    n_col = int(round(fig_w/ax_size[1]))
    n_row = int(np.ceil(n_ax/n_col))
    
    fig_h = n_row*ax_size[0]
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(fig_w, fig_h), tight_layout=True, squeeze=False)
    return fig, axs.flatten()

# project-specific
def plot_b(ax, t, b, extent, c, t_bar=None):
    """
    Plot bout structure on an axis given time-series bout repr. Note: extent: [x_min, x_max, y_min, y_max]
    """
    x_0 = extent[0]
    dx = (extent[1] - extent[0]) / len(b)
    modes = np.sort(np.unique(b)).astype(int)
    for mode in modes:
        bds = get_seg(b==mode, min_gap=1)[1]
        for istart, iend in bds:
            ax.fill_between([x_0+dx*istart, x_0+dx*iend], 2*[extent[2]], 2*[extent[3]], color=c[mode], lw=0)
    if t_bar:        
        dt = np.mean(np.gradient(t))
        dy_scale = .2*(extent[3]-extent[2])
        ax.fill_between([x_0, x_0+dx/dt*t_bar], 2*[extent[3] + 2*dy_scale], 2*[extent[3] + 3*dy_scale], color='k')
    return ax


def plot_b_seg(ax, ts, extent, c):
    """
    Plot bout structure on an axis given segment dict repr. 
    E.g. ts = {0: [(0, 1), (2, 3)], 1: [(1, 2)], 2: [(3, 4)]}
    Note: extent: [x_min, x_max, y_min, y_max]
    """
    t_min = 0
    ends = {mode: np.max([seg[1] for seg in segs]) if segs else 0 for mode, segs in ts.items()}
    t_max = np.max(list(ends.values()))
    
    x_0 = extent[0]
    
    for mode, ts_ in ts.items():
        for start, end in ts_:
            ax.fill_between([x_0+start, x_0+end], 2*[extent[2]], 2*[extent[3]], color=c[mode])
            
    return ax


def trj_2d(ax, t, x, y, b, c=None, s_max=500, v_max=10, gam=1.):
    """Plot 2D bout 'trajectory'."""
    dt, dx, dy = np.gradient(t), np.gradient(x), np.gradient(y)
    v = np.sqrt(dx**2 + dy**2) / dt
    
    if c is None:
        c = ['gray', 'b', 'r', (1, .5, 0)]
        
    c = np.array(c, dtype=object)
    
    ax.scatter(x, y, c=c[b], s=(s_max*(1-(v/v_max)**gam)), zorder=1)
    for mode in np.sort(np.unique(b)).astype(int):
        bds = get_seg(b==mode, min_gap=1)[1]
        for istart, iend in bds:
            ax.plot(x[max(0, istart-1):iend], y[max(0, istart-1):iend], lw=2, c=c[mode], zorder=0)
    return ax


def trj_3d(ax, t, x, y, z, b, c=None, s_max=500, v_max=10, gam=1.):
    """Plot 3D bout 'trajectory'."""
    dt, dx, dy, dz = np.gradient(t), np.gradient(x), np.gradient(y), np.gradient(z)
    v = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    
    if c is None:
        c = ['gray', 'b', 'r', (1, .5, 0)]
        
    c = np.array(c, dtype=object)
    
    ax.scatter(x, y, z, c=c[b], s=(s_max*(1-(v/v_max)**gam)), zorder=1)
    for mode in np.sort(np.unique(b)).astype(int):
        bds = get_seg(b==mode, min_gap=1)[1]
        for istart, iend in bds:
            ax.plot(x[max(0, istart-1):iend], y[max(0, istart-1):iend], z[max(0, istart-1):iend], lw=2, c=c[mode], zorder=0)
    return ax


def plot_xpl_ma_rslt(rslts, isplit, itr_test, targ, tau_rs, tau_as, x_ss, x_ps):
    
    rslt = rslts[isplit]
    t = rslt.ts_test[itr_test]
    song = rslt.songs_test[itr_test]
    rs = rslt.xs_test[itr_test]

    y = rslt.ys_test[targ][itr_test]
    y_hat = rslt.y_hats_test[targ][itr_test]

    fig, axs = plt.subplots(2, 1, figsize=(15, 5), tight_layout=True)

    axs[0].plot(t, rs)
    axs[0].set_ylabel('Activity level')
    axs[0].set_title(f'Neural responses (trial {rslt.itr_test[itr_test]})')

    axs[1].plot(t, y, c='orange')
    axs[1].plot(t, y_hat, c='k')
    axs[1].legend(['True', 'Predicted'], ncol=2)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel(targ)
    axs[1].set_title(f'True vs predicted target (trial {rslt.itr_test[itr_test]})')

    # song bounding boxes
    song_bds_0 = [0, t.max() + np.mean(np.diff(t)), np.nanmax(rs) + .2*(np.nanmax(rs) - np.nanmin(rs)), np.nanmax(rs) + .4*(np.nanmax(rs) - np.nanmin(rs))]
    song_bds_1 = [0, t.max() + np.mean(np.diff(t)), np.nanmax(y) + .2*(np.nanmax(y) - np.nanmin(y)), np.nanmax(y) + .4*(np.nanmax(y) - np.nanmin(y))]

    for ax, song_bds in zip(axs, [song_bds_0, song_bds_1]):
        plot_b(ax, t, song, extent=song_bds, c=[(.9, .9, .9), 'b', 'r', 'r'])
        set_plot(ax, font_size=16)

    # neuron parameters vs targ prediction weights
    ws = []
    for cr, (tau_r, tau_a, x_s, x_p) in enumerate(zip(tau_rs, tau_as, x_ss, x_ps)):
        w = np.mean([rslt.w[targ][cr] for rslt in rslts])
        print(f'Neuron {cr+1}: TAU_R={tau_r:.2f}, TAU_A={tau_a:.3f}, X_S={x_s:.2f}, X_P={x_p:.2f} (W = {w:.2f}) ')
        ws.append(w)
        
    axs[0].legend([f'w = {w:.2f}' for w in ws], loc='lower right', ncol=min(3, len(ws)), fontsize=12)
    
    return axs