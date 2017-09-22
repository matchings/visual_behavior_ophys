"""
Created on Wednesday August 30 14:09:00 2017

@author: marinag
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# formatting
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=1.5)



def saveFigure(fig, fname, formats=['.pdf'], transparent=False, **kwargs):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])
    else:
        fig.set_size_inches(11, 8.5)
    for f in formats:
        fig.savefig(fname + f, transparent=transparent, orientation='landscape')


def plot_mask_on_max_proj(self, cell_list, ax=None, save=False):
    for roi in cell_list:
        if ax is None:
            fig, ax = plt.subplots()
        tmp = self.roi_dict[roi]
        mask = np.empty(tmp.shape, dtype=np.float)
        mask[:] = np.nan
        mask[tmp == 1] = 1
        ax.imshow(self.max_image, cmap='gray', vmin=0, vmax=0.5)
        ax.imshow(mask, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        ax.set_title('roi_' + str(roi))
        ax.grid(False)
        ax.axis('off')
        if save:
            plt.tight_layout()
            fig_folder = 'roi_masks'
            fig_title = 'roi_' + str(roi)
            fig_dir = os.path.join(self.analysis_dir, fig_folder)
            if not os.path.exists(fig_dir): os.mkdir(fig_dir)
            saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'])
            plt.close()
            ax = None
        else:
            return ax


def plot_trace(trace, ylabel='dF/F', interval=5, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
    n_frames = trace.shape[0]
    frames_range = np.arange(0, n_frames, interval * 60 * 30)
    minutes_range = np.arange(0, (n_frames / 30) / 60, interval)
    ax.plot(trace);
    ax.set_xlim([0, n_frames])
    ax.set_ylabel(ylabel)
    ax.set_xticks(frames_range);
    ax.set_xticklabels(minutes_range);
    ax.set_xlabel('minutes')
    return ax


def plot_trace_hist(trace, xlabel='dF/F', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(trace, bins=50);
    ax.set_ylabel('count');
    ax.set_xlabel(xlabel);
    return ax


def plot_traces(dataset, traces_list):
    fig, ax = plt.subplots(len(traces_list), 1, figsize=(15, 10))
    ax = ax.ravel()
    for i, roi in enumerate(traces_list):
        ax[i].plot(dataset.dff_traces[roi, :])
        #         ax[i].axis('off')
        ax[i].set_ylabel('roi ' + str(i))
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_axis_bgcolor('white')
        ax[i].set_xlim([0, dataset.dff_traces.shape[1]])
        ax[i].set_ylabel(str(roi))
    plt.show()
    fig_title = 'dff_traces'
    saveFigure(fig, os.path.join(dataset.analysis_dir, fig_title), formats=['.png'], size=(15, 10))


def plot_traces_heatmap(self, traces, trace_type='dFF', baseline_method='percentile', cmap='magma', save=False):
    sns.set_context('notebook', font_scale=2, rc={'lines.markeredgewidth': 1})
    # normalization method needs work
    figsize = (20, 8)
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(traces, cmap=cmap, vmin=np.percentile(traces, 1), vmax=np.percentile(traces, 99))
    ax.set_ylim(0, traces.shape[0])
    ax.set_xlim(0, traces.shape[1])
    ax.set_ylabel('cells')
    cb = plt.colorbar(cax);
    cb.set_label('dF/F', labelpad=5)
    fig.tight_layout()
    if save:
        plt.tight_layout()
        fig_title = trace_type + '_traces_' + baseline_method + '_heatmap_' + cmap
        fig_dir = os.path.join(self.analysis_dir, 'traces')
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
        plt.close()
    return fig, ax

def plot_all_trials(dataset, response_df):
    df = response_df
    for cell in df.cell.unique():
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize)
        for trial in df.trial.unique():
            trace = df[(df.cell == cell) & (df.trial == trial)]['responses'].values[0]
            timestamps = df[(df.cell == cell) & (df.trial == trial)].response_timestamps.values[0]
            timestamps = timestamps - timestamps[0]
            ax.plot(timestamps, trace)
            ax.set_title('roi ' + str(cell) + ' - all trials')
            ax.set_xticks(np.arange(dataset.window[0] - dataset.window[0], dataset.window[1] - dataset.window[0] + 1, 1))
            ax.set_xticklabels(np.arange(dataset.window[0], dataset.window[1] + 1, 1))
            ax.set_ylabel('dFF')
            ax.set_xlabel('time(s)')
        fig.tight_layout()
        fig_folder = 'all_trials'
        fig_title = 'roi_' + str(cell)
        fig_dir = os.path.join(dataset.analysis_dir, fig_folder)
        if not os.path.exists(fig_dir): os.mkdir(fig_dir)
        saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
        plt.close()



def placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1], wspace=None, hspace=None, sharex=False, sharey=False):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec

    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively

    returns:
        subaxes handles
    '''
    import matplotlib.gridspec as gridspec

    outer_grid = gridspec.GridSpec(100, 100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(dim[0], dim[1],
                                                  subplot_spec=outer_grid[int(100 * yspan[0]):int(100 * yspan[1]),
                                                               int(100 * xspan[0]):int(100 * xspan[1])],
                                                  wspace=wspace, hspace=hspace)

    # NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0] * [dim[1] * [
        fig]]  # filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            if row > 0 and sharex == True:
                share_x_with = inner_ax[0][col]
            else:
                share_x_with = None

            if col > 0 and sharey == True:
                share_y_with = inner_ax[row][0]
            else:
                share_y_with = None

            inner_ax[row][col] = plt.Subplot(fig, inner_grid[idx], sharex=share_x_with, sharey=share_y_with)
            fig.add_subplot(inner_ax[row, col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist()  # remove redundant dimension
    return inner_ax

