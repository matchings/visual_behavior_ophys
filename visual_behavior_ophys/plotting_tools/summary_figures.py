# coding: utf-8

# In[130]:

###### warning! not all functions are functional. rewrite for scientifica data compatibility still in progress ########

import os
import numpy as np
import pandas as pd
import dro
import matplotlib.pyplot as plt
from visual_behavior_ophys.roi_mask_analysis import roi_mask_analysis as rm
from visual_behavior_ophys.plotting_tools import basic_plotting as bp
from visual_behavior_ophys.dro import utilities as du
# from visual_behavior_ophys.utilities import daily_figure_utilities as dfu
import seaborn as sns

# formatting
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=2.5, rc={'lines.markeredgewidth': 2})


def save_figure(fig, figsize, analysis_dir, folder, fig_title, formats=['.png']):
    fig_dir = os.path.join(analysis_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    filename = os.path.join(fig_dir, fig_title)
    for f in formats:
        fig.savefig(filename + f, transparent=False, orientation='landscape')


def get_xticks_xticklabels(trace, interval_sec):
    interval_frames = interval_sec * 30
    n_frames = len(trace)
    n_sec = n_frames / 30
    xticks = np.arange(0, n_frames + 1, interval_frames)
    xticklabels = np.arange(0, n_sec + 1, interval_sec)
    xticklabels = xticklabels - n_sec / 2
    return xticks, xticklabels


def exclude_aborted(df):
    df = df[df.trial_type != 'other']
    df = df[df.trial_type != 'aborted']
    return df


def make_lick_raster(df, ax, xmin=-4, xmax=8,figsize=None):
    df = exclude_aborted(df)
    for lap in range(len(df)):
        trialstart = df.iloc[lap]['starttime'] - df.iloc[lap]['change_time']
        licktimes = [(t - df.iloc[lap]['change_time']) for t in df.iloc[lap]['lick_times']]
        trialend = trialstart + df.iloc[lap]['trial_length']
        rewardtime = [(t - df.iloc[lap]['change_time']) for t in df.iloc[lap]['reward_times']]
        if len(rewardtime) > 1:
            rewardtime = rewardtime[0]
        if df.iloc[lap]['auto_rewarded'] == True:
            ax.axhspan(lap, lap + 1, -200, 200, color='gray', alpha=.5)
            ax.plot(rewardtime, lap + 0.5, '.', color='b', label='reward', markersize=6)
        if df.iloc[lap]['trial_type'] == 'go':
            if (df.iloc[lap]['auto_rewarded'] == True) == False:
                if df.iloc[lap]['response_type'] == 'HIT':
                    ax.axhspan(lap, lap + 1, -200, 200, color='#55a868', alpha=.5)
                    ax.plot(rewardtime, lap + 0.5, '.', color='b', label='reward', markersize=6)
                else:
                    ax.axhspan(lap, lap + 1, -200, 200, color='#ccb974', alpha=.5)
        if df.iloc[lap]['trial_type'] == 'catch':
            if df.iloc[lap]['response_type'] == 'FA':
                ax.axhspan(lap, lap + 1, -200, 200, color='#c44e52', alpha=.5)
            else:
                ax.axhspan(lap, lap + 1, -200, 200, color='#4c72b0', alpha=.5)

        ax.vlines(trialstart, lap, lap + 1, color='black', linewidth=1)
        ax.vlines(licktimes, lap, lap + 1, color='r', linewidth=1)
        ax.vlines(0, lap, lap + 1, color=[.5, .5, .5], linewidth=1)

    ax.axvspan(df.iloc[0]['response_window'][0], df.iloc[0]['response_window'][1], facecolor='gray', alpha=.4,
               edgecolor='none')
    ax.grid(False)
    ax.set_ylim(0, len(df))
    ax.set_xlim([xmin, xmax])


def plot_behavior(pkl_df, save_dir=None, ax=None):
    if ax is None:
        figsize = (6, 12)
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
    pdf = pkl_df[pkl_df.trial_type != 'aborted'].reset_index()
    make_lick_raster(pdf, ax, xmin=-1, xmax=4)
    #    ax.set_title('lick raster')
    ax.set_xlabel('time after change (s)')
    ax.set_ylabel('trials')
    plt.gca().invert_yaxis()
    if save_dir:
        save_figure(fig, figsize, save_dir, fig_title='behavior', folder='behavior', formats=['.png'])
    return ax


#
# # In[131]:
# def create_resp_prob_heatmap(df, ax=None, cmap=cmaps.magma, analysis_dir=None):
#     if ax is None:
#         figsize = (7, 7)
#         fig, ax = plt.subplots(figsize=figsize)
#
#     df.fillna(value=0, inplace=True)
#     # for dealing with initial_image_name=0 issues
#     #    df = df[(df.initial_image_name)!=0]
#     df = df[df.trial_type != 'aborted']
#     response_matrix = pd.pivot_table(df,
#                                      values='response',
#                                      index=['initial_image_category', 'initial_image'],
#                                      columns=['change_image_category', 'change_image'])
#
#     sns.heatmap(response_matrix, cmap=cmap, linewidths=0, linecolor='white', square=True, annot=True,
#                 annot_kws={"fontsize": 14}, vmin=0, vmax=1,
#                 robust=True, cbar_kws={"drawedges": False, "shrink": 0.7}, ax=ax)
#
#     ax.set_title('resp prob by image category and name', fontsize=14, va='bottom', ha='center')
#     ax.set_xlabel('final image', fontsize=14)
#     ax.set_ylabel('initial image', fontsize=14)
#
#     if analysis_dir:
#         plt.gcf().subplots_adjust(left=0.32)
#         plt.gcf().subplots_adjust(bottom=0.22)
#         save_figure(fig, figsize, analysis_dir, 'response_probability', 'behavior')
#
#     return ax
#
#

def create_resp_prob_heatmap_general(df, values='response', index='initial_image', columns='change_image',
                                     filter_by_reward_rate=True, aggfunc=np.mean, cmap='magma', ax=None,
                                     analysis_dir=None):
    if ax is None:
        figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize)

    df.fillna(value=0, inplace=True)
    # for dealing with initial_image_name=0 issues
    try:
        df = df[(df.change_image) != 0]
    except:
        'cant remove change image == 0'
    if filter_by_reward_rate:
        df = df[df.reward_rate >= 2]

    df = df[df.trial_type != 'aborted']
    response_matrix = pd.pivot_table(df,
                                     values=values,
                                     index=[index],
                                     columns=[columns],
                                     aggfunc=aggfunc)
    if values == 'response':
        vmax = 1
        vmin = 0
    else:
        vmax = np.amax(np.amax(response_matrix))
        vmin = np.amin(np.amin(response_matrix))

    sns.heatmap(response_matrix, cmap=cmap, linewidths=0, linecolor='white', square=True, annot=True,
                annot_kws={"fontsize": 14}, vmin=vmin, vmax=vmax,
                robust=True, cbar_kws={"drawedges": False, "shrink": 0.7}, ax=ax)

    #    ax.set_title(values+' by '+index+' and '+columns,va='bottom', ha='center')
    ax.set_title(values, va='bottom', ha='center')
    ax.set_xlabel(columns)
    ax.set_ylabel(index)
    if analysis_dir:
        plt.gcf().subplots_adjust(left=0.32)
        plt.gcf().subplots_adjust(bottom=0.22)
        save_figure(fig, figsize, analysis_dir, values + '_' + index + '_' + columns, 'behavior', formats=['.png'])

    return ax


# # In[ ]:
#
# def plot_mean_resp_by_transition_matrix(resp_df, analysis_dir, fig_title):
#     figsize = (6, 6)
#     fig, ax = plt.subplots(figsize=figsize)
#     # df = df[df.trial_type!='aborted']
#     response_matrix = pd.pivot_table(resp_df,
#                                      values='mean_response',
#                                      index=['initial_code'],
#                                      columns=['change_code'])
#
#     sns.heatmap(response_matrix, cmap=cmaps.magma, linewidths=0, linecolor='white', square=True, annot=True,
#                 annot_kws={"fontsize": 12}, vmin=0, vmax=np.amax(np.amax(response_matrix)),
#                 robust=True, cbar_kws={"drawedges": False, "shrink": 0.7}, ax=ax)
#
#     ax.set_title('mean response by transition', fontsize=16, va='bottom', ha='center')
#     ax.set_xlabel('final image', fontsize=16)
#     ax.set_ylabel('initial image', fontsize=16)
#
#     save_figure(fig, figsize, analysis_dir, fig_title='sig_cells', folder='cell_response_heatmaps')
#     plt.close()
#
#
# # In[ ]:
#
def plot_mean_resp_heatmap_cell(df, cell, values='mean_response', index='initial_code', columns='change_code',
                                analysis_dir=None, ax=None):
    resp_df = df[df.cell == cell]
    resp_df = resp_df[['cell', index, columns, values, 'responses']]
    #    resp_df = resp_df[['cell','initial_code','change_code','responses','mean_response']]
    if ax is None:
        figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize)
    # df = df[df.trial_type!='aborted']
    response_matrix = pd.pivot_table(resp_df,
                                     values='mean_response',
                                     index=[index],
                                     columns=[columns])

    sns.heatmap(response_matrix, cmap='magma', linewidths=0, linecolor='white', square=True, annot=True,
                annot_kws={"fontsize": 14}, vmin=0, vmax=np.amax(np.amax(response_matrix)),
                robust=True, cbar_kws={"drawedges": False, "shrink": 0.7}, ax=ax)

    ax.set_title('roi ' + str(cell) + ' - ' + values, fontsize=16, va='bottom', ha='center')
    ax.set_xlabel(columns, fontsize=16)
    ax.set_ylabel(index, fontsize=16)

    if analysis_dir is not None:
        fig.tight_layout()
        save_figure(fig, figsize, analysis_dir, fig_title='roi_' + str(cell) + '_' + index + '_pref',
                    folder='cell_response_heatmaps2', formats=['.png'])
        plt.close()
    return ax


#
#
# # In[154]:
#
# def plot_mean_response_matrix_traces(dataset, sdf, cell, xlim=[-2, 2], save=False, ax=None):
#     cond0_df = dataset.df.copy()
#     sc = dataset.stim_codes.stim_code.values
#     if ax is None:
#         figsize = (2 * len(sc), 2 * len(sc))
#         fig, ax = plt.subplots(len(sc), len(sc), figsize=figsize, sharey=True, sharex=True)
#         ax = ax.ravel();
#     colors = sns.color_palette()
#     i = 0
#     for x, initial_code in enumerate(sc):
#         for y, change_code in enumerate(sc):
#             cdf0 = cond0_df[(cond0_df.cell == cell) & (cond0_df.initial_code == initial_code) & (
#             cond0_df.change_code == change_code)]
#             ax[i] = plot_mean_trace(cdf0, xlim=xlim, color=colors[0], label=None, title=None, ax=ax[i])
#             ax[i].set_xlabel('')
#             ax[i].set_ylabel('')
#             i += 1
#     if save:
#         #     fig.tight_layout()
#         formats = ['.png']
#         save_figure(fig, figsize, dataset.analysis_dir, 'roi_' + str(cell), 'mean_resp_matrix_traces', formats=formats)
#         plt.close()
#         # return ax;
#
#
# # In[154]:

def plot_images(dataset, mdf, ax=None, save=False, orientation=None):
    pkl = dataset.pkl
    colors = get_colors_for_stim_codes(dataset.stim_codes.stim_code.unique())
    if orientation == 'row':
        figsize = (20, 5)
        cols = len(mdf.stim_code.unique())
        rows = 1
    elif orientation == 'column':
        figsize = (5, 20)
        cols = 1
        rows = len(mdf.stim_code.unique())
    elif len(mdf.stim_code.unique()) == 5:
        figsize = (10, 10)
        rows = len(mdf.stim_code.unique()) / 2 + 1
        cols = 2
        colors.append([1, 1, 1])
        shape = pkl['image_dict'][0]['img061_VH.tiff'].shape
        arr = np.zeros((shape))
        arr[:] = 1
        dataset.pkl['image_dict'][6] = {'  ': arr}
    elif len(mdf.stim_code.unique()) == 8:
        figsize = (15, 10)
        rows = len(mdf.stim_code.unique()) / 2
        cols = 2

    if ax is None:
        fig, ax = plt.subplots(rows, cols, figsize=figsize)
        ax = ax.ravel();
    if len(str(dataset.pkl['image_dict'].keys()[0])) > 3:
        for i, stim_code in enumerate(dataset.stim_codes.stim_code.values):
            img_name = dataset.stim_codes[dataset.stim_codes.stim_code == stim_code].full_image_name.values[0]
            img = pkl['image_dict'][img_name]
            ax[i].imshow(img, cmap='gray', vmin=0, vmax=np.amax(img))
            ax[i].grid('off')
            ax[i].axis('off')
            ax[i].set_title(img_name, color=colors[i])
            ax[i].set_title(str(stim_code), color=colors[i], fontweight='bold')
    else:
        #        for i,img_num in enumerate(pkl['image_dict'].keys()):
        #            img_name = pkl['image_dict'][img_num].keys()[0]
        if dataset.stimulus_name == 'natural_scene':
            for i, stim_code in enumerate(dataset.stim_codes.stim_code.values):
                img_name = dataset.stim_codes[dataset.stim_codes.stim_code == stim_code].image_name.values[0]
                for i, img_num in enumerate(pkl['image_dict'].keys()):
                    img_name2 = pkl['image_dict'][img_num].keys()[0]
                    if img_name2 == img_name:
                        img = pkl['image_dict'][img_num][img_name]
                        ax[i].imshow(img, cmap='gray', vmin=0, vmax=np.amax(img))
                        ax[i].grid('off')
                        ax[i].axis('off')
                        ax[i].set_title(img_name, color=colors[i])
                        ax[i].set_title(str(stim_code), color=colors[i])
        elif dataset.stimulus_name == 'mnist':
            for i, stim_code in enumerate(dataset.stim_codes.stim_code.values):
                img_num = dataset.stim_codes[dataset.stim_codes.stim_code == stim_code].image_num.values[0]
                img_name = dataset.stim_codes[dataset.stim_codes.stim_code == stim_code].image_name.values[0]
                img = pkl['image_dict'][img_num][img_name]
                ax[i].imshow(img, cmap='gray', vmin=0, vmax=np.amax(img))
                ax[i].grid('off')
                ax[i].axis('off')
                ax[i].set_title(img_name, color=colors[i])
                ax[i].set_title(str(stim_code), color=colors[i])
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, fig_title='images', folder='behavior', formats=['.png'])
        plt.close()
    return ax

    # In[154]:


#
#
# def plot_mean_trace(cond_df, xlim=[-1, 3], color='k', label=None, title=None, xspan=True, analysis_dir=False, folder=None,
#                     save_title=None, ax=None):
#     if ax is None:
#         figsize = (5, 4)
#         fig, ax = plt.subplots(figsize=figsize)
#     if len(cond_df) > 0:
#         traces = cond_df.responses.values
#         mean_trace = np.mean(traces)
#         sem = np.std(traces) / np.sqrt(float(len(traces)))
#         times = np.arange(0, len(mean_trace), 1)
#         ax.fill_between(times, mean_trace + sem, mean_trace - sem, color=color, alpha=0.5)
#         ax.plot(times, mean_trace, color=color, linewidth=3, label=label)
#         xticks, xticklabels = get_xticks_xticklabels(mean_trace, interval_sec=1)
#         ax.set_xticks(xticks)
#         ax.set_xticklabels(xticklabels)
#         x0 = xticks[np.where(xticklabels == xlim[0])[0][0]]
#         x1 = xticks[np.where(xticklabels == xlim[1])[0][0]]
#         xlim = [x0, x1]
#         ax.set_xlim(xlim)
#         if xspan:
#             span_start = xticks[np.where(xticklabels == 0)[0][0]]
#             ax.axvspan(span_start, span_start + 0.5 * 30, facecolor='gray', alpha=0.3)
#         ax.set_xlabel('time after change (sec)')
#         ax.set_ylabel('dF/F')
#         if title:
#             ax.set_title(title)
#         if label:
#             ax.legend(loc='upper right')
#             #             ax.legend(bbox_to_anchor=(1.55, 1.05))
#     if analysis_dir:
#         fig.tight_layout()
#         save_figure(fig, figsize, analysis_dir, save_title, folder)
#         plt.close()
#         ax = None
#     return ax;
#
#
#     # In[154]:
#
#
def plot_vsyncintervals(pkl, analysis_dir):
    figsize = (8, 6)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pkl['vsyncintervals'])
    ax.set_ylabel('vsync intervals')
    ax.set_xlabel('stimulus frames')
    dropped_frames = np.round(pkl['dropped_frame_count'] / float(pkl['vsynccount']), 2) * 100
    title = str(pkl['dropped_frame_count']) + '/' + str(pkl['vsynccount']) + ' (' + str(dropped_frames) + '%)'
    ax.set_title(title + ' dropped frames')
    plt.tight_layout()
    save_figure(fig, figsize, analysis_dir, 'dropped_frames', 'behavior')


#
#
# def plot_return_to_baseline(dataset, cell, sdf, code='initial_code'):
#     interval_sec = 1
#     df = dataset.df
#     pref_stim = sdf[sdf.cell == cell].pref_stim.values[0]
#     img_num = dataset.stim_codes[dataset.stim_codes.stim_code == pref_stim].image_name.values[0]
#     tmp = df[(df.cell == cell) & (df[code] == pref_stim) & (df.trial_type == 'go')]
#     trace = tmp.responses.mean()
#     figsize = (8, 5)
#     fig, ax = plt.subplots()
#     ax.plot(trace)
#     xticks, xticklabels = get_xticks_xticklabels(trace, interval_sec)
#     ax.set_xticks(xticks);
#     ax.set_xticklabels(xticklabels);
#     ax.set_xlabel('time after change (s)')
#     ax.set_ylabel('dF/F')
#     ax.set_title('cell ' + str(cell) + ' - go trials, ' + code + ' = ' + str(img_num))
#
#     save_figure(fig, figsize, dataset.analysis_dir, fig_title='cell_' + str(cell) + '_' + code, folder='return_to_baseline')
#     plt.close()
#
#
# # In[141]:

def plot_engaged_disengaged(dataset, cell, sdf, code='change_code', save=False, ax=None):
    # code can be 'change_code' or 'initial_code'
    interval_sec = 1
    colors = sns.color_palette()
    pref_stim = sdf[sdf.cell == cell].pref_stim.values[0]
    img_num = dataset.stim_codes[dataset.stim_codes.stim_code == pref_stim].image_name.values[0]
    if ax is None:
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize)
    pkl_df = dataset.pkl_df
    df = dataset.df
    engaged_trials = pkl_df[(pkl_df.reward_rate >= 2) & (pkl_df.trial_type != 'aborted')].index.values
    traces = df[(df.cell == cell) & (df[code] == pref_stim) & (df.trial_type == 'go') & (
        df.global_trial.isin(engaged_trials))].responses.values
    trace = traces.mean()
    times = np.arange(0, len(trace), 1)
    sem = (traces.std()) / np.sqrt(float(len(traces)))
    ax.plot(trace, label='engaged', linewidth=3, color=colors[0])
    ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=colors[0])

    disengaged_trials = pkl_df[(pkl_df.reward_rate < 2) & (pkl_df.trial_type != 'aborted')].index.values
    traces = df[(df.cell == cell) & (df[code] == pref_stim) & (df.trial_type == 'go') & (
        df.global_trial.isin(disengaged_trials))].responses.values
    trace = traces.mean()
    times = np.arange(0, len(trace), 1)
    sem = (traces.std()) / np.sqrt(float(len(traces)))
    ax.plot(trace, label='disengaged', linewidth=3, color=colors[2])
    ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=colors[2])

    xticks, xticklabels = get_xticks_xticklabels(trace, interval_sec)
    ax.set_xticks(xticks);
    ax.set_xticklabels(xticklabels);
    ax.set_xlim(2 * 30, 7 * 30)
    ax.set_xlabel('time after change (s)')
    ax.set_ylabel('dF/F')

    if save:
        plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.0))
        ax.set_title('cell ' + str(cell) + ' - ' + str(img_num))
        fig.tight_layout()
        save_figure(fig, figsize, dataset.analysis_dir, fig_title='cell_' + str(cell), folder='task_engagement')
        plt.close()
    else:
        plt.legend(loc='upper right', bbox_to_anchor=(0.85, 1.35))


def plot_running_not_running(dataset, cell, sdf, code='change_code', save=False, ax=None):
    # code can be 'change_code' or 'initial_code'
    interval_sec = 1
    colors = sns.color_palette()
    pref_stim = sdf[sdf.cell == cell].pref_stim.values[0]
    img_num = dataset.stim_codes[dataset.stim_codes.stim_code == pref_stim].image_name.values[0]
    if ax is None:
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize)
    df = dataset.df
    running_trials = df[df.avg_run_speed > 3].global_trial.values
    traces = df[(df.cell == cell) & (df[code] == pref_stim) & (df.trial_type == 'go') & (
        df.global_trial.isin(running_trials))].responses.values
    if len(traces) > 3:
        trace = traces.mean()
        times = np.arange(0, len(trace), 1)
        sem = (traces.std()) / np.sqrt(float(len(traces)))
        ax.plot(trace, label='running', linewidth=3, color=colors[0])
        ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=colors[0])

    non_running_trials = df[df.avg_run_speed < 3].global_trial.values
    traces = df[(df.cell == cell) & (df[code] == pref_stim) & (df.trial_type == 'go') & (
        df.global_trial.isin(non_running_trials))].responses.values
    if len(traces) > 3:
        trace = traces.mean()
        times = np.arange(0, len(trace), 1)
        sem = (traces.std()) / np.sqrt(float(len(traces)))
        ax.plot(trace, label='not running', linewidth=3, color=colors[2])
        ax.fill_between(times, trace + sem, trace - sem, alpha=0.5, color=colors[2])

    xticks, xticklabels = get_xticks_xticklabels(trace, interval_sec)
    ax.set_xticks(xticks);
    ax.set_xticklabels(xticklabels);
    ax.set_xlim(2 * 30, 7 * 30)
    ax.set_xlabel('time after change (s)')
    ax.set_ylabel('dF/F')

    if save:
        plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.0))
        ax.set_title('cell ' + str(cell) + ' - ' + str(img_num))
        fig.tight_layout()
        save_figure(fig, figsize, dataset.analysis_dir, fig_title='cell_' + str(cell), folder='task_engagement')
        plt.close()
    else:
        plt.legend(loc='upper right', bbox_to_anchor=(0.85, 1.35))


#
# def plot_trial_type_responses(dataset, cell_list, trace_type='dFF', save=True, ax=None):
#     df = dataset.df
#     colors = ['r', 'k']
#     alphas = [1, 0.5, 1, 0.5]
#     response_types = ['HIT', 'MISS', 'FA', 'CR']
#     trial_types = ['go', 'catch']
#     trial_type_labels = ['change trials', 'catch trials']
#     stim_code_labels = [['B-M', 'M-B'], ['M-M', 'B-B']]
#     w = dataset.window
#     for cell in cell_list:
#         if ax is None:
#             fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
#             ax = ax.ravel()
#         for i, trial_type in enumerate(trial_types):
#             for y, response_type in enumerate(response_types):
#                 for x, change_code in enumerate(np.sort(df.change_code.unique())):
#                     #                     print trial_type, response_type, change_code
#                     sub_df = df[
#                         (df.cell == cell) & (df.trial_type == trial_type) & (df.response_type == response_type) & (
#                         df.change_code == change_code)]
#                     if len(sub_df) > 0:
#                         times = sub_df.response_timestamps.values.mean()
#                         times = times - times[0]
#                         if trace_type == 'dF':
#                             traces = sub_df.responses.values
#                         elif trace_type == 'dFF':
#                             traces = sub_df.responses_dFF.values
#                         avg_trace = traces.mean()
#                         sem = (traces.std()) / np.sqrt(float(len(traces)))
#                         label = response_types[y] + ' ' + stim_code_labels[i][x]
#                         line, = ax[i].plot(times, avg_trace, label=label, color=colors[x], alpha=alphas[y])
#                         ax[i].fill_between(times, avg_trace + sem, avg_trace - sem, facecolor=colors[x],
#                                            alpha=alphas[y] / 3.)
#                         ax[i].set_xticks(np.arange(0, (w[1] - w[0]) + 1, 1))
#                         ax[i].set_xticklabels(np.arange(w[0], w[1] + 1, 1))
#                         ax[0].set_ylabel(trace_type)
#                         ax[i].set_xlabel('time(s)')
#                         ax[i].set_title(trial_type_labels[i])
#                         handles, labels = ax[i].get_legend_handles_labels()
#                         ax[i].legend(handles, labels, loc=9, ncol=2)
#                         plt.tight_layout()
#         if save:
#             fig_title = 'roi_' + str(cell)
#             fig_dir = os.path.join(dataset.analysis_dir, 'trial_type_responses')
#             if not os.path.exists(fig_dir):
#                 os.mkdir(fig_dir)
#             saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=(10, 5))
#             plt.close()
#             ax = None
#         else:
#             return ax


def get_lick_times(dataset):
    lick_times = dataset.sync['lickTimes_0']['timestamps']
    return lick_times


def get_reward_times(dataset):
    visual_frames = dataset.sync['visualFrames']['timestamps']
    reward_times = []
    for idx in dataset.pkl_df.index:
        reward_frames = dataset.pkl_df.reward_frames.loc[idx]
        for reward_frame in reward_frames:
            reward_times.append(visual_frames[reward_frame])
    dataset.reward_times = reward_times
    return reward_times


def plot_trace(timestamps, trace, ax, xlabel='seconds', ylabel='fluorescence', title='roi'):
    colors = sns.color_palette()
    ax.plot(timestamps, trace, color=colors[0], linewidth=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


#
# def plot_trace_summary(trace, ylabel='dF/F', interval=5, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(20, 5))
#     n_frames = trace.shape[0]
#     frames_range = np.arange(0, n_frames, interval * 60 * 30)
#     minutes_range = np.arange(0, (n_frames / 30) / 60, interval)
#     ax.plot(trace);
#     ax.set_xlim([0, n_frames])
#     ax.set_ylabel(ylabel)
#     ax.set_xticks(frames_range);
#     ax.set_xticklabels(minutes_range);
#     ax.set_xlabel('minutes')
#     return ax
#
#
# # In[152]:

def plot_trace_hist(trace, xlabel, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(trace, bins=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')


def restrict_axes(xmin, xmax, interval, ax):
    xticks = np.arange(xmin, xmax, interval)
    ax.set_xlim([xmin, xmax])
    ax.set_xticks(xticks)
    return ax


def get_color_for_response_type(response_type):
    c = sns.color_palette()
    colors_dict = {'HIT': c[1], 'MISS': c[4], 'CR': c[0], 'FA': c[2]}
    color = colors_dict[response_type]
    return color


def get_color_for_stim(stim, stim_codes):
    stim_codes = np.sort(stim_codes)
    if len(stim_codes) > 2:
        c = sns.color_palette("hls", len(stim_codes))
        colors_dict = {}
        for i, code in enumerate(stim_codes):
            colors_dict[code] = c[i]
    elif len(stim_codes) == 2:
        c = sns.color_palette()
        colors_dict = {0: c[5], 1: c[3]}
    color = colors_dict[stim]
    return color


def get_colors_for_response_types(values):
    c = sns.color_palette()
    colors_dict = {'HIT': c[1], 'MISS': c[4], 'CR': c[0], 'FA': c[2]}
    colors = []
    for val in values:
        colors.append(colors_dict[val])
    return colors


def get_colors_for_stim_codes(stim_codes):
    if len(stim_codes) > 2:
        c = sns.color_palette("hls", len(stim_codes))
        colors_dict = {}
        for i, code in enumerate(stim_codes):
            colors_dict[code] = c[i]
    elif len(stim_codes) == 2:
        c = sns.color_palette()
        colors_dict = {0: c[5], 1: c[3]}
    colors = []
    for stim_code in stim_codes:
        colors.append(colors_dict[stim_code])
    return colors


def plot_reward_rate(frame_times, reward_rate, ax=None, label=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    colors = sns.color_palette()
    engaged = np.asarray(reward_rate).copy()
    engaged[engaged <= 2.] = np.nan
    disengaged = np.asarray(reward_rate).copy()
    disengaged[disengaged >= 2.] = np.nan
    ax.plot(frame_times, reward_rate, color=colors[2], linewidth=4, label='disengaged', zorder=1)
    ax.plot(frame_times, engaged, color=colors[0], linewidth=4, label='engaged', zorder=2)
    if label:
        plt.legend(loc='center right', bbox_to_anchor=(1.23, 0.8))
        ax.set_title('reward rate')
        ax.set_xlabel('time(s)')
    return ax


def plot_run_speed(times, run_speed, ax=None, label=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(times, run_speed, color='gray')
    if label:
        ax.set_title('running speed (cm/s)')
        ax.set_xlabel('time(s)')
    return ax


def plot_d_prime(frame_times, d_prime, colors, ax):
    ax.plot(frame_times, d_prime, color=colors[3], linewidth=4, label='d_prime')
    ax.set_ylabel('d prime')
    return ax


def plot_hit_false_alarm_rates(frame_times, hr, cr, ax):
    ax.plot(frame_times, hr, color='#55a868', linewidth=4, label='hit_rate')
    ax.plot(frame_times, cr, color='#c44e52', linewidth=4, label='fa_rate')
    ax.set_ylabel('response rate')
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc='upper right')


def plot_task_performance(frame_times, pkl_df, ax=None, label=False, plot_d_prime=False, plot_hr_fa=True):
    colors = sns.color_palette()
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 5))
    tmp = pkl_df.copy()
    tmp = tmp[tmp.trial_type != 'other']
    df_in = tmp[tmp.trial_type != 'aborted'].reset_index()
    if 'level_0' in df_in:
        df_in = df_in.drop('level_0', axis=1)
    hr, cr, d_prime = du.get_response_rates(df_in, sliding_window=100, reward_window=None)
    if plot_d_prime and plot_hr_fa:
        plot_d_prime(frame_times, d_prime, colors, ax)
        ax2 = ax.twinx()
        plot_hit_false_alarm_rates(frame_times, hr, cr, ax2)
        ax2.grid(False)
    elif plot_d_prime:
        plot_d_prime(frame_times, d_prime, colors, ax)
    elif plot_hr_fa:
        plot_hit_false_alarm_rates(frame_times, hr, cr, ax)
    ax.legend(loc='center right')
    if label:
        plt.legend(loc='upper right')
        ax.set_title('behavioral performance')
        ax.set_xlabel('time(s)')
    return ax


def plot_trace_and_stuff(dataset, cell, second_axis=None, plot_stim=True, ax=None):
    # second_axis can be 'reward_rate' or 'task_performance'

    times = dataset.sync['visualFrames']['timestamps']
    frame_times = []
    reward_rate = []
    pkl_df = dataset.pkl_df.copy()
    for i, total_trial in enumerate(dataset.stim_table.total_trial.values):
        frame_times.append(dataset.stim_table.loc[i].change_time)
        reward_rate.append(dataset.pkl_df.iloc[total_trial].reward_rate)

    if ax is None:
        figsize = (20, 4)
        fig, ax = plt.subplots(figsize=figsize)

    trace = dataset.dff_traces[cell, :]
    times_2p = dataset.sync['2PFrames']['timestamps'][:len(trace)]
    ax.plot(times_2p, trace, color='gray', linewidth=1)
    ax.set_ylabel('dF/F')
    if second_axis:
        ax2 = ax.twinx()
        if second_axis == 'reward_rate':
            ax2 = sf.plot_reward_rate(frame_times, reward_rate, ax=ax2)
            ax2.set_ylabel('reward rate')
        elif second_axis == 'task_performance':
            ax2 = plot_task_performance(frame_times, pkl_df, ax=ax2)
        ax2.grid(False)

    upper_limit, time_interval, frame_interval = get_upper_limit_and_intervals(dataset)
    ax.set_xlim(time_interval[0], np.uint64(upper_limit / 30.))
    ax.set_xlabel('time (s)')
    plt.legend()
    return ax


def addSpan(ax, amin, amax, color='k', alpha=0.3, axtype='x', zorder=1):
    if axtype == 'x':
        ax.axvspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0, zorder=zorder)
    if axtype == 'y':
        ax.axhspan(amin, amax, facecolor=color, edgecolor='none', alpha=alpha, linewidth=0, zorder=zorder)


def add_stim_color_span(dataset, ax, stim_code=None):
    if stim_code is not None:
        stim_list = [stim_code]
    colors = get_colors_for_stim_codes(np.sort(dataset.stim_codes.stim_code.unique()))
    amin = 0
    for idx in dataset.stim_table.index:
        amax = dataset.stim_table.loc[idx]['change_time']
        code = dataset.stim_table.loc[idx]['initial_code']
        color = colors[code]
        if stim_code is not None:
            if code in stim_list:
                addSpan(ax, amin, amax, color=color)
        else:
            addSpan(ax, amin, amax, color=color)
        amin = amax
    return ax


def plot_behavior_events(dataset, ax):
    ax = add_stim_color_span(dataset, ax)

    lick_times = get_lick_times(dataset)
    reward_times = get_reward_times(dataset)
    ymin, ymax = ax.get_ylim()
    lick_y = ymin + (ymax * 0.05)
    reward_y = ymin + (ymax * 0.1)
    lick_y_array = np.empty(len(lick_times))
    lick_y_array[:] = lick_y
    reward_y_array = np.empty(len(reward_times))
    reward_y_array[:] = reward_y
    ax.plot(lick_times, lick_y_array, '|', color='g')
    ax.plot(reward_times, reward_y_array, 'o', markerfacecolor='purple', markeredgecolor='purple')
    return ax


def plot_behavior_events_trace(dataset, cell_list, xmin=360, length=3, ax=None, save=False, ylabel='dF/F'):
    sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})  # normally font_scale 2.5
    xmax = xmin + 60 * length
    interval = 20
    for cell in cell_list:
        if ax is None:
            figsize = (15, 4)
            fig, ax = plt.subplots(figsize=figsize)
        timestamps = dataset.timestamps_2p
#         trace = np.squeeze(dataset.dff_traces[cell, :])
        ax = plot_trace(timestamps, dataset.dff_traces[cell, :], ax, title='roi_' + str(cell), ylabel=ylabel)
        ax = plot_behavior_events(dataset, ax)
        ax = restrict_axes(xmin, xmax, interval, ax)
        plt.tight_layout()
        if save:
            save_figure(fig, figsize, dataset.analysis_dir, fig_title='cell ' + str(cell), folder='behavior_events_traces')
            plt.close()
            ax = None
    return ax
#
# # In[163]:
#
# def plot_behavior_events_run_speed(dataset, xmin=360, length=3, ax=None, save=False):
#     sns.set_context('notebook', font_scale=2.5, rc={'lines.markeredgewidth': 1})
#     xmax = xmin + 60 * length
#     interval = 20
#     if ax is None:
#         figsize = (20, 3)
#         fig, ax = plt.subplots(figsize=figsize)
#     # vis_times = dataset.sync['visualFrames']['timestamps']
#     vis_times = dataset.sync['visualFrames']['timestamps'][:dataset.run_speed.shape[0]]
#     ax = plot_trace(vis_times, dataset.run_speed, ax, ylabel='run speed\n(cm/s)', title='running speed')
#     ax = restrict_axes(xmin, xmax, interval, ax)
#     ax = plot_behavior_events(dataset, ax)
#     ax.set_yticks(np.arange(0, 70, 20))
#     plt.tight_layout()
#     if save:
#         save_figure(fig, figsize, dataset.analysis_dir, fig_title='run_speed', folder='behavior_events_traces')
#         plt.close()
#         ax = None
#     return ax
#
#
# # In[163]:
#
# def plot_all_trials_heatmap(df, cell_list, dataset, save=False, ax=None):
#     for cell in cell_list:
#         responses = df[(df.cell == cell)].responses.values
#         response_matrix = np.empty((responses.shape[0], responses[0].shape[0]))
#         for pos, trial in enumerate(range(responses.shape[0])[::-1]):
#             response_matrix[trial, :] = responses[trial]
#         if ax is None:
#             figsize = (5, 15)
#             fig, ax = plt.subplots(figsize=figsize)
#         cax = ax.pcolormesh(response_matrix, cmap='jet')
#         plt.colorbar(cax, ax=ax)
#         ax.set_ylim(0, response_matrix.shape[0])
#         ax.set_xlim(0, response_matrix.shape[1])
#         ax.set_title('roi - ' + str(cell) + ' all trials')
#         ax.set_ylabel('trial')
#         ax.set_xlabel('time (s)')
#         ax.set_xticks(np.arange(0, (dataset.window[1] * 30 - (dataset.window[0] * 30) + 30), 30));
#         ax.set_xticklabels(np.arange(dataset.window[0], dataset.window[1] + 1, 1));
#         y_array = np.arange(0, responses.shape[0] + 1, 50)
#         ax.set_yticks(y_array);
#         ax.set_yticklabels(y_array[::-1]);
#         if save:
#             plt.tight_layout()
#             save_figure(fig, figsize, dataset.analysis_dir, fig_title='roi_' + str(cell), folder='all_trials_heatmap')
#             plt.close()
#             ax = None
#     return ax
#
#
# # In[284]:

def plot_transition_type_heatmap(ra, cell_list, cmap='jet', vmax=None, save=False, ax=None, colorbar=True):
    response_types = ['HIT', 'MISS', 'FA', 'CR']
    df = ra.response_df
    figsize = (15, 10)
    rows = 2
    cols = len(df.change_code.unique()) / 2
    for cell in cell_list:
        if ax is None:
            fig, ax = plt.subplots(rows, cols, figsize=figsize, sharex=True);
            ax = ax.ravel();
        resp_types = []
        for i, change_code in enumerate(np.sort(df.change_code.unique())):
            im_df = df[(df.cell == cell) & (df.change_code == change_code) & (df.trial_type != 'autorewarded')]
            n_frames = im_df.response.values[0].shape[0]
            n_trials = im_df.response.shape[0]
            response_matrix = np.empty((n_trials, n_frames))
            response_type_list = []
            segments = []
            idx = 0
            for y, response_type in enumerate(response_types):
                segments.append(idx)
                sub_df = im_df[(im_df.behavioral_response_type == response_type)]
                responses = sub_df.response.values
                for pos, trial in enumerate(range(responses.shape[0])[::-1]):
                    #             print idx,pos,trial
                    response_matrix[idx, :] = responses[trial]
                    response_type_list.append(response_type)
                    idx += 1
                segments.append(idx)
                if vmax:
                    cax = ax[i].pcolormesh(response_matrix, cmap=cmap, vmax=vmax, vmin=0);
                else:
                    cax = ax[i].pcolormesh(response_matrix, cmap=cmap);
                ax[i].set_ylim(0, response_matrix.shape[0]);
                ax[i].set_xlim(0, response_matrix.shape[1]);
                ax[i].set_yticks(segments);
                ax[i].set_xlabel('time (s)');
                ax[i].set_xticks(np.arange(0, (ra.trial_window[1] * 30 - (ra.trial_window[0] * 30) + 30), 30));
                ax[i].set_xticklabels(np.arange(ra.trial_window[0], ra.trial_window[1] + 1, 1));
                change_im = ra.stim_codes[ra.stim_codes.stim_code == change_code].image_name.values[0].split('_')[0]
                ax[i].set_title(str(change_code));
            resp_types.append(response_type_list)
            if colorbar:
                plt.colorbar(cax, ax=ax[i]);
        plt.tight_layout()
        if save:
            if vmax is None:
                save_figure(fig, figsize, ra.analysis_dir, fig_title='roi_' + str(cell),
                            folder='transition_type_heatmap')
            else:
                save_figure(fig, figsize, ra.analysis_dir, fig_title='roi_' + str(cell),
                            folder='transition_type_heatmap_vmax', formats=['.png'])
            plt.close()
            ax = None
    return ax

#
# # In[279]:
#
# def plot_mean_response_heatmap(dataset, sdf, cell_list, save=False, ax=None):
#     response_types = ['HIT', 'MISS', 'FA', 'CR']
#     df = dataset.df
#     if ax is None:
#         figsize = (20, 10)
#         fig, ax = plt.subplots(2, 4, figsize=figsize, sharex=True, sharey=True)
#         ax = ax.ravel()
#     i = 0
#     for x, change_code in enumerate(np.sort(df.change_code.unique())):
#         for y, response_type in enumerate(response_types):
#             n_frames = df.responses_dFF.values[0].shape[0]
#             n_cells = len(df.cell.unique())
#             response_matrix = np.empty((n_cells, n_frames))
#             for c, cell in enumerate(cell_list):
#                 im_df = df[(df.cell == cell) & (df.change_code == change_code) & (df.response_type == response_type) & (
#                 df.trial_type != 'autorewarded')]
#                 response = im_df.responses_dFF.mean()
#                 response_matrix[c, :] = response
#             im = ax[i].pcolormesh(response_matrix, cmap='RdBu', vmin=-0.5, vmax=0.5)
#             ax[i].set_ylim(0, response_matrix.shape[0])
#             ax[i].set_xlim(0, response_matrix.shape[1])
#             ax[i].set_xticks(np.arange(0, (dataset.window[1] * 30 - (dataset.window[0] * 30) + 30), 60));
#             ax[i].set_xticklabels(np.arange(dataset.window[0], dataset.window[1] + 1, 2));
#             if i <= 3:
#                 ax[i].set_title(response_type)
#             if i >= 4:
#                 ax[i].set_xlabel('time (s)')
#             i += 1
#     ax[0].set_ylabel('cells')
#     ax[4].set_ylabel('cells')
#     cax = fig.add_axes([0.92, 0.12, 0.03, 0.78])
#     cb = fig.colorbar(im, cax=cax)
#     cb.set_label('mean dF/F')
#     # plt.tight_layout()
#     if save:
#         save_figure(fig, figsize, dataset.analysis_dir, fig_title='all_cells_mean_response_rb',
#                     folder='transition_type_heatmap')
#         plt.close()
#     return ax
#
#
# # In[279]:
#
# def plot_mean_response_heatmap_dF(dataset, sdf, cell_list, cmap='jet', save=False, ax=None):
#     response_types = ['HIT', 'MISS', 'FA', 'CR']
#     df = dataset.df
#     if ax is None:
#         figsize = (20, 10)
#         fig, ax = plt.subplots(2, 4, figsize=figsize, sharex=True, sharey=True)
#         ax = ax.ravel()
#     i = 0
#     for x, change_code in enumerate(np.sort(df.change_code.unique())):
#         for y, response_type in enumerate(response_types):
#             n_frames = df.responses.values[0].shape[0]
#             n_cells = len(df.cell.unique())
#             response_matrix = np.empty((n_cells, n_frames))
#             for c, cell in enumerate(cell_list):
#                 im_df = df[(df.cell == cell) & (df.change_code == change_code) & (df.response_type == response_type) & (
#                 df.trial_type != 'autorewarded')]
#                 response = im_df.responses.mean()
#                 response_matrix[c, :] = response
#             im = ax[i].pcolormesh(response_matrix, cmap=cmap, vmax=np.percentile(response_matrix, 95))
#             #            im = ax[i].pcolormesh(response_matrix,cmap=cmaps.plasma,vmax=0.5)
#             ax[i].set_ylim(0, response_matrix.shape[0])
#             ax[i].set_xlim(0, response_matrix.shape[1])
#             ax[i].set_xticks(np.arange(0, (dataset.window[1] * 30 - (dataset.window[0] * 30) + 30), 60));
#             ax[i].set_xticklabels(np.arange(dataset.window[0], dataset.window[1] + 1, 2));
#             if i <= 3:
#                 ax[i].set_title(response_type)
#             if i >= 4:
#                 ax[i].set_xlabel('time (s)')
#             i += 1
#     ax[0].set_ylabel('cells')
#     ax[4].set_ylabel('cells')
#     cax = fig.add_axes([0.92, 0.12, 0.03, 0.78])
#     cb = fig.colorbar(im, cax=cax)
#     cb.set_label('mean dF/F')
#     # plt.tight_layout()
#     if save:
#         if cmap == cmaps.plasma:
#             cmap = 'plasma'
#         save_figure(fig, figsize, dataset.analysis_dir, fig_title='all_cells_mean_response_dF_' + cmap,
#                     folder='transition_type_heatmap')
#         plt.close()
#     return ax
#
#
# # In[396]:

def plot_transition_type_heatmap_sig_cells(dataset, sig_cell_list, change_code, save=False, ax=None):
    response_types = ['HIT', 'MISS', 'FA', 'CR']
    df = dataset.df
    if ax is None:
        figsize = (20, 20)
        fig, ax = plt.subplots(6, 8, figsize=figsize, sharex=True, sharey=True)
        ax = ax.ravel()
    for i, cell in enumerate(sig_cell_list):
        #     for i,change_code in enumerate(np.sort(sdf.change_code.unique())):
        im_df = df[(df.cell == cell) & (df.change_code == change_code) & (df.trial_type != 'autorewarded')]
        n_frames = im_df.responses.values[0].shape[0]
        n_trials = im_df.responses.shape[0]
        response_matrix = np.empty((n_trials, n_frames))
        response_type_list = []
        segments = []
        idx = 0
        for y, response_type in enumerate(response_types):
            segments.append(idx)
            sub_df = im_df[(im_df.response_type == response_type)]
            responses = sub_df.responses_dFF.values
            for pos, trial in enumerate(range(responses.shape[0])[::-1]):
                #             print idx,pos,trial
                response_matrix[idx, :] = responses[trial]
                response_type_list.append(response_type)
                idx += 1
            segments.append(idx)
            cax = ax[i].pcolormesh(response_matrix, cmap='RdBu', vmin=-1.5, vmax=1.5)
            ax[i].set_ylim(0, response_matrix.shape[0])
            ax[i].set_xlim(0, response_matrix.shape[1])
            ax[i].set_yticks(segments)
            #             ax[i].set_xlabel('time (s)')
            ax[i].set_xticks(np.arange(0, (dataset.window[1] * 30 - (dataset.window[0] * 30) + 30), 60));
            ax[i].set_xticklabels(np.arange(dataset.window[0], dataset.window[1] + 1, 2));
            ax[i].set_title(str(cell))
            #     plt.colorbar(cax,ax=ax[i])
        plt.tight_layout()
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, fig_title='sig_cells_stim_' + str(change_code),
                    folder='transition_type_heatmap')
        plt.close()
    return ax


#
# # In[397]:
#
# def plot_behavior_performance_sns(pkl, pkl_df, analysis_dir=None, save=False):
#     figsize = (8, 15)
#     fig, ax = plt.subplots(figsize=figsize)
#     pkl_df = pkl_df[pkl_df.trial_type != 'aborted']  # only change trials
#     i = 0
#     for trial in pkl_df.index:
#         licks = pkl_df.lick_times[trial] - pkl_df.change_time[trial]
#         ax.vlines(licks, i, i + 1, color='k', linewidth=1, alpha=0.7)
#         color = get_color_for_response_type(pkl_df.response_type[trial])
#         ax.axhspan(i, i + 1, 0, 200, color=color, alpha=.5)
#         if len(pkl_df.reward_times[trial]) > 0:
#             diff = pkl_df.reward_times[trial][0] - pkl_df.change_time[trial]
#             ax.plot(diff, i + 0.5, '.', color='b', label='reward')
#         ax.vlines(0, i, i + 1, color=[.5, .5, .5], linewidth=1)
#         i += 1
#     ax.grid(False)
#     ax.set_ylim(-1, len(pkl_df) + 1)
#     ax.set_xlim([-0.5, 10])
#     ax.set_xlabel('time after change (s)')
#     ax.set_ylabel('trial #')
#     ax.set_title(pkl['mouseid'] + '-' + pkl['startdatetime'][:10])
#     if save:
#         fig.tight_layout()
#         fig_title = 'behavior_performance_sns'
#         saveFigure(fig, os.path.join(analysis_dir, 'behavior', fig_title), formats=['.png'], size=figsize)
#         plt.close()
#
#
# # In[397]:
#
# def plot_behavior_performance_rk(dataset, save=False, ax=None):
#     if ax is None:
#         figsize = (8, 15)
#         fig, ax = plt.subplots(figsize=figsize)
#     for i, lap in enumerate(dataset.stim_table.global_trial):
#         licks = dataset.pkl_df.lick_times[lap] - dataset.pkl_df.change_time[lap]
#         ax.vlines(licks, i, i + 1, color='k', linewidth=1, alpha=0.7)
#         if dataset.pkl_df.change_image[lap] != dataset.pkl_df.initial_image[lap]:
#             if dataset.pkl_df.change_image[lap] == 'mushroom_black.png':
#                 ax.axhspan(i, i + 1, 0, 200, color='r', alpha=.5)
#             elif dataset.pkl_df.change_image[lap] == 'bird_black.png':
#                 ax.axhspan(i, i + 1, 0, 200, color='k', alpha=.5)
#         if len(dataset.pkl_df.reward_times[lap]) > 0:
#             diff = dataset.pkl_df.reward_times[lap][0] - dataset.pkl_df.change_time[lap]
#             ax.plot(diff, i + 0.5, '.', color='b', label='reward')
#         if dataset.pkl_df.change_image[lap] == dataset.pkl_df.initial_image[lap]:
#             if dataset.pkl_df.change_image[lap] == 'mushroom_black.png':
#                 ax.axhspan(i, i + 1, 0, 200, color='r', alpha=.2)
#             elif dataset.pkl_df.change_image[lap] == 'bird_black.png':
#                 ax.axhspan(i, i + 1, 0, 200, color='k', alpha=.2)
#                 #         ax.axhspan(i, i + 1, 0, 200,color='b',alpha=.3)
#         ax.vlines(0, i, i + 1, color=[.5, .5, .5], linewidth=1)
#     ax.grid(False)
#     ax.set_ylim(-1, len(dataset.stim_table) + 1)
#     ax.set_xlim([-0.5, 10])
#     ax.set_xlabel('time after change (s)')
#     ax.set_ylabel('trial #')
#     ax.set_title(dataset.pkl['mouseid'] + '-' + dataset.pkl['startdatetime'][:10])
#     if save:
#         fig_title = 'behavior_performance_rk'
#         save_figure(fig, figsize, dataset.analysis_dir, fig_title, 'behavior')
#     # saveFigure(fig,os.path.join(dataset.analysis_dir,fig_title),formats = ['.png'],size=figsize)
#     return ax
#
#
# # In[ ]:
#
# def plot_behavior_performance(dataset, save=False, ax=None):
#     if ax is None:
#         figsize = (8, 15)
#         fig, ax = plt.subplots(figsize=figsize)
#     for i, lap in enumerate(dataset.stim_table.global_trial):
#         # plot licks
#         licks = dataset.pkl_df.lick_times[lap] - dataset.pkl_df.change_time[lap]
#         ax.vlines(licks, i, i + 1, color='k', linewidth=1, alpha=0.7)
#         # change trials - blue
#         if dataset.pkl_df.change_image[lap] != dataset.pkl_df.initial_image[lap]:
#             ax.axhspan(i, i + 1, 0, 200, color='b', alpha=.3)
#         # change trials - green
#         if dataset.pkl_df.change_image[lap] == dataset.pkl_df.initial_image[lap]:
#             ax.axhspan(i, i + 1, 0, 200, color='g', alpha=.3)
#         # plot rewards
#         if len(dataset.pkl_df.reward_times[lap]) > 0:
#             diff = dataset.pkl_df.reward_times[lap][0] - dataset.pkl_df.change_time[lap]
#             ax.plot(diff, i + 0.5, '.', color='b', label='reward')
#         ax.vlines(0, i, i + 1, color=[.5, .5, .5], linewidth=1)
#     ax.grid(False)
#     ax.set_ylim(-1, len(dataset.stim_table) + 1)
#     ax.set_xlim([-0.5, 10])
#     ax.set_xlabel('time after change (s)')
#     ax.set_ylabel('trial #')
#     ax.set_title(dataset.pkl['mouseid'] + '-' + dataset.pkl['startdatetime'][:10])
#     if save:
#         fig_title = 'behavior_performance'
#         save_figure(fig, figsize, dataset.analysis_dir, fig_title, 'behavior')
#     # saveFigure(fig,os.path.join(dataset.analysis_dir,fig_title),formats = ['.png'],size=figsize)
#     return ax
#
#
# # In[ ]:
#
# def plot_mean_image_response_trial_type(dataset, mdf, cell_list, ax=None, save=False):
#     fig_folder = 'mean_image_response_trial_type'
#     fig_dir = os.path.join(dataset.analysis_dir, fig_folder)
#     if not os.path.exists(fig_dir): os.mkdir(fig_dir)
#     sc = dataset.stim_codes
#     stim_codes = np.sort(mdf.stim_code.unique())
#     trial_types = np.sort(mdf.trial_type.unique())[1:]
#     colors = ['r', 'k']
#     alphas = [0.5, 1]
#     figsize = (5, 5)
#     for cell in cell_list:
#         if ax is None:
#             fig, ax = plt.subplots(figsize=figsize)
#         names = []
#         for x, stim_code in enumerate(stim_codes):
#             name = sc[sc.stim_code == stim_code].image_name.values[0].split('_')[0]
#             names.append(name)
#             for y, trial_type in enumerate(trial_types):
#                 cond = mdf[(mdf.cell == cell) & (mdf.stim_code == stim_code) & (mdf.trial_type == trial_type)]
#                 if len(cond) > 0:
#                     ax.errorbar(stim_code, cond.mean_response, yerr=cond.sem_response, color=colors[x], alpha=alphas[y])
#                     ax.plot(stim_code, cond.mean_response, 'o', color=colors[x], alpha=alphas[y], label=trial_type)
#         ax.set_xlim([stim_codes[0] - 0.5, stim_codes[1] + 0.5])
#         ax.set_xticks(stim_codes);
#         ax.set_xticklabels(names, rotation='vertical');
#         ax.set_ylabel('mean dF/F')
#         fig_title = 'roi_' + str(cell)
#         ax.set_title(fig_title)
#         plt.legend(loc=9, ncol=1, bbox_to_anchor=(1.2, 1.0));
#         plt.tight_layout()
#         ax = None
#         if save:
#             saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#             plt.close()
#
#
# # In[ ]:
#
# def plot_mean_trial_type_response_by_image(dataset, mdf, cell_list, ax=None, save=False):
#     fig_folder = 'mean_trial_type_response_by_image'
#     fig_dir = os.path.join(dataset.analysis_dir, fig_folder)
#     if not os.path.exists(fig_dir): os.mkdir(fig_dir)
#     df = dataset.df
#     colors = ['r', 'k']
#     figsize = (5, 5)
#     for cell in cell_list:
#         if ax is None:
#             fig, ax = plt.subplots(figsize=figsize)
#         cdf = df[df.cell == cell]
#         mdf = sa.get_mean_df(cdf, conditions, mean_window)
#         mdf = mdf[mdf.trial_type != 'autorewarded']
#         values = sa.get_values_for_conditions(mdf, conditions)
#         for i in range(len(values)):
#             vals = values[i]
#             cond = sa.get_cond_for_vals(mdf, conditions, vals)
#             if len(cond) > 0:
#                 if cond.trial_type.values[0] == 'go':
#                     alpha = 1
#                 elif cond.trial_type.values[0] == 'catch':
#                     alpha = 0.5
#                 ax.errorbar(vals[0], cond.mean_response, yerr=cond.sem_response, color=colors[vals[0]], alpha=alpha)
#                 ax.plot(vals[0], cond.mean_response, 'o', color=colors[vals[0]], alpha=alpha, label=vals[1])
#         xpos = mdf.change_code.unique()
#         ax.set_xlim([xpos[0] - 0.5, xpos[1] + 0.5])
#         ax.set_xticks(xpos);
#         ax.set_xticklabels(['mushroom', 'bird'], rotation='vertical');
#         ax.set_ylabel('mean dF/F')
#         fig_title = 'roi_' + str(cell)
#         ax.set_title(fig_title)
#         plt.tight_layout()
#         ax = None
#         if save:
#             saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#             plt.close()
#         ax = None
#
#
# # In[ ]:
#
# def plot_trial_type_responses_separated(cell, mdf, dataset, save=False, ax=None):
#     cdf = mdf[(mdf.cell == cell)]
#     figsize = (10, 10)
#     stim_codes = np.sort(cdf.stim_code.unique())
#     response_types = ['HIT', 'FA', 'MISS', 'CR']
#     colors = get_colors_for_response_types(response_types)
#     for stim_code in stim_codes:
#         if ax is None:
#             fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
#             ax = ax.ravel()
#         for i, response_type in enumerate(response_types):
#             trace = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].mean_trace.values[0]
#             sem = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].sem_trace.values[0]
#             times = np.arange(0, len(trace), 1)
#             xlabels = np.arange(0, len(trace) + 10, 60)
#             ax[i].fill_between(times, trace + sem, trace - sem, color=colors[i], alpha=0.5)
#             ax[i].plot(times, trace, color=colors[i], label=response_type, linewidth=2)
#             ax[i].set_title(response_type)
#             ax[i].set_xlim([0, len(trace)])
#             ax[i].set_xticks(xlabels)
#             ax[i].set_xticklabels(xlabels / 30 + dataset.window[0])
#         ax[0].set_ylabel('dF/F')
#         ax[2].set_ylabel('dF/F')
#         ax[2].set_xlabel('time after change (sec)')
#         ax[3].set_xlabel('time after change (sec)')
#         plt.suptitle('transition to ' + sa.get_image_for_code(dataset, stim_code))
#         ax = None
#         if save:
#             fig.tight_layout()
#             fig_title = 'roi_' + str(cell) + '_' + str(stim_code)
#             fig_dir = os.path.join(dataset.analysis_dir, 'trial_type_responses_separated')
#             if not os.path.exists(fig_dir):
#                 os.mkdir(fig_dir)
#             saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#             plt.close()
#
#
# # In[ ]:
# def plot_trial_type_responses_column(cell, mdf, dataset, save=False, ax=None):
#     cdf = mdf[(mdf.cell == cell)]
#     figsize = (5, 12)
#     stim_codes = np.sort(cdf.stim_code.unique())
#     response_types = ['CR', 'FA', 'MISS', 'HIT']
#     colors = get_colors_for_response_types(response_types)
#     for stim_code in stim_codes:
#         if ax is None:
#             fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True, sharey=True)
#             ax = ax.ravel()
#         for i, response_type in enumerate(response_types):
#             trace = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].mean_trace.values[0]
#             sem = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].sem_trace.values[0]
#             times = np.arange(0, len(trace), 1)
#             xlabels = np.arange(0, len(trace) + 10, 60)
#             ax[i].fill_between(times, trace + sem, trace - sem, color=colors[i], alpha=0.5)
#             ax[i].plot(times, trace, color=colors[i], label=response_type, linewidth=2)
#             ax[i].set_title(response_type)
#             ax[i].set_xlim([0, len(trace)])
#             ax[i].set_xticks(xlabels)
#             ax[i].set_xticklabels(xlabels / 30 + dataset.window[0])
#             ax[i].set_ylabel('dF/F')
#         ax[3].set_xlabel('time after change (sec)')
#         plt.suptitle('transition to ' + sa.get_image_for_code(dataset, stim_code), x=0.58, y=1,
#                      horizontalalignment='center')
#         ax = None
#         fig.tight_layout()
#         if save:
#
#             fig_title = 'roi_' + str(cell) + '_' + str(stim_code)
#             fig_dir = os.path.join(dataset.analysis_dir, 'trial_type_responses_column')
#             if not os.path.exists(fig_dir):
#                 os.mkdir(fig_dir)
#             saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#             plt.close()
#
#
# # In[ ]:
#
# def plot_trial_type_responses_row(cell, mdf, dataset, save=False, ax=None):
#     cdf = mdf[(mdf.cell == cell)]
#     figsize = (15, 8)
#     stim_codes = np.sort(cdf.stim_code.unique())
#     response_types = ['HIT', 'MISS', 'CR', 'FA']
#     colors = get_colors_for_response_types(response_types)
#     if ax is None:
#         fig, ax = plt.subplots(2, 4, figsize=figsize, sharey=True)
#         ax = ax.ravel()
#     i = 0
#     for stim_code in stim_codes:
#         for x, response_type in enumerate(response_types):
#             trace = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].mean_trace.values[0]
#             sem = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].sem_trace.values[0]
#             times = np.arange(0, len(trace), 1)
#             xlabels = np.arange(0, len(trace) + 10, 60)
#             ax[i].fill_between(times, trace + sem, trace - sem, color=colors[x], alpha=0.5)
#             ax[i].plot(times, trace, color=colors[x], label=response_type, linewidth=2)
#             ax[i].set_title(response_type)
#             ax[i].set_xlim([0, len(trace)])
#             ax[i].set_xticks(xlabels)
#             ax[i].set_xticklabels(xlabels / 30 + dataset.window[0])
#             #            if i>3:
#             ax[i].set_xlabel('time after change (sec)', fontsize=16)
#             i += 1
#         ax[0].set_ylabel('dF/F')
#         ax[4].set_ylabel('dF/F')
#         #         plt.suptitle('transition to '+sa.get_image_for_code(dataset,stim_code),x=0.58,y=1,horizontalalignment='center')
#         fig.tight_layout()
#     if save:
#         fig_title = 'roi_' + str(cell)
#         fig_dir = os.path.join(dataset.analysis_dir, 'trial_type_responses_row')
#         if not os.path.exists(fig_dir):
#             os.mkdir(fig_dir)
#         saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#         plt.close()
#
#
# # In[ ]:
#
# def plot_trial_type_responses_overlay(cell, mdf, dataset, save=False, ax=None):
#     cdf = mdf[(mdf.cell == cell)]
#     stim_codes = np.sort(mdf.stim_code.unique())
#     response_types = ['HIT', 'MISS', 'CR', 'FA']
#     #    response_types = ['CR','MISS','HIT','FA']
#     if ax is None:
#         figsize = (15, 8)
#         fig, ax = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)
#         ax = ax.ravel()
#     colors = get_colors_for_stim_codes(stim_codes)
#     for x, stim_code in enumerate(stim_codes):
#         c = 0
#         for i, response_type in enumerate(response_types):
#             image = sa.get_image_for_code(dataset, stim_code)
#             trace = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].mean_trace.values[0]
#             sem = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].sem_trace.values[0]
#             times = np.arange(0, len(trace), 1)
#             xlabels = np.arange(0, len(trace) + 1, 60)
#             ax[i].fill_between(times, trace + sem, trace - sem, color=colors[x], alpha=0.5)
#             ax[i].plot(times, trace, color=colors[x], linewidth=2, label=image if c == 0 else "")
#             ax[i].set_title(response_type)
#             ax[i].set_xlim([0, len(trace)])
#             ax[i].set_xticks(xlabels)
#             ax[i].set_xticklabels((xlabels / 30) + dataset.window[0])
#             ax[i].set_xlabel('time after change (sec)')
#         ax[0].set_ylabel('dF/F')
#     plt.legend(bbox_to_anchor=(1.9, 1.0))
#     if save:
#         plt.tight_layout()
#         plt.gcf().subplots_adjust(right=.83)
#         fig_title = 'roi_' + str(cell)
#         fig_dir = os.path.join(dataset.analysis_dir, 'trial_type_responses_overlay')
#         if not os.path.exists(fig_dir):
#             os.mkdir(fig_dir)
#         saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#         plt.close()
#
#
# # In[ ]:
#
# def plot_stim_responses(cell, mdf, dataset, window=[-2, 2], save=False, ax=None):
#     cdf = mdf[(mdf.cell == cell)]
#     stim_codes = np.sort(mdf.stim_code.unique())
#     response_types = ['HIT', 'FA', 'MISS', 'CR']
#     #    response_types = ['CR','MISS','HIT','FA']
#     if ax is None:
#         figsize = (7, 7)
#         fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
#         ax = ax.ravel()
#     colors = get_colors_for_stim_codes(stim_codes)
#     frames_range = [(window[0] - dataset.window[0]) * 30, (dataset.window[1] + window[1]) * 30]
#     n_frames = window[1] - window[0] * 60
#     times = np.arange(0, n_frames, 1)
#     for x, stim_code in enumerate(stim_codes):
#         c = 0
#         for i, response_type in enumerate(response_types):
#             image = sa.get_image_for_code(dataset, stim_code)
#             trace = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].mean_trace.values[0]
#             trace = trace[frames_range[0]:frames_range[1]]
#             sem = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].sem_trace.values[0]
#             sem = sem[frames_range[0]:frames_range[1]]
#             times = np.arange(0, len(trace), 1)
#             ax[i].fill_between(times, trace + sem, trace - sem, color=colors[x], alpha=0.5)
#             ax[i].plot(times, trace, color=colors[x], linewidth=2, label=image if c == 0 else "")
#             ax[i].set_title(response_type)
#             ax[i].set_xlim([0, n_frames])
#             ax[i].set_xticks(np.arange(0, n_frames + 30, 60))
#             ax[i].set_xticklabels(np.arange(window[0], window[1] + 1, 2))
#     ax[2].set_xlabel('time after change (sec)')
#     ax[0].set_ylabel('dF/F')
#     ax[3].set_xlabel('time after change (sec)')
#     ax[2].set_ylabel('dF/F')
#     plt.legend(bbox_to_anchor=(0.35, 2.75))
#     if save:
#         plt.tight_layout()
#         plt.gcf().subplots_adjust(top=.85)
#         fig_title = 'roi_' + str(cell)
#         fig_dir = os.path.join(dataset.analysis_dir, 'stim_responses_' + str(window[1]) + 's')
#         if not os.path.exists(fig_dir):
#             os.mkdir(fig_dir)
#         saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#         plt.close()
#
#
# # In[ ]:
#
# def plot_trial_types_selectivity(dataset, mdf, cell_list, window=2, save=False, ax=None):
#     response_types = ['HIT', 'MISS', 'FA', 'CR']
#     colors = get_colors_for_response_types(response_types)
#     frames_range = [(-window + dataset.window[1]) * 30, (window + dataset.window[1]) * 30]
#     frames = [frames_range[0] - frames_range[0], frames_range[1] - frames_range[0]]
#     xlabels = np.arange((-window), (window) + 1, 1)
#     xtimes = np.arange(0, (window * 2 * 30) + 5, 30)
#     for cell in cell_list:
#         cdf = mdf[(mdf.cell == cell)]
#         stim_codes = np.sort(cdf.stim_code.unique())
#         stim_colors = get_colors_for_stim_codes(stim_codes)
#         figsize = (6, 6)
#         if ax is None:
#             fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
#             ax = ax.ravel()
#         stim_code = 0
#         i = 0
#         for y, stim_code in enumerate([0, 1]):
#             if y == 1:
#                 i += 1
#             for x, response_type in enumerate(response_types):
#                 trace = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].mean_trace.values[0][
#                         frames_range[0]:frames_range[1]]
#                 sem = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].sem_trace.values[0][
#                       frames_range[0]:frames_range[1]]
#                 times = np.arange(0, len(trace), 1)
#                 #                xlabels = np.arange(0,len(trace)+1,60)
#                 if x == 2:
#                     i = i + 1
#                 ax[i].fill_between(times, trace + sem, trace - sem, color=colors[x], alpha=0.5)
#                 ax[i].plot(times, trace, color=colors[x], linewidth=2, label=response_type if y == 0 else "")
#                 ax[i].set_xlim([0, len(trace)])
#                 ax[i].set_xticks(xtimes)
#                 ax[i].set_xticklabels(xlabels)
#                 if response_type == 'HIT' or response_type == 'MISS':
#                     ax[i].axvspan(frames[0], frames[1] / 2, zorder=-1, alpha=0.2,
#                                   color=stim_colors[stim_codes[stim_codes != stim_code][0]])
#                     ax[i].axvspan(frames[1] / 2, frames[1], zorder=-1, alpha=0.2, color=stim_colors[stim_code])
#                 else:
#                     ax[i].axvspan(frames[0], frames[1], zorder=-1, alpha=0.2, color=stim_colors[stim_code])
#         ax[0].set_title('Go')
#         ax[1].set_title('Catch')
#         ax[2].set_xlabel('time after change (sec)')
#         ax[3].set_xlabel('time after change (sec)')
#         ax[0].set_ylabel('dF/F')
#         ax[2].set_ylabel('dF/F')
#         #        ymin,ymax = ax[0].get_ylim()
#         #        ax[0].text(-50,(ymax+ymin)/2,'mushroom',horizontalalignment='center',verticalalignment='center',rotation=90)
#         #        ymin,ymax = ax[2].get_ylim()
#         #        ax[2].text(-50,(ymax+ymin)/2,'bird',horizontalalignment='center',verticalalignment='center',rotation=90)
#         #        ax[0].legend()
#         #        ax[1].legend()
#         fig.tight_layout()
#         plt.suptitle('cell ' + str(cell), x=0.55, y=1, horizontalalignment='center')
#         if save:
#             plt.gcf().subplots_adjust(left=0.2)
#             save_figure(fig, figsize, dataset.analysis_dir, fig_title='roi_' + str(cell),
#                         folder='trial_types_selectivity_' + str(window))
#             plt.close()
#         return ax
#
#
# # In[ ]:

def plot_trial_types_selectivity_ns(dataset, mdf, cell_list, window=2, save=False, ax=None):
    response_types = ['HIT', 'MISS', 'FA', 'CR']
    #    frame_int = 0.7*30 #flashes happen every 700ms
    #    stim_start_frame = (dataset.mean_window[0]*30) #change time
    #    x_start_frame = (dataset.mean_window[0]*30) - (window*30) #start x axis 2 seconds before change time
    #    num_flashes = 6
    #    last_flash = stim_start_frame+(frame_int*num_flashes)
    #    flash_onsets = np.arange(stim_start_frame,last_flash,frame_int)

    frames_range = [(-window + dataset.window[1]) * 30, (window + dataset.window[1]) * 30]
    # frames = [frames_range[0]-frames_range[0],frames_range[1]-frames_range[0]]
    xlabels = np.arange((-window), (window) + 1, 1)
    xtimes = np.arange(0, (window * 2 * 30) + 5, 30)
    stim_codes = np.sort(mdf.stim_code.unique())
    stim_colors = get_colors_for_stim_codes(stim_codes)
    stim_df = dataset.stim_codes
    for cell in cell_list:
        cdf = mdf[(mdf.cell == cell)]
        if ax is None:
            figsize = (15, 4)
            fig, ax = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)
            ax = ax.ravel();
        for i, response_type in enumerate(response_types):
            for y, stim_code in enumerate(stim_codes):
                tdf = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)]
                if len(tdf) > 0:
                    trace = tdf.mean_trace.values[0][frames_range[0]:frames_range[1]]
                    sem = tdf.sem_trace.values[0][frames_range[0]:frames_range[1]]
                    times = np.arange(0, len(trace), 1)
                    ax[i].fill_between(times, trace + sem, trace - sem, color=stim_colors[y], alpha=0.5)
                    image_name = stim_df[stim_df.stim_code == stim_code].image_name.values[0]
                    ax[i].plot(times, trace, color=stim_colors[y], linewidth=3, label=image_name)
                    #                    flash_means = tdf.flash_means.values[0]
                    #                    for x in range(len(flash_onsets)):
                    #                        ax[i].axhspan(ymin=0.29,ymax=0.31,color='gray')
                    #                        pu.addSpan(ax[i],flash_onsets[x],flash_onsets[x]+0.5*30,color='gray')
                    ax[i].set_xlim([0, len(trace)])
                    ax[i].set_xticks(xtimes)
                    ax[i].set_xticklabels(xlabels)
                    ax[i].set_title(response_type)
                    ax[i].set_xlabel('time after change (sec)')
        ax[0].set_ylabel('dF/F')

        if save:
            ax[1].legend(bbox_to_anchor=(1.55, 1.05))
            plt.suptitle('cell ' + str(cell), x=0.47, y=1, horizontalalignment='center')
            fig.tight_layout()
            plt.gcf().subplots_adjust(right=0.85)
            save_figure(fig, figsize, dataset.analysis_dir, fig_title='roi_' + str(cell),
                        folder='trial_types_selectivity_' + str(window))
            plt.close()
            ax = None
    return ax


#
#
# # In[ ]:

def plot_trial_types_selectivity_pref_image(dataset, mdf, sdf, cell_list, response_types=['HIT', 'MISS'],
                                            window=[-1, 2], save=False, ax=None):
    #    response_types = ['HIT','MISS','FA','CR']
    #    response_types = ['HIT','MISS']
    #    colors = get_colors_for_response_types(response_types)
    colors = get_colors_for_response_types(['CR', 'FA'])
    frames_range = [(window[0] + dataset.window[1]) * 30, (window[1] + dataset.window[1]) * 30]
    # frames = [frames_range[0]-frames_range[0],frames_range[1]-frames_range[0]]
    xlabels = np.arange((window[0]), (window[1]) + 1, 1)
    xtimes = np.arange(0, ((-window[0] + window[1]) * 30) + 5, 30)
    #    stim_codes = np.sort(mdf.stim_code.unique())
    #    stim_colors = get_colors_for_stim_codes(stim_codes)
    stim_df = dataset.stim_codes
    for cell in cell_list:
        cdf = mdf[(mdf.cell == cell)]
        pref_stim = sdf[sdf.cell == cell].pref_stim.values[0]
        pref_image = stim_df[stim_df.stim_code == pref_stim].image_name.values[0]
        if ax is None:
            figsize = (5, 4)
            fig, ax = plt.subplots(figsize=figsize)
            labels = True
        else:
            labels = False
        for i, response_type in enumerate(response_types):
            tdf = cdf[(cdf.stim_code == pref_stim) & (cdf.response_type == response_type)]
            if len(tdf) > 0:
                trace = tdf.mean_trace.values[0][frames_range[0]:frames_range[1]]
                sem = tdf.sem_trace.values[0][frames_range[0]:frames_range[1]]
                times = np.arange(0, len(trace), 1)
                ax.fill_between(times, trace + sem, trace - sem, color=colors[i], alpha=0.5)
                ax.plot(times, trace, color=colors[i], linewidth=3, label=response_type)
                ax.set_xlim([0, len(trace)])
                ax.set_xticks(xtimes)
                ax.set_xticklabels(xlabels)
                if labels:
                    ax.set_xlabel('time after change (sec)')
                    ax.set_ylabel('dF/F')
                    #        plt.legend(bbox_to_anchor=(1.5, 1.05))

        if save:
            ax.set_title('cell ' + str(cell))
            fig.tight_layout()
            plt.gcf().subplots_adjust(right=0.75)
            save_figure(fig, figsize, dataset.analysis_dir, fig_title='roi_' + str(cell) + '_ttspi',
                        folder='trial_types_selectivity_pref_image')
            plt.close()
        else:
            plt.legend(loc='upper right', bbox_to_anchor=(0.75, 1.35))
        if labels:
            ax = None
    return ax


# # In[ ]:
#
# def plot_avg_trial_type_selectivity(dataset, mdf, sdf, response_types, save=False, ax=None):
#     frames_range = [4 * 30, 8 * 30]
#     times = np.arange(0, 120, 1)
#     colors = get_colors_for_response_types(response_types)
#     figsize = (5, 8)
#     fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)
#     for x, stim_code in enumerate(np.sort(mdf.stim_code.unique())):
#         if stim_code == 0:
#             cells = sdf[(sdf.stim_SI >= 0.6)].cell.values
#         elif stim_code == 1:
#             cells = sdf[(sdf.stim_SI <= -0.6)].cell.values
#         for y, response_type in enumerate(response_types):
#             traces_list = []
#             for cell in cells:
#                 trace = mdf[(mdf.cell == cell) & (mdf.response_type == response_type) & (
#                 mdf.stim_code == stim_code)].mean_trace.values[0]
#                 trace = trace[frames_range[0]:frames_range[1]]
#                 traces_list.append(trace)
#             traces = np.asarray(traces_list)
#             avg_trace = np.mean(traces, axis=0)
#             sem = np.std(traces, axis=0) / np.sqrt(len(traces))
#             ax[x].plot(times, avg_trace, color=colors[y], label=response_type, linewidth=3)
#             ax[x].fill_between(times, avg_trace + sem, avg_trace - sem, facecolor=colors[y], alpha=0.2)
#             ax[x].set_title(sa.get_image_for_code(dataset, stim_code) + ' selective cells (n=' + str(len(cells)) + ')')
#             ax[x].set_xlim([0, 120])
#             ax[x].set_xticks(np.arange(0, 125, 30))
#             ax[x].set_xticklabels(np.arange(-2, 3, 1))
#             ax[x].set_ylabel('dF/F')
#             ax[1].set_xlabel('time after change (sec)')
#             ax[x].set_ylim([0, 1.6])
#             plt.legend()
#             plt.tight_layout()
#     save_figure(fig, figsize, dataset.analysis_dir,
#                 fig_title=response_types[0] + '_' + response_types[1] + 'selective_cells', folder='selectivity')
#
#
# # In[ ]:
#
# def plot_trial_type_responses_with_stim(dataset, mdf, cell, window=2, save=False, ax=None):
#     sns.set_context('notebook', font_scale=1.7, rc={'lines.markeredgewidth': 1.5})
#     frames_range = [(-window + dataset.window[1]) * 30, (window + dataset.window[1]) * 30]
#     frames = [frames_range[0] - frames_range[0], frames_range[1] - frames_range[0]]
#
#     xlabels = np.arange((-window), (window) + 1, 1)
#     xtimes = np.arange(0, (window * 2 * 30) + 5, 30)
#     # def plot_trial_type_responses_row(cell,mdf,dataset,save=False,ax=None):
#     cdf = mdf[(mdf.cell == cell)]
#     figsize = (15, 8)
#     stim_codes = np.sort(cdf.stim_code.unique())
#     stim_colors = get_colors_for_stim_codes(stim_codes)
#     response_types = ['HIT', 'MISS', 'CR', 'FA']
#     colors = get_colors_for_response_types(response_types)
#     if ax is None:
#         fig, ax = plt.subplots(2, 4, figsize=figsize, sharey=True)
#         ax = ax.ravel()
#     i = 0
#     for y, stim_code in enumerate(stim_codes):
#         for x, response_type in enumerate(response_types):
#             trace = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].mean_trace.values[0][
#                     frames_range[0]:frames_range[1]]
#             sem = cdf[(cdf.stim_code == stim_code) & (cdf.response_type == response_type)].sem_trace.values[0][
#                   frames_range[0]:frames_range[1]]
#             times = np.arange(0, len(trace), 1)
#             ax[i].fill_between(times, trace + sem, trace - sem, color=colors[x], alpha=0.6)
#             ax[i].plot(times, trace, color=colors[x], label=response_type, linewidth=3)
#             ax[i].set_title(response_type)
#             ax[i].set_xlim([0, len(trace)])
#             ax[i].set_xticks(xtimes)
#             ax[i].set_xticklabels(xlabels)
#             ax[i].set_xlabel('time after change (sec)', fontsize=16)
#             if response_type == 'HIT' or response_type == 'MISS':
#                 ax[i].axvspan(frames[0], frames[1] / 2, zorder=-1, alpha=0.3,
#                               color=stim_colors[stim_codes[stim_codes != stim_code][0]])
#                 ax[i].axvspan(frames[1] / 2, frames[1], zorder=-1, alpha=0.3, color=stim_colors[stim_code])
#             else:
#                 ax[i].axvspan(frames[0], frames[1], zorder=-1, alpha=0.3, color=stim_colors[stim_code])
#             i += 1
#         ax[0].set_ylabel('dF/F')
#         ax[4].set_ylabel('dF/F')
#         ax[0].text(frames_range[1] / 2 - 20, 3.05, 'go trials', fontsize=22)
#         ax[2].text(frames_range[1] / 2 - 20, 3.05, 'catch trials', fontsize=22)
#         fig.tight_layout()
#     if save:
#         plt.gcf().subplots_adjust(top=.87)
#         fig_title = 'roi_' + str(cell) + '_2'
#         fig_dir = os.path.join(dataset.analysis_dir, 'trial_type_responses_row_with_stim')
#         if not os.path.exists(fig_dir):
#             os.mkdir(fig_dir)
#         saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#         #     plt.close()
#
#
# # In[ ]:
#
#
# def get_run_speed_segment(dataset, timestamps):
#     data = dataset.session.behavior_log_dict
#     times_2p = dataset.sync['2PFrames']['timestamps']
#     times_vis = dataset.sync['visualFrames']['timestamps']
#     ts, rs = ut.get_running_speed(data, time_array=times_vis)
#     run_times_2p = ut.resample(ts, rs, times_2p)
#
#     start_frame = np.where(times_2p == timestamps[0])[0]
#     stop_frame = np.where(times_2p == timestamps[-1])[0]
#     run_speed = run_times_2p[start_frame:stop_frame]
#     return run_speed
#
#
# # In[ ]:
#
# def get_stim_array(dataset):
#     stimlog = dataset.pkl['stimuluslog']
#     stimlog = pd.DataFrame(stimlog)
#     stim_list = []
#     for frame in range(len(stimlog)):
#         if stimlog.iloc[frame].state == True:
#             stim_code = sa.get_code_for_image_name(dataset, stimlog.iloc[frame].image_name)
#             stim_list.append(stim_code)
#         elif stimlog.iloc[frame].state == False:
#             stim_code = -1
#             stim_list.append(stim_code)
#     stim_array = np.asarray(stim_list)
#     return stim_array
#
#
# # In[ ]:
#
# def plot_behavior_events_single_trial(trial_pkl_df, dataset, ax=None):
#     sns.set_context('notebook', font_scale=2, rc={'lines.markeredgewidth': 1})
#     times_vis = dataset.sync['visualFrames']['timestamps']
#     pdf = trial_pkl_df
#     reward_times = times_vis[pdf.reward_frames]
#     lick_times = times_vis[pdf.lick_frames]
#
#     ymin, ymax = ax.get_ylim()
#     lick_y = ymin + (ymax * 0.05)
#     reward_y = ymin + (ymax * 0.1)
#     lick_y_array = np.empty(len(lick_times))
#     lick_y_array[:] = lick_y
#     reward_y_array = np.empty(len(reward_times))
#     reward_y_array[:] = reward_y
#     ax.plot(lick_times, lick_y_array, '|', color='g')
#     ax.plot(reward_times, reward_y_array, 'o', markerfacecolor='purple', markeredgecolor='purple')
#     return ax
#
#
# # In[ ]:
# def add_stimulus_times(pdf, dataset, ax=None):
#     stim_array = get_stim_array(dataset)
#     startframe = pdf.change_frame - (60 * dataset.window[1])
#     endframe = pdf.change_frame + (60 * dataset.window[1])
#     stim_times_array = dataset.sync['visualFrames']['timestamps'][startframe:endframe]
#     stim_on_frames = stim_array[startframe:endframe]
#     change_time = dataset.sync['visualFrames']['timestamps'][pdf.change_frame]
#     colors = get_colors_for_stim_codes(np.sort(dataset.df.change_code.unique()))
#     amin = stim_times_array[0]
#     for i, frame in enumerate(stim_on_frames):
#         if stim_times_array[i] < change_time:
#             color = colors[sa.get_code_for_image_name(dataset, pdf.initial_image)]
#         elif stim_times_array[i] >= change_time:
#             color = colors[sa.get_code_for_image_name(dataset, pdf.change_image)]
#         if (frame == 0) and (stim_on_frames[i - 1] == -1):
#             amin = stim_times_array[i]
#         else:
#             pass
#         if (frame == 0) and (stim_on_frames[i + 1] == -1):
#             amax = stim_times_array[i]
#             pu.addSpan(ax, amin, amax, color=color, alpha=0.7)
#         else:
#             pass
#         if (frame == 1) and (stim_on_frames[i - 1] == -1):
#             amin = stim_times_array[i]
#         else:
#             pass
#         if (frame == 1) and (stim_on_frames[i + 1] == -1):
#             amax = stim_times_array[i]
#             pu.addSpan(ax, amin, amax, color=color, alpha=0.7)
#         else:
#             pass
#     return ax
#
#
# # In[ ]:
# def plot_single_trial_response(cell, trial, response_type, change_code, dataset, ax=None, save=False):
#     fig_folder = 'single_trial_responses'
#     df = dataset.df
#     cdf = df[(df.cell == cell) & (df.response_type == response_type) & (df.change_code == change_code)]
#     responses = cdf.responses.values
#     trace = responses[trial]
#     timestamps = cdf.response_timestamps.values[trial]
#     run_speed = get_run_speed_segment(dataset, timestamps)
#     figsize = (5, 8)
#     fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
#     ax = ax.ravel()
#     ax[0].plot(timestamps, trace, color=get_colors_for_response_types([response_type])[0], linewidth=3)
#     #     ax[0].set_title('roi '+str(cell)+' trial '+str(cdf.trial.values[trial])+' '+response_type)
#     ax[0].set_title('neuronal response')
#     plt.suptitle('cell ' + str(cell) + ' trial ' + str(cdf.trial.values[trial]), x=0.58, y=1,
#                  horizontalalignment='center')
#
#     ax[1].plot(timestamps[:len(run_speed)], run_speed, color='k', alpha=0.5, linewidth=3)
#     pkl_df = dataset.pkl_df
#     pkl_df = pkl_df[pkl_df.change_image != 0]
#     pdf = pkl_df.iloc[cdf.global_trial.values[trial]]
#     ax[1] = plot_behavior_events_single_trial(pdf, dataset, ax=ax[1])
#     ax[1].set_title('behavioral response')
#
#     xlabels = np.arange(timestamps[0], timestamps[-1] + 1, 2)
#     for i in range(2):
#         ax[i].set_xlim([timestamps[0], timestamps[-1]])
#         ax[i].set_xticks(xlabels)
#         ax[i].set_xticklabels(np.arange(dataset.window[0], dataset.window[1] + 1, 2))
#         ax[i] = add_stimulus_times(pdf, dataset, ax=ax[i])
#     ax[0].set_ylabel('dF/F')
#     ax[1].set_ylabel('run speed (cm/s)')
#     ax[1].set_xlabel('time after change (sec)')
#     fig.tight_layout()
#     if save:
#         fig_title = 'roi_' + str(cell) + '_' + response_type + '_trial_' + str(cdf.trial.values[trial]) + '_' + str(
#             change_code)
#         save_figure(fig, figsize, dataset.analysis_dir, fig_title, fig_folder)
#         plt.close()
#
#
# # In[ ]:
#
# def plot_mean_behavior_events(response_type, dataset, window=[-4, 4], save=False):
#     fig_folder = 'behavior_events'
#     df = dataset.df
#     pkl_df = dataset.pkl_df
#     frames_range = [(window[0] - dataset.window[0]) * 30, (dataset.window[1] + window[1]) * 30]
#     n_frames = window[1] - window[0] * 60
#     times = np.arange(0, n_frames, 1)
#     color = get_colors_for_response_types([response_type])
#     cdf = df[(df.cell == 0) & (df.response_type == response_type)]
#     run_speeds = []
#     reward_times = []
#     lick_times = np.empty((1))
#     for trial in range(len(cdf)):
#         timestamps = cdf.response_timestamps.values[trial]
#         timestamps = timestamps[frames_range[0]:frames_range[1]]
#         run_speed = get_run_speed_segment(dataset, timestamps)
#         run_speeds.append(run_speed)
#         pdf = pkl_df.iloc[cdf.global_trial.values[trial]]
#         reward_times.append(pdf.reward_times - pdf.change_time)
#         lick_times = np.hstack((lick_times, (pdf.lick_times - pdf.change_time) + window[1]))
#     run_speeds = np.asarray(run_speeds)
#     mean_run_speed = np.mean(run_speeds, axis=0)
#     sem = np.std(run_speeds) / np.sqrt(len(run_speeds))
#
#     figsize = (4, 7)
#     fig, ax = plt.subplots(2, 1, figsize=figsize)
#     ax = ax.ravel()
#     ax[0].plot(times[:len(mean_run_speed)], mean_run_speed, color=color[0])
#     ax[0].fill_between(np.arange(0, len(mean_run_speed), 1), mean_run_speed + sem, mean_run_speed - sem, color=color,
#                        alpha=0.4)
#     ax[0].set_xlim([0, n_frames])
#     ax[0].set_xticks(np.arange(0, n_frames + 30, 60))
#     ax[0].set_xticklabels(np.arange(window[0], window[1] + 1, 2))
#     ax[0].set_title('mean run speed')
#     ax[0].set_ylabel('run speed (cm/s)')
#     ax[0].set_xlabel('time after change (sec)')
#
#     ax[1].hist(lick_times[1:], bins=10, color=color[0])
#     ax[1].set_xlim([0, window[1] - window[0]]);
#     ax[1].set_xticks(np.arange(0, window[1] - window[0] + 1, 2));
#     ax[1].set_xticklabels(np.arange(window[0], window[1] + 1, 2));
#     ax[1].set_ylabel('licks')
#     ax[1].set_xlabel('time after change (sec)')
#     ax[1].set_title('response latency')
#     plt.tight_layout()
#     if save:
#         #         fig.tight_layout()
#         fig_title = 'mean_behavior_events_' + response_type + '_' + str(window[1]) + 's'
#         save_figure(fig, figsize, dataset.analysis_dir, fig_title, fig_folder)
#         plt.close()
#
#
# # In[ ]:
#
# def plot_mean_run_speed(dataset, window=[-6, 6]):
#     response_types = ['HIT', 'FA', 'MISS', 'CR']
#     df = dataset.df
#     figsize = (7, 7)
#     fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
#     ax = ax.ravel()
#     colors = get_colors_for_response_types(response_types)
#     # def plot_mean_run_speed()
#     frames_range = [(window[0] - dataset.window[0]) * 30, (dataset.window[1] + window[1]) * 30]
#     n_frames = window[1] - window[0] * 60
#     for i, response_type in enumerate(response_types):
#         cdf = df[(df.cell == 0) & (df.response_type == response_type)]
#         run_speeds = []
#         for trial in range(len(cdf)):
#             timestamps = cdf.response_timestamps.values[trial]
#             timestamps = timestamps[frames_range[0]:frames_range[1]]
#             run_speed = get_run_speed_segment(dataset, timestamps)
#             run_speeds.append(run_speed)
#         run_speeds = np.asarray(run_speeds)
#         mean_run_speed = np.mean(run_speeds, axis=0)
#         sem = np.std(run_speeds) / np.sqrt(len(run_speeds))
#         ax[i].plot(mean_run_speed, color=colors[i])
#         ax[i].fill_between(np.arange(0, len(mean_run_speed), 1), mean_run_speed + sem, mean_run_speed - sem,
#                            color=colors[i], alpha=0.4)
#         ax[i].set_xlim([0, n_frames])
#         ax[i].set_xticks(np.arange(0, n_frames + 30, 60))
#         ax[i].set_xticklabels(np.arange(window[0], window[1] + 1, 2))
#         #     ax[i].set_xlim([0,len(mean_run_speed)])
#         #     ax[i].set_xticks(np.arange(0,len(mean_run_speed)+30,60))
#         #     ax[i].set_xticklabels(np.arange(0,len(mean_run_speed),60)/30+dataset.window[0]+1)
#         ax[i].set_title(response_type)
#     ax[0].set_ylabel('run speed (cm/s)')
#     ax[2].set_xlabel('time after change (sec)')
#     ax[2].set_ylabel('run speed (cm/s)')
#     ax[3].set_xlabel('time after change (sec)')
#     plt.tight_layout()
#     save_figure(fig, figsize, dataset.analysis_dir, fig_title='avg_run_speed_' + str(window[1]) + 's',
#                 folder='behavior_events')
#
#
# # In[ ]:
#
#
# def plot_auroc(dataset, cell, change_code, response_types, window=[-2, 2], ax=None, save=False):
#     df = dataset.df
#     colors = get_colors_for_response_types(response_types)
#     stim_codes = np.sort(dataset.stim_codes.stim_code.unique())
#     stim_colors = get_colors_for_stim_codes(stim_codes)
#
#     responses = []
#     for response in df[(df.cell == cell) & (df.response_type == response_types[0]) & (
#         df.change_code == change_code)].responses.values:
#         responses.append(response)
#     responses = np.asarray(responses)
#     mean_response = np.mean(responses, axis=0)
#     sem_mr = np.std(responses) / np.sqrt(len(responses))
#
#     nonresponses = []
#     for response in df[(df.cell == cell) & (df.response_type == response_types[1]) & (
#         df.change_code == change_code)].responses.values:
#         nonresponses.append(response)
#     nonresponses = np.asarray(nonresponses)
#     mean_non_response = np.mean(nonresponses, axis=0)
#     sem_mnr = np.std(nonresponses) / np.sqrt(len(nonresponses))
#
#     auroc = []
#     for i in range(np.shape(responses)[1]):
#         fp, tp = ut.roc(nonresponses[:, i], responses[:, i])
#         auroc.append(np.trapz(x=np.sort(fp), y=np.sort(tp)))
#     if ax is None:
#         figsize = (4, 6)
#         fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
#     times = np.linspace(-6, 6, 360)
#     ax[0].plot(times, mean_response, color=colors[0], label=response_types[0])
#     ax[0].fill_between(times, mean_response + sem_mr, mean_response - sem_mr, color=colors[0], alpha=0.5)
#     ax[0].plot(times, mean_non_response, color=colors[1], label=response_types[1])
#     ax[0].fill_between(times, mean_non_response + sem_mnr, mean_non_response - sem_mnr, color=colors[1], alpha=0.5)
#     ax[0].set_xlim(window)
#     #    ax[0].set_title(response_types[0]+' / '+response_types[1])
#     ax[0].set_title('cell ' + str(cell))
#     # ax[0].set_ylim(0,3)
#     if ax is None:
#         ax[0].set_title('roi_' + str(cell))
#         ax[0].legend(loc=9, ncol=1, bbox_to_anchor=(1.25, 1.0));
#
#     ax[1].plot(times, auroc, color='gray', linewidth=3)
#     ax[1].set_ylim(0, 1)
#     # ax[1].set_title('AUROC')
#     ax[1].set_ylabel('AUROC')
#     ax[1].set_xlim(window)
#     ax[1].set_xticks(np.arange(window[0], window[1] + 1, 1))
#     #    ax[1].set_xticklabels(np.arange(window[0],window[1],1))
#     fig.tight_layout()
#
#     for i in [0, 1]:
#         if ('HIT' in response_types):
#             ax[i].axvspan(window[0], window[1] + window[0], zorder=-1, alpha=0.3,
#                           color=stim_colors[stim_codes[stim_codes != change_code][0]])
#             ax[i].axvspan(window[1] + window[0], window[1], zorder=-1, alpha=0.3, color=stim_colors[change_code])
#         else:
#             ax[i].axvspan(window[0], window[1], zorder=-1, alpha=0.3, color=stim_colors[change_code])
#
#     if save:
#         plt.gcf().subplots_adjust(right=.73)
#         fig_title = 'roi_' + str(cell) + '_' + response_types[0] + '_' + response_types[1] + '_' + str(change_code)
#         fig_dir = os.path.join(dataset.analysis_dir, 'auroc')
#         if not os.path.exists(fig_dir):
#             os.mkdir(fig_dir)
#         saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#         plt.close()
#     return ax
#
#
# # In[ ]:
#
# def plot_auroc2(dataset, cell, ax=None, save=False):
#     df = dataset.df
#     window = [-2, 2]
#     response_types = [['HIT', 'MISS'], ['FA', 'CR']]
#     stim_codes = np.sort(dataset.stim_codes.stim_code.unique())
#     stim_colors = get_colors_for_stim_codes(stim_codes)
#
#     figsize = (6, 6)
#     if ax is None:
#         fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
#         ax = ax.ravel();
#     i = 0
#     for y, stim_code in enumerate(stim_codes):
#         if y == 1:
#             i += 1
#         for x, response_type in enumerate(response_types):
#             auroc = get_auroc(df, cell, stim_code, response_type)
#             if x == 1:
#                 i = i + 1
#             times = np.linspace(-6, 6, 360)
#             ax[i].plot(times, auroc, color='gray', linewidth=3)
#             ax[i].set_ylim(0, 1)
#             ax[i].set_xlim(window[0], window[1])
#             ax[i].set_xticks(np.arange(window[0], window[1] + 1, 1))
#             ax[0].set_ylabel('AUROC')
#             ax[2].set_ylabel('AUROC')
#             ax[2].set_xlabel('time after change (sec)', fontsize=16)
#             ax[3].set_xlabel('time after change (sec)', fontsize=16)
#             ax[i].set_title(response_type[0] + ' / ' + response_type[1])
#
#             if ('HIT' in response_type):
#                 ax[i].axvspan(window[0], window[1] + window[0], zorder=-1, alpha=0.3,
#                               color=stim_colors[stim_codes[stim_codes != stim_code][0]])
#                 ax[i].axvspan(window[1] + window[0], window[1], zorder=-1, alpha=0.3, color=stim_colors[stim_code])
#             else:
#                 ax[i].axvspan(window[0], window[1], zorder=-1, alpha=0.3, color=stim_colors[stim_code])
#     plt.suptitle('cell ' + str(cell), x=0.52, y=1, horizontalalignment='center')
#     if save:
#         fig.tight_layout()
#         #         plt.gcf().subplots_adjust(right=.73)
#         fig_title = 'roi_' + str(cell)
#         fig_dir = os.path.join(dataset.analysis_dir, 'auroc2')
#         if not os.path.exists(fig_dir):
#             os.mkdir(fig_dir)
#         saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'], size=figsize)
#         plt.close()
#     return ax
#
#
# # In[ ]:
#
# def plot_auroc_alone(dataset, cell, change_code, response_types, ax=None, label=True):
#     window = [-2, 2]
#     df = dataset.df
#     stim_codes = dataset.stim_codes.stim_code.unique()
#     stim_colors = get_colors_for_stim_codes(stim_codes)
#     responses = []
#     for response in df[(df.cell == cell) & (df.response_type == response_types[0]) & (
#         df.change_code == change_code)].responses.values:
#         responses.append(response)
#     responses = np.asarray(responses)
#
#     nonresponses = []
#     for response in df[(df.cell == cell) & (df.response_type == response_types[1]) & (
#         df.change_code == change_code)].responses.values:
#         nonresponses.append(response)
#     nonresponses = np.asarray(nonresponses)
#
#     auroc = []
#     for i in range(np.shape(responses)[1]):
#         fp, tp = ut.roc(nonresponses[:, i], responses[:, i])
#         auroc.append(np.trapz(x=np.sort(fp), y=np.sort(tp)))
#
#     times = np.linspace(-6, 6, 360)
#     ax.plot(times, auroc, color='gray', linewidth=3)
#     ax.set_ylim(0, 1)
#     # ax[1].set_title('AUROC')
#     if label:
#         ax.set_ylabel('AUROC')
#     ax.set_xlim(window[0], window[1])
#     ax.set_xticks(np.arange(window[0], window[1], 1))
#     ax.set_xlabel('time after change (sec)', fontsize=16)
#     if 'HIT' in response_types:
#         ax.axvspan(window[0], window[1] + window[0], zorder=-1, alpha=0.3,
#                    color=stim_colors[stim_codes[stim_codes != change_code][0]])
#         ax.axvspan(window[1] + window[0], window[1], zorder=-1, alpha=0.3, color=stim_colors[change_code])
#     else:
#         ax.axvspan(window[0], window[1], zorder=-1, alpha=0.3, color=stim_colors[change_code])
#
#     return ax
#
#
# # In[ ]:
#
# def reformat_axes(ax, num=4):
#     if num == 4:
#         a = [0, 1, 2, 3]
#         a[0] = ax[0][0]
#         a[1] = ax[0][1]
#         a[2] = ax[1][0]
#         a[3] = ax[1][1]
#     if num == 6:
#         a = [0, 1, 2, 3, 4, 5]
#         a[0] = ax[0][0]
#         a[1] = ax[0][1]
#         a[2] = ax[1][0]
#         a[3] = ax[1][1]
#         a[4] = ax[2][0]
#         a[5] = ax[2][1]
#     if num == 8:
#         a = [0, 1, 2, 3, 4, 5, 6, 7]
#         a[0] = ax[0][0]
#         a[1] = ax[0][1]
#         a[2] = ax[1][0]
#         a[3] = ax[1][1]
#         a[4] = ax[2][0]
#         a[5] = ax[2][1]
#         a[6] = ax[3][0]
#         a[7] = ax[3][1]
#
#     return a
#
#
# # In[ ]:
#
# def get_auroc(df, cell, change_code, response_types):
#     responses = []
#     for response in df[(df.cell == cell) & (df.response_type == response_types[0]) & (
#         df.change_code == change_code)].responses.values:
#         responses.append(response)
#     responses = np.asarray(responses)
#
#     nonresponses = []
#     for response in df[(df.cell == cell) & (df.response_type == response_types[1]) & (
#         df.change_code == change_code)].responses.values:
#         nonresponses.append(response)
#     nonresponses = np.asarray(nonresponses)
#
#     auroc = []
#     for i in range(np.shape(responses)[1]):
#         fp, tp = ut.roc(nonresponses[:, i], responses[:, i])
#         auroc.append(np.trapz(x=np.sort(fp), y=np.sort(tp)))
#     auroc = np.asarray(auroc)
#     return auroc
#
#
# # In[ ]:
#
# def get_auroc_values(sdf, df, window, response_types):
#     mean_auroc_list = []
#     max_auroc_list = []
#     for cell in df.cell.unique():
#         change_code = sdf[sdf.cell == cell].pref_stim.values[0]
#         auroc = get_auroc(df, cell, change_code, response_types)
#         mean_auroc = np.mean(auroc[window[0] * 30:window[1] * 30])
#         mean_auroc_list.append(mean_auroc)
#         max_auroc = np.amax(auroc[window[0] * 30:window[1] * 30])
#         max_auroc_list.append(max_auroc)
#     return mean_auroc_list, max_auroc_list
#
#
# # In[ ]:
#
# def plot_metrics_mask(dataset, metrics, cell_list, metric_name, max_image=True, cmap='RdBu', ax=None, save=False):
#     roi_dict = dataset.roi_dict.copy()
#     if ax is None:
#         fig, ax = plt.subplots()
#     if max_image is True:
#         ax.imshow(dataset.max_image, cmap='gray', vmin=0, vmax=np.amax(dataset.max_image) / 2)
#     for roi in cell_list:
#         tmp = roi_dict[roi].copy()
#         mask = np.empty(tmp.shape, dtype=np.float)
#         mask[:] = np.nan
#         mask[tmp == 1] = metrics[roi]
#         cax = ax.imshow(mask, cmap=cmap, alpha=0.5, vmin=np.amin(metrics), vmax=np.amax(metrics))
#         #        cax = ax.imshow(mask,cmap=cmap,alpha=0.6,vmin=-1,vmax=1)
#         ax.set_title(metric_name)
#         ax.grid(False)
#         ax.axis('off')
#     plt.colorbar(cax, ax=ax)
#     if save:
#         plt.tight_layout()
#         fig_folder = 'metric_masks'
#         fig_title = metric_name
#         fig_dir = os.path.join(dataset.analysis_dir, fig_folder)
#         if not os.path.exists(fig_dir): os.mkdir(fig_dir)
#         saveFigure(fig, os.path.join(fig_dir, fig_title), formats=['.png'])
#
#
# # In[ ]:




def plot_mdf_metric_dist(mdf, metric, expt_id, sig_cells=False, ax=None, analysis_dir=None, show=True):
    if sig_cells:
        tmp = mdf[mdf.sig_thresh == True]
        title = metric + '_sig_cells'
    else:
        tmp = mdf.copy()
        title = metric
    if ax is None:
        figsize = (10, 4)
        fig, ax = plt.subplots(1, 4, figsize=figsize, sharey=True)
        ax = ax.ravel();
    for i, response_type in enumerate(mdf.response_type.unique()):
        metric_dist = tmp[(tmp.pref_stim == True) & (tmp.response_type == response_type)][metric].values
        sns.violinplot(metric_dist, orient='v', ax=ax[i], color="0.8")
        sns.stripplot(metric_dist, orient='v', ax=ax[i], jitter=0.08)
        ax[i].set_ylabel('')
        ax[0].set_ylabel(metric)
        ax[i].set_xlabel(response_type)
        plt.suptitle(expt_id, y=1.)
    if analysis_dir:
        save_figure(fig, figsize, analysis_dir, title, 'response_variability')
    if not show:
        plt.close()

        # In[ ]:


def plot_mean_response_heatmap_ns(dataset, mdf, sorted_cell_ids, cmap='magma', save=False, ax=None):
    response_types = ['HIT', 'MISS', 'FA', 'CR']
    df = dataset.df
    # if ax is None:
    figsize = (15, 6)
    fig, ax = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)
    ax = ax.ravel()
    i = 0
    # for x,change_code in enumerate(np.sort(df.change_code.unique())):
    for i, response_type in enumerate(response_types):
        n_frames = df.responses.values[0].shape[0]
        n_cells = len(sorted_cell_ids)
        response_matrix = np.empty((n_cells, n_frames))
        for c, cell in enumerate(sorted_cell_ids):
            tmp = mdf[(mdf.cell == cell) & (mdf.pref_stim == True) & (mdf.response_type == response_type)]
            if len(tmp) > 0:
                response = tmp.mean_trace.values[0]
                response_matrix[c, :] = response
            im = ax[i].pcolormesh(response_matrix, cmap=cmap, vmin=0, vmax=np.percentile(response_matrix, 98))
            #            im = ax[i].pcolormesh(response_matrix,cmap=cmaps.plasma,vmax=0.5)
            ax[i].set_ylim(0, response_matrix.shape[0])
            ax[i].set_xlim(0, response_matrix.shape[1])
            ax[i].set_xticks(np.arange(0, (dataset.window[1] * 30 - (dataset.window[0] * 30) + 30), 60));
            ax[i].set_xticklabels(np.arange(dataset.window[0], dataset.window[1] + 1, 2));
            ax[i].set_title(response_type)
            ax[i].set_xlabel('time (s)')
    # i+=1
    ax[0].set_ylabel('cells')
    # ax[4].set_ylabel('cells')
    cax = fig.add_axes([1.01, 0.2, 0.03, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('mean dF/F')
    plt.tight_layout()
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, fig_title='all_cells_mean_response_by_trial_type',
                    folder='transition_type_heatmap')
        plt.close()
    return ax


# In[ ]:

def plot_single_cell_variability(df, cell, ax=None, analysis_dir=None):
    figsize = (6, 6)
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
        ax = ax.ravel();
    for i, trial_type in enumerate(df.trial_type.unique()):
        means = df[(df.cell == cell) & (df.pref_stim == True) & (df.trial_type == trial_type)].mean_response
        CV = np.std(means) / np.mean(means)
        sns.violinplot(means, orient='v', ax=ax[i], color='0.8', linewidth=3)
        sns.stripplot(means, orient='v', ax=ax[i], jitter=0.08, alpha=0.7)
        #     sns.pointplot(means,orient='v',ax=ax[i],color='0.2',join=False,capsize=.2)
        #     ax[i].errorbar(0,np.mean(means),yerr=np.std(means),elinewidth=4,capsize=None,ecolor='green')
        ax[i].set_ylabel('')
        ax[0].set_ylabel('mean response')
        ax[i].set_title('CV: ' + str(np.round(CV, 2)))
        ax[i].set_xlabel(trial_type)
    plt.suptitle('cell ' + str(cell), y=1.05)
    if analysis_dir is not None:
        save_figure(fig, figsize, analysis_dir, 'roi_' + str(cell), 'response_variability')
        plt.close()
    return ax


# In[ ]:

def plot_response_matrix(mdf, values, index, columns, cell_label='cell', remove_cells=None, fig_length=5,
                         sig=True, ax=None, analysis_dir=None, show=True, yticks=True):
    if sig:
        title = values + '_' + columns[0] + '_' + index[0] + '_sig'
        sig_cells = mdf[mdf.sig_thresh == True][cell_label].unique()
        tmp = mdf[mdf[cell_label].isin(sig_cells)]
    else:
        title = values + '_' + columns[0] + '_' + index[0]
        tmp = mdf
    if remove_cells:
        tmp = tmp[tmp[cell_label].isin(remove_cells) == False]
    if ax is None:
        figsize = (5, fig_length)
        fig, ax = plt.subplots(figsize=figsize)
    response_matrix = pd.pivot_table(tmp, values, index, columns)
    sns.heatmap(response_matrix, ax=ax, vmin=0, vmax=0.5, cmap='magma', cbar_kws={'label': 'mean dF/F'})
    ax.set_title('mean response \nby image', va='bottom', ha='center')
    ax.set_xlabel(columns[0])
    ax.set_ylabel('cells')
    if yticks == False:
        ax.set_yticklabels('')
    elif yticks == 'reorder':
        spacing = 10
        ax.set_yticks(np.arange(0, len(tmp[cell_label].unique()), spacing))
        ax.set_yticklabels(np.arange(0, len(tmp[cell_label].unique()), spacing))
    # ax.set_yticklabels(np.arange(0,len(tmp[cell_label].unique()),spacing)[::-1])
    if analysis_dir:
        fig.tight_layout()
        #        plt.gcf().subplots_adjust(right=.82)
        save_figure(fig, figsize, analysis_dir, title, 'response_matrix', formats=['.png'])
    if not show:
        plt.close()
    return ax


# In[ ]:

def plot_adaptation(mdf, cell, mean_window, ratio='r2_r1', analysis_dir=None):
    sns.set_context('poster', font_scale=1.5, rc={'lines.markeredgewidth': 2})
    frame_int = 0.7 * 30  # flashes happen every 700ms
    stim_start_frame = (mean_window[0] * 30)  # change time
    x_start_frame = (mean_window[0] * 30) - frame_int * 3  # start x axis 3 flashes before change time
    num_flashes = 6
    last_flash = stim_start_frame + (frame_int * num_flashes)
    flash_onsets = np.arange(stim_start_frame, last_flash, frame_int)

    figsize = (20, 5)
    fig, ax = plt.subplots(1, 4, figsize=figsize, sharey=True)
    ax = ax.ravel()
    for i, response_type in enumerate(['HIT', 'MISS', 'CR', 'FA']):
        mask = ((mdf.cell == cell) & (mdf.pref_stim == True) & (mdf.response_type == response_type))
        if len(mdf[mask]) > 0:
            response = mdf[mask].mean_trace.values[0]
            sem = mdf[mask].sem_trace.values[0]
            flash_means = mdf[mask].flash_means.values[0]
            r = np.round(mdf[mask][ratio].values[0], 2)
            pref_stim = mdf[mask].stim_code.values[0]
            color = get_color_for_stim(pref_stim, mdf.stim_code.unique())
            ax[i].plot(response)
            ax[i].fill_between(np.arange(0, len(response), 1), response + sem, response - sem, alpha=0.5)
            for x in range(len(flash_onsets)):
                ax[i].plot(flash_onsets[x] + 0.25 * 30, flash_means[x], '.', color='gray')
                pu.addSpan(ax[i], flash_onsets[x], flash_onsets[x] + 0.5 * 30, color=color)
            # xmin = flash_onsets[0]-((0.7*30)*3) #first flash minus num frames in 3 flashes
            #            xmax = flash_onsets[-1]+(0.7*30)
            #            xmin=117 #above equation doesnt work somehow
            times = np.arange(0, (len(response) / 30) + 1, 2) - mean_window[0]
            ax[i].set_xticks(np.arange(0, len(response) + 1, 60));
            ax[i].set_xticklabels(times);
            ax[i].set_xlim(x_start_frame, len(response) + 1)
            ax[0].set_ylabel('mean dF/F')
            ax[i].set_title(response_type + ' - ratio: ' + str(r))
            ax[i].set_xlabel('time after change (s)')
        plt.suptitle('cell ' + str(cell), y=0.98, x=0.52)
    if analysis_dir is not None:
        fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.80)
        save_figure(fig, figsize, analysis_dir, 'roi_' + str(cell), 'response_ratio_' + ratio)
        plt.close()


# In[ ]:
#
# def plot_mdf_metric_dist(mdf,metric,expt_id,sig_cells=False,ax=None,analysis_dir=None,show=True):
#    if sig_cells:
#        tmp = mdf[mdf.sig_thresh==True]
#        title = metric+'_sig_cells'
#    else:
#        tmp = mdf
#        title = metric
#    if ax is None:
#        figsize=(11,4)
#        fig,ax = plt.subplots(1,4,figsize=figsize,sharey=True)
#        ax = ax.ravel();
#    for i,response_type in enumerate(['HIT','MISS','CR','FA']):
#        metric_dist = tmp[(tmp.pref_stim==True)&(tmp.response_type==response_type)][metric].values
##        sns.violinplot(metric_dist,orient='v',ax=ax[i],color="0.8",linewidth=1.5)
#        sns.boxplot(data=metric_dist,orient='v',ax=ax[i],color="0.8",width=0.4)
#        sns.stripplot(metric_dist,orient='v',ax=ax[i])#,jitter=0.05)
#        ax[i].set_ylim([-0.2,2.2])
#        ax[i].set_ylabel('')
#        ax[0].set_ylabel(metric)
#        ax[i].set_xlabel(response_type)
#        plt.suptitle(expt_id,y=1.)
#    if analysis_dir:
#        save_figure(fig,figsize,analysis_dir,title,'response_variability')
#    if not show:
#        plt.close()
#
#
## In[ ]:
#
# def plot_mean_response_heatmap_ns(dataset,mdf,sorted_cell_ids,cmap=cmaps.magma,save=False,ax=None):
#    response_types = ['HIT','MISS','FA','CR']
#    df = dataset.df
#    # if ax is None:
#    figsize = (15,6)
#    fig,ax=plt.subplots(1,4,figsize=figsize,sharex=True,sharey=True)
#    ax = ax.ravel()
#    i=0
#    # for x,change_code in enumerate(np.sort(df.change_code.unique())):
#    for i,response_type in enumerate(response_types):
#        n_frames = df.responses.values[0].shape[0]
#        n_cells = len(sorted_cell_ids)
#        response_matrix = np.empty((n_cells,n_frames))
#        for c,cell in enumerate(sorted_cell_ids):
#            tmp = mdf[(mdf.cell==cell)&(mdf.pref_stim==True)&(mdf.response_type==response_type)]
#            if len(tmp) > 0:
#                response = tmp.mean_trace.values[0]
#                response_matrix[c,:] = response
#            im = ax[i].pcolormesh(response_matrix,cmap=cmap,vmin=0,vmax=np.percentile(response_matrix,98))
#        #            im = ax[i].pcolormesh(response_matrix,cmap=cmaps.plasma,vmax=0.5)
#            ax[i].set_ylim(0,response_matrix.shape[0])
#            ax[i].set_xlim(0,response_matrix.shape[1])
#            ax[i].set_xticks(np.arange(0,(dataset.window[1]*30-(dataset.window[0]*30)+30),60));
#            ax[i].set_xticklabels(np.arange(dataset.window[0],dataset.window[1]+1,2));
#            ax[i].set_title(response_type)
#            ax[i].set_xlabel('time (s)')
#    #         i+=1
#    ax[0].set_ylabel('cells')
#    # ax[4].set_ylabel('cells')
#    cax = fig.add_axes([0.92, 0.12, 0.03, 0.78])
#    cb = fig.colorbar(im, cax=cax)
#    cb.set_label('mean dF/F')
#    plt.tight_layout()
#    if save:
#        save_figure(fig,figsize,dataset.analysis_dir,fig_title='all_cells_mean_response_by_trial_type',folder='transition_type_heatmap')
#        plt.close()
#    return ax


def get_upper_limit_and_intervals(dataset):
    traces = dataset.dff_traces
    upper = np.round(traces[0].shape[0], -3) + 1000
    times = dataset.sync['2PFrames']['timestamps']
    interval = 5 * 60
    frame_interval = np.arange(0, len(traces[0, :]), interval * 30)
    time_interval = np.uint64(np.round(np.arange(times[0], times[-1], interval), 1))
    return upper, time_interval, frame_interval


def plot_traces_heatmap(dataset, save=False, cbar=True, ax=None):
    sns.set_context('notebook', font_scale=2, rc={'lines.markeredgewidth': 1})
    traces = dataset.dff_traces
    upper_limit, time_interval, frame_interval = get_upper_limit_and_intervals(dataset)
    if ax is None:
        figsize = (20, 8)
        fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(traces, cmap='magma', vmin=0, vmax=np.percentile(traces[np.isnan(traces)==False], 99))
    # cax = ax.pcolormesh(traces, cmap=cmap, vmin=np.percentile(traces, 5), vmax=np.percentile(traces, 95))
    ax.set_ylim(0, traces.shape[0])
    #    ax.set_xlim(0,traces.shape[1])
    ax.set_ylabel('cells')
    ax.set_xticks(np.arange(0, upper_limit, 600 * 30))
    ax.set_xticklabels(np.arange(0, upper_limit / 30, 600))
    #    ax.set_xticks(np.arange(600*30,upper_limit,600*30))
    #    ax.set_xticklabels(np.arange(600,upper_limit/30,600))
    #    ax.set_xticks(frame_interval)
    #    ax.set_xticklabels(time_interval)
    #    ax.set_xbound(lower=0,upper=upper_limit)
    if cbar:
        cb = plt.colorbar(cax);
        cb.set_label('dF/F', labelpad=5)
        c = '1'
    else:
        ax.set_xticklabels('')
        c = '0'
    # fig.tight_layout()
    if save:
        title = 'dff_traces_heatmap_' + c
        save_figure(fig, figsize, dataset.analysis_dir, title, 'traces', formats=['.png'])
    return ax


# # In[ ]:
#
# def plot_run_reward_rate(dataset, save=False):
#     times = dataset.sync['visualFrames']['timestamps']
#     frame_times = []
#     reward_rate = []
#     for i, frame in enumerate(dataset.pkl_df.change_frame.values):
#         if frame != 0:
#             frame_times.append(times[frame])
#             reward_rate.append(dataset.pkl_df.iloc[i].reward_rate)
#     upper, time_interval, frame_interval = get_upper_limit_and_intervals(dataset)
#
#     colors = sns.color_palette()
#     figsize = (20, 4)
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.plot(times, dataset.run_speed, color=colors[1], label='run speed')
#     ax.set_ylabel('run speed (cm/s)')
#     ax2 = ax.twinx()
#     ax2.plot(frame_times, reward_rate, color=colors[0], linewidth=4, label='reward rate')
#     ax2.grid(False)
#     ax2.set_ylabel('reward rate')
#     #
#     #    ax.set_xticks(np.arange(600*30,upper_limit,600*30))
#     #    ax.set_xticklabels(np.arange(600,upper_limit/30,600))
#     ax2.set_xlim(time_interval[0], np.uint64(upper / 30.))
#     ax2.set_xlabel('time (s)')
#     ax.set_xlabel('time (s)')
#     #     fig.tight_layout()
#     plt.gcf().subplots_adjust(bottom=0.2)
#     if save:
#         save_figure(fig, figsize, dataset.analysis_dir, 'run_reward', 'behavior', formats=['.png'])
#
#
# # In[ ]:
#
# def plot_response_types_CV(df, mdf, cell, mean_window, analysis_dir=None):
#     sns.set_context('poster', font_scale=1.5, rc={'lines.markeredgewidth': 2})
#     frame_int = 0.7 * 30  # flashes happen every 700ms
#     stim_start_frame = (mean_window[0] * 30)  # change time
#     x_start_frame = (mean_window[0] * 30) - frame_int * 2  # start x axis 3 flashes before change time
#     x_stop_frame = (mean_window[0] * 30) + frame_int * 2  # start x axis 3 flashes before change time
#     num_flashes = 6
#     last_flash = stim_start_frame + (frame_int * num_flashes)
#     flash_onsets = np.arange(stim_start_frame, last_flash, frame_int)
#
#     figsize = (20, 5)
#     fig, ax = plt.subplots(1, 4, figsize=figsize, sharey=True)
#     ax = ax.ravel()
#     for i, response_type in enumerate(['HIT', 'MISS', 'CR', 'FA']):
#         masked_df = df[(df.cell == cell) & (df.pref_stim == True) & (df.response_type == response_type)]
#         masked_mdf = mdf[(mdf.cell == cell) & (mdf.pref_stim == True) & (mdf.response_type == response_type)]
#         if len(masked_df) > 0:
#             responses = masked_df.responses.values
#             len_response = responses[0].shape[0]
#             CV = np.round(masked_mdf.CV.values[0], 2)
#             pref_stim = masked_mdf.stim_code.values[0]
#             color = get_color_for_stim(pref_stim, mdf.stim_code.unique())
#             for response in responses:
#                 ax[i].plot(response, color='gray', linewidth=1)
#             ax[i].plot(np.mean(responses, axis=0), color='k', linewidth=2)
#             for x in range(len(flash_onsets)):
#                 pu.addSpan(ax[i], flash_onsets[x], flash_onsets[x] + 0.5 * 30, color=color)
#             times = np.arange(0, (len_response / 30) + 1, 2) - mean_window[0]
#             ax[i].set_xticks(np.arange(0, len_response + 1, 60));
#             ax[i].set_xticklabels(times);
#             ax[i].set_xlim(x_start_frame, x_stop_frame + 1)
#             ax[0].set_ylabel('mean dF/F')
#             ax[i].set_title(response_type + ' - CV: ' + str(CV))
#             ax[i].set_xlabel('time after change (s)')
#         plt.suptitle('cell ' + str(cell), y=0.98, x=0.52)
#     if analysis_dir is not None:
#         fig.tight_layout()
#         plt.gcf().subplots_adjust(top=0.80)
#         save_figure(fig, figsize, analysis_dir, 'roi_' + str(cell), 'response_types_CV')
#         plt.close()
#
#
# # In[ ]:
#
# def plot_trial_types_CV(df, cell, mean_window, analysis_dir=None):
#     sns.set_context('poster', font_scale=1.5, rc={'lines.markeredgewidth': 2})
#     frame_int = 0.7 * 30  # flashes happen every 700ms
#     stim_start_frame = (mean_window[0] * 30)  # change time
#     x_start_frame = (mean_window[0] * 30) - frame_int * 2  # start x axis 3 flashes before change time
#     x_stop_frame = (mean_window[0] * 30) + frame_int * 2  # start x axis 3 flashes before change time
#     num_flashes = 6
#     last_flash = stim_start_frame + (frame_int * num_flashes)
#     flash_onsets = np.arange(stim_start_frame, last_flash, frame_int)
#
#     figsize = (10, 5)
#     fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
#     ax = ax.ravel()
#     for i, trial_type in enumerate(['go', 'catch']):
#         masked_df = df[(df.cell == cell) & (df.pref_stim == True) & (df.trial_type == trial_type)]
#         if len(masked_df) > 0:
#             responses = masked_df.responses.values
#             means = masked_df.mean_response.values
#             len_response = responses[0].shape[0]
#             CV = np.round(np.std(means) / np.mean(means), 2)
#             pref_stim = masked_df.change_code.values[0]
#             color = get_color_for_stim(pref_stim, df.change_code.unique())
#             for response in responses:
#                 ax[i].plot(response, color='gray', linewidth=1)
#             ax[i].plot(np.mean(responses, axis=0), color='k', linewidth=2)
#             for x in range(len(flash_onsets)):
#                 pu.addSpan(ax[i], flash_onsets[x], flash_onsets[x] + 0.5 * 30, color=color)
#             times = np.arange(0, (len_response / 30) + 1, 1) - mean_window[0]
#             ax[i].set_xticks(np.arange(0, len_response + 1, 30));
#             ax[i].set_xticklabels(times);
#             ax[i].set_xlim(x_start_frame, x_stop_frame + 1)
#             ax[0].set_ylabel('mean dF/F')
#             ax[i].set_title(trial_type + ' - CV: ' + str(CV))
#             ax[i].set_xlabel('time after change (s)')
#         plt.suptitle('cell ' + str(cell), y=0.98, x=0.52)
#     if analysis_dir is not None:
#         fig.tight_layout()
#         plt.gcf().subplots_adjust(top=0.80)
#         save_figure(fig, figsize, analysis_dir, 'roi_' + str(cell), 'trial_type_CV')
#         plt.close()
#
#
# def get_pref_stim(sdf, cell):
#     pref_stim = sdf[sdf.cell == cell].pref_stim.values[0]
#     return pref_stim
#
#
# def get_stim_mean_for_trials(df, cell, stim, trials):
#     mean_responses = df[(df.cell == cell) & (df.change_code == stim) & (df.trial.isin(trials))].mean_response
#     mean = mean_responses.mean()
#     sem = mean_responses.std() / np.sqrt(len(mean_responses))
#     return mean, sem
#

def plot_cell_zoom(dataset, cell, spacex=10, spacey=10, show_mask=False, save=False, ax=None):
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    mask = dataset.roi_dict[cell]
    (y, x) = np.where(mask == 1)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    inds = np.where(mask == 0)
    mask[inds] = np.nan
    ax.imshow(dataset.max_image, cmap='gray', vmin=0, vmax=np.amax(dataset.max_image) / 3)
    if show_mask:
        ax.imshow(mask, cmap='jet', alpha=0.3, vmin=0, vmax=1)
    ax.set_xlim(xmin - spacex, xmax + spacex)
    ax.set_ylim(ymin - spacey, ymax + spacey)
    ax.set_title('cell ' + str(cell))
    ax.grid(False)
    ax.axis('off')
    if save:
        if show_mask:
            folder = 'roi_masks_zoom'
        else:
            folder = 'roi_zoom'
        save_figure(fig, figsize, dataset.analysis_dir, 'roi_' + str(cell), folder, formats=['.png'])
        plt.close()


def plot_summary_figure_image(dataset, mdf, sdf, rdf, cell, save=False):
    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.2, .7), yspan=(0, .2))
    ax = plot_behavior_events_trace(dataset, [cell], xmin=360, length=3, ax=ax, save=False, ylabel='dF/F')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .20), yspan=(0, .22))
    ax = plot_cell_zoom(dataset, cell, spacex=25, spacey=25, show_mask=False, save=False, ax=ax)
    #    ax = dataset.plot_mask_on_max_proj([cell],ax=ax)

    ax = placeAxesOnGrid(fig, dim=(1, len(mdf.stim_code.unique())), xspan=(.0, .7), yspan=(.2, .4), wspace=0.35)
    vmax = np.percentile(dataset.traces[cell, :], 99.9)
    ax = plot_transition_type_heatmap(dataset, rdf, [cell], vmax=vmax, ax=ax, cmap='jet', colorbar=False)

    ax = placeAxesOnGrid(fig, dim=(1, 4), xspan=(.0, .7), yspan=(.4, .6), wspace=0.35, sharex=True, sharey=True)
    ax = plot_trial_types_selectivity_ns(dataset, mdf, [cell], window=2, save=False, ax=ax)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.0, .2))
    ax = plot_trial_types_selectivity_pref_image(dataset, mdf, sdf, [cell], ['HIT', 'MISS'], window=[-2, 3], save=False,
                                                 ax=ax)
    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.2, .4))
    ax = plot_trial_types_selectivity_pref_image(dataset, mdf, sdf, [cell], ['FA', 'CR'], window=[-2, 3], save=False,
                                                 ax=ax)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.4, .6))
    #    ax = plot_trace_hist(trace,xlabel='dF/F',ax=ax)
    ax = plot_running_not_running(dataset, cell, sdf, code='change_code', save=False, ax=ax)

    colors = get_colors_for_stim_codes(dataset.stim_codes.stim_code.unique())
    colors.append([1, 1, 1])

    #    if len(mdf.stim_code.unique()) == 8:
    #        ax = pu.placeAxesOnGrid(fig,dim=(4,2),xspan=(.68,0.9),yspan=(.4,.8),wspace=0.2,hspace=0.2)
    #        ax = reformat_axes(ax,num=8)
    #    else:
    #        ax = pu.placeAxesOnGrid(fig,dim=(3,2),xspan=(.68,0.9),yspan=(.4,.8),wspace=0.2,hspace=0.2)
    #        ax = reformat_axes(ax,num=6)
    #    ax = plot_images(dataset,mdf,ax=ax)

    ax = placeAxesOnGrid(fig, dim=(8, 1), xspan=(.83, 1.), yspan=(0, .8), wspace=0.25, hspace=0.25)
    #    ax = reformat_axes(ax,num=8)
    ax = plot_images(dataset, mdf, ax=ax, save=False, orientation='column');

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.6, 0.8))
    ax = plot_engaged_disengaged(dataset, cell, sdf, code='change_code', save=False, ax=ax)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, .7), yspan=(.6, .8))
    ax = plot_trace_and_stuff(dataset, sdf, cell, second_axis='reward_rate', plot_stim=True, ax=ax)

    #    ax = pu.placeAxesOnGrid(fig,dim=(1,1),xspan=(0,.7),yspan=(.8,1.))
    ##    trace = dataset.dff_traces[cell,:]
    ##    ax = plot_trace_summary(trace,ylabel='dF/F',ax=ax)
    #    ax = plot_trace_and_stuff(dataset,sdf,cell,second_axis='task_performance',plot_stim=False,ax=ax)
    ##
    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.2), yspan=(.77, 1))
    ax = plot_mean_resp_heatmap_cell(dataset.df, cell, values='mean_response', index='initial_code',
                                     columns='change_code', analysis_dir=None, ax=ax)

    fig.tight_layout()

    if save:
        fig_title = dataset.expt_id + '-roi_' + str(cell)
        save_figure(fig, figsize, dataset.analysis_dir, fig_title, 'cell_summary_plots')
        plt.close()


def plot_experiment_summary_figure(dataset, mdf, sdf, rdf, save=False):
    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')
    sns.axes_style({'axes.facecolor': 'white'})

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .22), yspan=(0, .22))
    ax.imshow(dataset.max_image, cmap='gray', vmin=0, vmax=np.amax(dataset.max_image) / 2)
    mask = rm.make_filtered_mask(dataset.roi_dict, dataset.roi_dict.keys())
    ax.imshow(mask, cmap='jet', alpha=0.3, vmin=0, vmax=0.7)
    ax.axis('off');
    ax.set_title(dataset.expt_id)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .22), yspan=(.2, .8))
    ax = plot_behavior(dataset.pkl_df, ax=ax)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.18, 0.94), yspan=(0, .3))
    ax = plot_traces_heatmap(dataset, cbar=True, ax=ax)

    ###behavior figures
    times = dataset.sync['visualFrames']['timestamps'][:len(dataset.run_speed)]
    frame_times = []
    reward_rate = []
    pkl_df = dataset.pkl_df
    for i, frame in enumerate(dataset.pkl_df.change_frame.values):
        if frame != 0:
            frame_times.append(times[frame])
            reward_rate.append(dataset.pkl_df.iloc[i].reward_rate)
    upper_limit, time_interval, frame_interval = get_upper_limit_and_intervals(dataset)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.18, .8), yspan=(.3, .44))
    ax = plot_task_performance(frame_times, pkl_df, label=True, ax=ax)
    ax.set_xlim(time_interval[0], np.uint64(upper_limit / 30.))
    ax.set_xticks(np.arange(600, upper_limit / 30, 600))
    ax.set_xlabel('')

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.18, .8), yspan=(.43, .58))
    ax = plot_run_speed(times, frame_times, dataset.run_speed, ax=ax, label=True)
    ax2 = ax.twinx()
    ax2 = plot_reward_rate(frame_times, reward_rate, label=True, ax=ax2)
    ax2.set_title('')
    ax.set_title('reward rate and run speed')
    ax.set_xlim(time_interval[0], np.uint64(upper_limit / 30.))
    ax.set_xticks(np.arange(600, upper_limit / 30, 600))
    ax2.grid(False)
    ax.set_xlabel('')
    ax2.set_xlabel('')

    ###mean trace
    mean_trace = []
    for roi in range(dataset.dff_traces.shape[0]):
        trace = dataset.dff_traces[roi]
        if (np.amax(trace) > 20) or (np.amax(trace) < -20):
            print str(roi), ' bad trace'
        else:
            mean_trace.append(trace)
    mean_trace = np.mean(np.asarray(mean_trace), axis=0)
    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.18, .8), yspan=(.57, .71))
    times_2p = dataset.sync['2PFrames']['timestamps'][:len(mean_trace)]
    ax.plot(times_2p, mean_trace)
    ax.set_xlim(0, upper_limit / 30)
    # ax.set_xticks(np.arange(600*30,upper_limit,600*30))
    ax.set_xticks(np.arange(600, upper_limit / 30, 600))
    ax = add_stim_color_span(dataset, ax=ax)
    ax.set_title('mean trace')
    ax.set_xlabel('time(s)')

    ###response matrices.
    if dataset.expt_id[:2] == '16':
        dataset.stim_codes = dataset.stim_codes.rename(
            columns={'image_name': 'image_num', 'full_image_name': 'image_name'})
        dataset.stim_codes['full_image_name'] = dataset.stim_codes.image_name.values
    sc = dataset.stim_codes
    tmp = pkl_df.copy()
    tmp = tmp[tmp.trial_type != 'aborted']
    tmp = tmp[tmp.trial_type != 'other']
    if dataset.stimulus_type == 'grating':
        #        tmp = pkl_df[pkl_df.trial_type!='aborted']
        tmp = tmp.rename(columns={'initial_ori_str': 'initial_image', 'change_ori_str': 'change_image'})

    tmp['initial_code'] = [sc[sc.image_name == tmp.iloc[trial].initial_image].stim_code.values[0] for trial in
                           range(len(tmp))]
    tmp['change_code'] = [sc[sc.image_name == tmp.iloc[trial].change_image].stim_code.values[0] for trial in
                          range(len(tmp))]

    # ax = pu.placeAxesOnGrid(fig,dim=(1,1),xspan=(.75,1.),yspan=(0.,.25))
    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.5, .75), yspan=(.7, .99))
    ax = create_resp_prob_heatmap_general(tmp, index='initial_code', columns='change_code', ax=ax, cmap='magma',
                                          filter_by_reward_rate=False)

    # ax = pu.placeAxesOnGrid(fig,dim=(1,1),xspan=(.75,1.),yspan=(.25,.5))
    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.25, .5), yspan=(.7, .99))
    ax = create_resp_prob_heatmap_general(tmp, values='trial_num', index='initial_code', columns='change_code',
                                          aggfunc=np.sum, ax=ax, cmap='magma', filter_by_reward_rate=False)

    ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.77, 1.), yspan=(.55, .99))
    ax = plot_response_matrix(mdf, values='mean_response', index=['cell'], columns=['stim_code'], sig=True, ax=ax)

    fig.tight_layout()

    save = True
    if save:
        save_figure(fig, figsize, dataset.analysis_dir, dataset.expt_id + '_DoC', 'expt_summary', formats=['.png'])
        analysis_dir = r'\\aibsdata2\nc-ophys\BehaviorImaging\DoC'
        save_figure(fig, figsize, analysis_dir, dataset.expt_id + '_DoC', 'experiment_summaries')
