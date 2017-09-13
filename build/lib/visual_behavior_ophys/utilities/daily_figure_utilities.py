'''
##################################
######Maintained by Sahar#########
##################################
'''

import numpy as np
import pandas as pd
import os
import sys
import time
import fnmatch
import socket
import warnings
import collections
import math

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import imaging_behavior.core.utilities as ut
import imaging_behavior.plotting.plotting_functions as pf
import imaging_behavior.plotting.utilities as pu
import seaborn as sns 

import platform
import imp
import getpass

if platform.system()=='Linux':
    dro_path = os.path.dirname(os.path.realpath(__file__))
    imp.load_source('dro',os.path.join(dro_path,'utilities.py'))
    import dro as du
elif getpass.getuser() == 'dougo':
    dro_path = '/Users/dougo/Dropbox/PythonCode/dro'
elif getpass.getuser() == 'saharm':
    dro_path = "C:\Users\saharm\Documents\dro\dro"
else:
    dro_path = '//aibsdata2/nc-ophys/BehaviorCode/dro'

imp.load_source('dro',os.path.join(dro_path,'utilities.py'))
import dro as du


def exclude_aborted_method(df):
    df = df[df.trial_type!='other']
    df = df[df.trial_type!='aborted']

    return df

def create_df_summary(df_sum, exclude_aborted):
    if exclude_aborted==True:
        exclude_aborted_method(df_sum)
         
    dfs = []
    dates = []
    for date in df_sum.startdatetime.unique():
        df = df_sum[df_sum.startdatetime==date].reset_index()
        dfs.append(df)
        dates.append('{:.10}'.format(date))
        
    return dfs, dates, df_sum

def show_images(stim_type, datapath='None', df='None'):
    '''datapath should either be path to a pkl file OR 
    datapath (to file folder) & df need to be called'''

    if ".pkl" in datapath:
        pkl = pd.read_pickle(datapath)
    else:
        pkl = pd.read_pickle(os.path.join(datapath, df.iloc[0]['filename']))

    num_categories = len(pkl['image_dict'].keys())
    images_per_category = len(pkl['image_dict'][pkl['image_dict'].keys()[0]])
    num_images = num_categories*images_per_category

    if stim_type=='MNIST':
        ax = plt.subplots(num_categories,images_per_category)
    if stim_type=='NaturalImage':
        ax = plt.subplots(images_per_category,num_categories)
    
    for x,category in enumerate(pkl['image_dict'].keys()):
        for y,image in enumerate(pkl['image_dict'][category].keys()):
            img = pkl['image_dict'][category][image]
            
            if stim_type=='MNIST':
                ax[x,y].imshow(img,cmap='gray',vmin=0,vmax=np.amax(img))
                ax[x,y].grid('off')
                ax[x,y].axis('off')
                ax[x,y].set_title(image)
            if stim_type=='NaturalImage':
                ax[x].imshow(img,cmap='gray',vmin=0,vmax=np.amax(img))
                ax[x].grid('off')
                ax[x].axis('off')
                ax[x].set_title(image)

    # return fig



def make_lick_raster(df, ax, exclude_aborted, xmin=-4, xmax=8, figsize=None):

    if exclude_aborted==True:
        df = exclude_aborted_method(df)
    
    for lap in range(len(df)):
        trialstart = df.iloc[lap]['starttime'] - df.iloc[lap]['change_time']
        licktimes = [(t - df.iloc[lap]['change_time']) for t in df.iloc[lap]['lick_times']]
        trialend = trialstart + df.iloc[lap]['trial_length']
        rewardtime = [(t - df.iloc[lap]['change_time']) for t in df.iloc[lap]['reward_times']]
        if len(rewardtime) > 1:
            rewardtime = rewardtime[0]
    
        if df.iloc[lap]['auto_rewarded']==True:
            ax.axhspan(lap, lap + 1, -200, 200, color='gray', alpha=.5)
            ax.plot(rewardtime, lap+0.5,'.',color='b',label='reward', markersize=6)
        if df.iloc[lap]['trial_type']=='go':
            if (df.iloc[lap]['auto_rewarded']==True)==False:
                if df.iloc[lap]['response_type']=='HIT':
                    ax.axhspan(lap, lap + 1, -200, 200, color='#55a868', alpha=.5)
                    ax.plot(rewardtime, lap+0.5,'.',color='b',label='reward', markersize=6)
                else:
                    ax.axhspan(lap, lap + 1, -200, 200, color='#ccb974', alpha=.5)
        if df.iloc[lap]['trial_type']=='catch':
            if df.iloc[lap]['response_type']=='FA':
                ax.axhspan(lap, lap + 1, -200, 200, color='#c44e52', alpha=.5)
            else:
                ax.axhspan(lap, lap + 1, -200, 200, color='#4c72b0', alpha=.5)


        ax.vlines(trialstart, lap, lap + 1, color='black',linewidth=1)    
        ax.vlines(licktimes, lap, lap + 1, color='r',linewidth=1)
        ax.vlines(0, lap, lap + 1, color=[.5,.5,.5],linewidth=1)
    
    
    ax.axvspan(df.iloc[0]['response_window'][0],df.iloc[0]['response_window'][1], facecolor='gray', alpha=.4, edgecolor='none')
    ax.grid(False)  
    ax.set_ylim(0, len(df))
    ax.set_xlim([xmin,xmax])



def make_first_licks_his(df, ax, bins=np.arange(0,2,.02), xmin=-1, xmax=4):
    reward_adj_first_lick = []

    for lap in range(df.index[0],len(df)):
        if len(df.lick_times[lap])>0:
            first_lick = df.lick_times[lap]
            reward_adj_first_lick.append(first_lick[0] - df.change_time[lap])

    reward_adj_first_lick = [x for x in reward_adj_first_lick if 2 > x > 0]

    bins = bins
    ax.hist(reward_adj_first_lick, bins=bins, histtype='barstacked', color='black')

    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.set_xlim([xmin,xmax])


def get_response_rates(df_in, exclude_aborted, sliding_window=100,reward_window=None):
    '''copied from doug's utilities but customized'''
    if exclude_aborted==True:
        try:
            df_in = exclude_aborted_method(df_in)
            # print 'exclude aborted worked'
        except:
            pass
            

    try:
        df_in = df_in.reset_index()
        # print 'index reset'
    except Exception as e:
        # print 'exception for reset index', e
        pass
    
    go_responses = pd.Series([np.nan]*len(df_in))
    go_responses[df_in[(df_in.trial_type=='go')&(df_in.response==1)].index] = 1
    go_responses[df_in[(df_in.trial_type=='go')&((df_in.response==0)|np.isnan(df_in.response))].index] = 0
    hit_rate = go_responses.rolling(window=100,min_periods=0).mean()

    catch_responses = pd.Series([np.nan]*len(df_in))
    catch_responses[df_in[(df_in.trial_type=='catch')&(df_in.response==1)].index] = 1
    catch_responses[df_in[(df_in.trial_type=='catch')&((df_in.response==0)|np.isnan(df_in.response))].index] = 0
    catch_rate = catch_responses.rolling(window=100,min_periods=0).mean()

    d_prime = du.dprime(hit_rate,catch_rate)

    return hit_rate.values,catch_rate.values,d_prime


def make_rolling_response_probability_plot(hit_rate,fa_rate,ax):

    ax.plot(hit_rate,np.arange(len(hit_rate)),color='#55a868',linewidth=3, alpha=.9)
    ax.plot(fa_rate,np.arange(len(fa_rate)),color='#c44e52',linewidth=3, alpha=.9)

    ax.set_title('FA & Hit rates',fontsize=14)
    ax.set_xticks([.25,.5,.75,1])
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)



def make_info_table(df, ax, datapath, stim_type, frame_type='single', bbox=[.6,0,.65,1]):
    '''
    generates a table with info extracted from the dataframe
    DRO - 10/13/16
    modified by SM
    '''

    #define the data
    try:
        user_id = df.iloc[0]['user_id']
    except:
        user_id = 'unspecified'

    try:
    	task_id = df.iloc[0]["task"]
    	task_id = task_id.split('_')[1]
    except:
    	task_id = df.iloc[0]["task"]
    
    if ".k" in datapath:
        orig_pkl = pd.read_pickle(datapath)
        pkl_folder = datapath.split('output')[0]
        pkl_folder = os.path.join(pkl_folder, 'output')
    else:
        orig_pkl = pd.read_pickle(os.path.join(datapath, df.iloc[0]['filename']))
        pkl_folder = datapath
        
    try:
        groups = df.groupby(['trial_type']).size()
        percent_aborted = (np.divide(groups['aborted'],float(groups.sum()))*100)
        percent_aborted = ('{:.1f}'.format(percent_aborted)+"%")
    except:
        percent_aborted = str('aborted trials excluded')

    #Sahar's imperfect way of calculating training day
    training_day = len([f for f in os.listdir(pkl_folder) if fnmatch.fnmatch(f, '*DoC*.pkl')])    
    set_catch = orig_pkl['catch_frequency']
    stimulus = orig_pkl['stimulus']
    LDT_mode = orig_pkl['lick_detect_training_mode']
    LDT_blocks = orig_pkl['lick_detect_training_block_length']


    #I'm using a list of lists instead of a dictionary so that it maintains order
    #the second entries are in quotes so they can be evaluated below in a try/except
    
    if frame_type=='single':
        data = [['Mouse ID','df.iloc[0]["mouse_id"]'],
                ['Trained by','user_id'],
                ['Date','df.iloc[0].startdatetime.strftime("%m-%d-%Y")'], 
                ['Training Day','training_day'],
                ['Total water received (ml)','df["cumulative_volume"].max()'],
                ['Percent of Trials Aborted','percent_aborted'],
                ['Training Time','df.iloc[0].startdatetime.strftime("%H:%M")'],
                ['Duration (minutes)','round(df.iloc[0]["session_duration"]/60.,2)'],
                ['Rig ID','df.iloc[0]["rig_id"]'],
                
                ['Stimulus','stimulus'],
                ['Set Catch Trial Freq','set_catch'],
                ['Response Window','df.iloc[0].response_window'],
                ['Minimum pre-change time','df.iloc[0]["prechange_minimum"]'],
                ['Trial duration','df.iloc[0].trial_duration'],
                ['Inter-stimulus interval','df.iloc[0].blank_duration_range[0]'],
                ['Lick Detect Training','LDT_mode, LDT_blocks']            
                ]

    if frame_type=='multi':
        num_of_sessions = str(len(create_df_summary(df, exclude_aborted=False)[0]))
        first_session = df.date[0]
        last_session = df.date[len(df)-1]
        current_stim = df.stimulus[len(df)-1]
        
        stim_pattern = str('*'+stim_type+'*')
        stim_sessions = len([f for f in os.listdir(pkl_folder) if fnmatch.fnmatch(f, stim_pattern)])

        data = [['Mouse ID','df.mouse_id[0]'],
                ['Number of Sessions','num_of_sessions'],
                ['First Session','first_session'],
                ['Most Recent Session','last_session'],
                ['Current Stim Set','current_stim'],
                ['Sessions With Current Stim', 'stim_sessions']

                ]


    cell_text = []
    for x in data:
        try:
            cell_text.append([eval(x[1])])
        except:
            cell_text.append([np.nan])

    #define row colors
    row_colors = [['lightgray'],['white']]*(len(data))


    #make the table
    table = ax.table(cellText=cell_text,
                          rowLabels=[x[0] for x in data],
                          rowColours=du.flatten_list(row_colors)[:len(data)],
                          colLabels=None,
                          loc='center',
                          cellLoc='left',
                          rowLoc='right',
                          cellColours=row_colors[:len(data)],
                         bbox=bbox)
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    #do some cell resizing
    cell_dict=table.get_celld()
    for cell in cell_dict:
        if cell[1] == -1:
            cell_dict[cell].set_width(0.1)
        if cell[1] == 0:
            cell_dict[cell].set_width(0.5)



def response_prob_heatmap(df, ax, stim_type, cmap='magma', multi_ses=False, first_ses='all'):
    '''stim_type options: 'NaturalImage', 'MNIST', 'gratings' or 'None' 
        set multi_ses to True if dataframe has multiple sessions
        set first_ses: 'all' uses all sessions in dataframe, or enter date 'YYYY-MM-DD' to start with that date
        -- only works if multi_ses=True'''

    if multi_ses==True:
        df = create_df_summary(df, exclude_aborted=False)[2]

        if first_ses is not 'all':
            grouped = df.groupby(['date']).size()
            idx_df = pd.DataFrame(grouped)
            idx_df['idx'] = np.arange(len(idx_df))
            ses_n = idx_df.get_value(first_ses,'idx')
            ses_n = ses_n-len(idx_df)

            df_list = create_df_summary(df, exclude_aborted=False)[0]
            df = pd.concat(df_list[ses_n:])

    #make the response_matrix
    if stim_type=='NaturalImage':
        response_matrix = pd.pivot_table(df,
                values='response',
                index=['initial_image_name'],
                columns=['change_image_name'])

    if stim_type=='MNIST':
        response_matrix = pd.pivot_table(df,
                values='response',
                index=['initial_image_category', 'initial_image_name'],
                columns=['change_image_category', 'change_image_name'])

    if stim_type=='gratings':
        df['initial_ori_adj'] = pd.Series(df.initial_ori%360., index=df.index)
        df['change_ori_adj'] = pd.Series(df.change_ori%360., index=df.index)
        df['initial_ori_adj'] = df['initial_ori_adj'].astype('str')
        df['change_ori_adj'] = df['change_ori_adj'].astype('str')
        df.replace(['0.0', '270.0', 'nan', '180.0', '90.0'], ['vert', 'horiz', np.nan, 'vert', 'horiz'], inplace=True)

        response_matrix = pd.pivot_table(df,
                    values='response',
                    index=['initial_ori_adj'],
                    columns=['change_ori_adj'])

    heatmap = sns.heatmap(response_matrix, cmap=cmap, linewidths=0, linecolor='white', square=True, annot=True, 
                      annot_kws={"fontsize":12}, vmin=0, vmax=1,
                 robust=True, cbar_kws={"drawedges":False, "shrink":1}, ax=ax)

    ax.set_title('resp prob by image since: '+str(first_ses), fontsize=16, va='bottom', ha='center')
    ax.set_xlabel('final', fontsize=14)
    ax.set_ylabel('initial', fontsize=14)

    if stim_type=='NaturalImage':
        new_names = []
        rev_names = []
        for name in range(len(response_matrix.index)):
            

            if '_' in str(response_matrix.index[name]):
                name_one = response_matrix.index[name].split('_')[0]
                new_names.append(name_one)
                rev_names.append(name_one)
            elif '.' in str(response_matrix.index[name]):
                name_one = str(response_matrix.index[name]).split('.')[0]
                new_names.append(name_one.split('k')[1])
                rev_names.append(name_one.split('k')[1])
        rev_names.reverse()
        
        ax.set_xticklabels(new_names)
        ax.set_yticklabels(rev_names)


def nat_img_detectability(df, exclude_aborted, ax, cmap='magma', multi_ses=False):
    if multi_ses==True:
        df = create_df_summary(df, exclude_aborted=exclude_aborted)[2]
        
    response_matrix = pd.pivot_table(df,
                values='response',
                index=['initial_image_name'],
                columns=['change_image_name'])

    samesies = [response_matrix[col][col] for col in response_matrix.columns]

    name_dict = {}
    for name in range(len(response_matrix.index)):
        try:
            if '_' in str(response_matrix.index[name]):
                name_dict[name] = response_matrix.index[name].split('_')[0]
            elif '.' in str(response_matrix.index[name]):
                name_dict[name] = (str(response_matrix.index[name]).split('.')[0]).split('k')[1]
        except:
            name_dict[name] = name
            
    new_matrix = response_matrix
    for col in new_matrix.columns:
        new_matrix = new_matrix.set_value(col, col, np.nan)

    df = pd.DataFrame([np.nanmean(new_matrix[col]) for col in new_matrix.columns]).transpose()
    df = df.rename(columns = name_dict)
    df['catch'] = np.nanmean(samesies)
    
    heatmap = sns.heatmap(df, linewidths=0,  cmap=cmap, square=True, robust=True, annot=True,
                     vmax = 1, vmin=0, fmt='.2f', cbar=False, ax=ax)
    
            
    ax.set_title('mean detectability of final images', fontsize=16, va='bottom', ha='center')
    ax.set_xticklabels(df.columns, rotation='horizontal')
    ax.set_yticklabels([])
    ax.set_xlabel('final image', fontsize=14)


def save_sum_fig(fig, mouse, date, root_dir, text_str=None):
    '''saves a figure in folder titled root_dir+mouse. fig name is date_mouse_text_str.png
    enter date in YYYY-MM-DD or YYMMDD format, or enter None for multi-session figures'''
    
    if date==None:
        r_date = ''
    else:
        try:
            r_date = date.split('-')
            r_date = ''.join(r_date[0:3])
            r_date = r_date[2:8]
        except:
            r_date = date
    
    fig_name = str(r_date+'_'+mouse+'_'+text_str+'.png')
    fig_dir = os.path.join(root_dir,mouse)
    img_path = os.path.join(fig_dir,fig_name)
    
    try:
        fig.savefig(img_path, bbox_inches='tight')
        print "your file was saved at ", img_path
    except Exception as e:
        print e
        print "your file was not saved"

def make_daily_summary(df, stim_type, datapath, exclude_aborted=True):
    '''stim_type options: 'bird_shroom', 'NaturalImage', 'MNIST', 'gratings' or 'None'
    set exclude_aborted to False if dataframe already excludes aborted trials
    '''

    df_a = df[df.trial_type!='other']
    df_a = df_a[df_a.trial_type!='aborted'].reset_index()

    if ".pkl" in datapath:
        orig_pkl = pd.read_pickle(datapath)
    else:
        orig_pkl = pd.read_pickle(os.path.join(datapath, df.iloc[0]['filename']))

    ### FORMATTING
    fig = plt.figure(figsize=(15,8))
    gd = gs.GridSpec(4,5)

    ax0 = plt.subplot(gd[:3,0]) #lick raster
    ax1 = plt.subplot(gd[3,0]) # first licks histogram
    ax2 = plt.subplot(gd[:3,1]) #d' plot
    ax3 = plt.subplot(gd[:3,2]) #resp prob
    ax4 = plt.subplot(gd[2:4, 3:]) #heatmap
    # ax5 = plt.subplot(gd[3,1:3]) # -- IN IF STATEMENT BELOW-- extra nat_img heatmap
    ax6 = plt.subplot(gd[0:2,3:4]) #info table
    # ax7 = plt.subplot(gd[3,2]) #show images


    sns.set_style('darkgrid')

    try:
        task_id = df.iloc[0]["task"]
        task_id = task_id.split('_')[1]
    except:
        task_id = df.iloc[0]["task"]

    plt.suptitle(df.iloc[0]['mouse_id']+' '+'{:.10}'.format(str(df.iloc[0]['startdatetime']))+' '+task_id, 
                 fontsize=16, fontweight='bold', va='bottom', ha='center', y=1.05)

    plt.tight_layout(pad=0, w_pad=-1.2, h_pad=0)


    ### ax0: LICK RASTER
    make_lick_raster(df, ax0, exclude_aborted=exclude_aborted, xmin=-.5, xmax=4)         
    ax0.set_title('lick raster & first licks', fontsize=14)
    ax0.set_ylabel('trial number', fontsize=12)
    ax0.set_xticks([])


    ### ax1: FIRST LICKS HISTOGRAM
    make_first_licks_his(df_a, ax1, xmin=-.5)


    ax1.set_xlabel('s after image change', fontsize=12)
    ax1.set_xticks([0,1,2,3,4])


    ### ax2: D' & REWARD RATE
    ax2.plot(df_a['reward_rate'], df_a.index, color='b', linewidth=3, alpha=.9)
    

    ax2.plot(get_response_rates(df, exclude_aborted=exclude_aborted)[2], range(len(get_response_rates(df, exclude_aborted=exclude_aborted)[2])), 
        color='purple', linewidth=3, alpha=.9)

    ax2.spines["top"].set_visible(False)  
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax2.set_ylim(-1, len(df_a)+1)
    ax2.set_yticklabels('', visible=False)
    ax2.set_xticks(np.arange(0,6,1))

    ax2.set_title("d' & reward rate", fontsize=14)

    ### ax3: HIT & FA RATES
    make_rolling_response_probability_plot(get_response_rates(df, exclude_aborted=exclude_aborted)[0],
        get_response_rates(df, exclude_aborted=exclude_aborted)[1],ax3)
    ax3.set_ylim(-1, len(df_a) + 1)
    ax3.set_yticklabels('', visible=False)

    ### ax4: HEAT MAP
    if stim_type is not 'None':
        response_prob_heatmap(df_a, ax4, stim_type, cmap='magma', multi_ses=False, first_ses='all')
    else:
        pass

    ### AX5: extra nat_img heatmap
    if stim_type=='NaturalImage':
        ax5 = plt.subplot(gd[3,1:3]) 
        nat_img_detectability(df, exclude_aborted=exclude_aborted, ax=ax5, cmap='magma', multi_ses=False)


    ### ax6: INFO TABLE
    make_info_table(df, ax6, datapath=datapath, stim_type=stim_type, bbox=[1,.05,.7,1])
    ax6.set_axis_off()

    ### AX7: show images
    # show_images(stim_type=stim_type, ax=ax7, datapath=datapath, df=df)
    # ax7.images.append(show_images(stim_type=stim_type, ax=ax7, datapath=datapath, df=df))
    # ax7.set_axis_off()

    return fig

###########################################
###########################################
###########################################
###########################################
###########################################
########multi-session plot stuff below here
###########################################
###########################################
###########################################
###########################################
###########################################

def plot_dprime_multi_sess(df_sum, ax):
    
    i_dfs = create_df_summary(df_sum, exclude_aborted=False)[0]
    
    dates = []
    for date in create_df_summary(df_sum, exclude_aborted=False)[1]:
        d = date.split('-')
        d = '{}-{}'.format(d[1],d[2])
        dates.append(d)
    
    dp_session_means = []
    dp_session_std = []
    dp_session_max = []
    for x in range(len(i_dfs)):
        dp_session_means.append(np.nanmean([rate for rate in get_response_rates(i_dfs[x],exclude_aborted=False)[2] if rate > 0]))
        dp_session_std.append(np.nanstd([rate for rate in get_response_rates(i_dfs[x],exclude_aborted=False)[2] if rate > 0]))
        dp_session_max.append(np.nanmax([rate for rate in get_response_rates(i_dfs[x],exclude_aborted=False)[2]]))

    dp_session_means = np.nan_to_num(dp_session_means)
    dp_session_std = np.nan_to_num(dp_session_std)
    dp_session_max = np.nan_to_num(dp_session_max)
    
    x_vals = np.arange(1,len(dp_session_means)+1, 1)

    ax.plot(x_vals, dp_session_means, 'o-', color='#4f0672')
    ax.fill_between(x_vals, np.subtract(dp_session_means,dp_session_std), np.add(dp_session_means,dp_session_std), color='#4f0672', alpha=.4)

    ymin = 0
    ymax = 5
    ax.set_yticks(np.arange(ymin,ymax+.5,1))

    ax.set_xticks(x_vals)
    ax.set_xlim([0, len(dp_session_means)+.5])
    
    ax.set_title('mean dprime')
    ax.set_ylabel('dprime')
    ax.set_xticklabels(dates)

def first_lick_timing(df_sum, ax, exclude_aborted, legend_loc=(.9, 1, 0, 0)):
    
    i_dfs = create_df_summary(df_sum,exclude_aborted)[0]
    if len(i_dfs) > 7:
        i_dfs = i_dfs[-7:]

    color_list = plt.cm.gist_rainbow(np.linspace(0, 1, len(i_dfs)))
    
    dates = []
    for date in create_df_summary(df_sum, exclude_aborted)[1]:
        d = date.split('-')
        d = '{}-{}'.format(d[1],d[2])
        dates.append(d)

    if len(dates) > 7:
        dates = dates[-7:]
    
    first_lick_times = []
    for x in range(len(i_dfs)):
        templist = []
        for lap in range(len(i_dfs[x])):
            if len(i_dfs[x].lick_times[lap]) > 0:
                licks = i_dfs[x].lick_times[lap]
                templist.append(licks[0] - i_dfs[x].change_time[lap])
        templist = np.asarray(templist) 
        templist = templist[~np.isnan(templist)]    
        first_lick_times.append(templist)
        del templist
        
    bins = np.arange(0,4,.2)
    for x in range(len(first_lick_times)):
        counts,bin_edges = np.histogram(first_lick_times[x],bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
        ax.errorbar(bin_centres, counts, fmt='-o', label=dates[x], color=color_list[x], alpha=.7)
        
    
    ax.hist(first_lick_times[len(i_dfs)-1], bins=bins, alpha=.3, color=color_list[len(i_dfs)-1]) 

    ax.legend(bbox_to_anchor=legend_loc, loc=2, borderaxespad=0.)

    ymin = 0
    ymax = 130
    ax.set_yticks(np.arange(ymin,ymax+10,20))
    ax.set_title('First Lick Timing')
    ax.set_ylim([0, ymax])

    ax.set_ylabel('lick count')
    ax.set_xlabel('seconds after image change')


def trial_types_plot_ms(df_sum, ax, exclude_aborted, remove_auto_rew=True):

    if remove_auto_rew==True:
        df_sum = df_sum[df_sum.auto_rewarded!=True]
    
    i_dfs = create_df_summary(df_sum, exclude_aborted)[0]
    
    dates = []
    for date in create_df_summary(df_sum, exclude_aborted)[1]:
        d = date.split('-')
        d = '{}-{}'.format(d[1],d[2])
        dates.append(d)
    
    x_vals = np.arange(1, len(i_dfs)+1, 1)

    go_trials = []
    hit_trials = []
    miss_trials = []
    for x in range(len(i_dfs)):
        go_trials.append(len(i_dfs[x][i_dfs[x].trial_type=='go']))
        hit_trials.append(len(i_dfs[x][i_dfs[x].response_type=='HIT']))
        miss_trials.append(len(i_dfs[x][i_dfs[x].response_type=='MISS']))
        
    ax.plot(x_vals, hit_trials, 'o-', color='#55a868', linewidth=4, label="Hit")
    ax.plot(x_vals, miss_trials, 'o-',color='#ccb974', linewidth=4, label="Miss")

    ax.set_xticks(x_vals)
    ax.set_xlim([0, len(i_dfs)+.5])

    ax.set_title('# of hit and miss trials')
    ax.set_ylabel('# of trials')
    ax.set_xticklabels(dates)

def catch_fa_bars(df_sum, ax):
    i_dfs = create_df_summary(df_sum, exclude_aborted=False)[0]
    
    dates = []
    for date in create_df_summary(df_sum, exclude_aborted=False)[1]:
        d = date.split('-')
        d = '{}-{}'.format(d[1],d[2])
        dates.append(d)
    
    x_vals = np.arange(1, len(i_dfs)+1, 1)

    catch_trials = []
    fa_trials = []
    for x in range(len(i_dfs)):
        catch_trials.append(len(i_dfs[x][i_dfs[x].trial_type=='catch']))
        fa_trials.append(len(i_dfs[x][i_dfs[x].response_type=='FA']))
    
    data = ([x for x in catch_trials],
            [x for x in fa_trials])

    rows = ['catch', 'fa']
    colors = ['#c48b8d', '#c44e52']

    for row in range(len(rows)):
        ax.bar(x_vals-.25, data[row],  color=colors[row], width=.5, linewidth=0)
        
    ax.set_title('# of catch and FA trials')

    ax.set_xticks(x_vals)
    ax.set_xticklabels(dates)
    ax.set_xlim([0, len(i_dfs)+.5])

    ax.set_ylabel('# of trials')

def aborted_trials_bars(df_sum, ax):
    i_dfs = create_df_summary(df_sum, exclude_aborted=False)[0]
    
    dates = []
    for date in create_df_summary(df_sum, exclude_aborted=False)[1]:
        d = date.split('-')
        d = '{}-{}'.format(d[1],d[2])
        dates.append(d)
    
    x_vals = np.arange(1, len(i_dfs)+1, 1)

    total_trials = []
    aborted_trials = []
    for x in range(len(i_dfs)):
        total_trials.append(len(i_dfs[x]))
        aborted_trials.append(len(i_dfs[x][i_dfs[x].trial_type=='aborted']))
    
    data = ([x for x in total_trials],
            [x for x in aborted_trials])

    rows = ['total', 'aborted']
    colors = ['gray', 'black']

    for row in range(len(rows)):
        ax.bar(x_vals-.25, data[row],  color=colors[row], width=.5, linewidth=0)
        
    ax.set_title('# of total and aborted trials')

    ax.set_xticks(x_vals)
    ax.set_xticklabels(dates)
    ax.set_xlim([0, len(i_dfs)+.5])

    ax.set_ylabel('# of trials')

def MS_summary_plot(df_sum, datapath, stim_type, first_ses='all', learning=True, exclude_aborted=False):
    '''stim_type options: 'NaturalImages', 'MNIST', 'gratings' or 'None'
    if learning=True: first lick timing plot is shown instead of catch/FA plot
    set exclude_aborted to False if aborted trials are already removed from dataframe
    hit&miss plot excludes auto rewards'''

    df_sum_a = df_sum[df_sum.trial_type!='other']
    df_sum_a = df_sum_a[df_sum_a.trial_type!='aborted']

    fig = plt.figure(figsize=(15,8))
    gd = gs.GridSpec(4,6)

    sns.set_style('darkgrid')

    try:
        task_id = df_sum.iloc[len(df_sum)-1]["task"]
        task_id = task_id.split('_')[1]
    except:
        task_id = df_sum.iloc[len(df_sum)-1]["task"]
    
    plt.suptitle(df_sum.mouse_id[0]+' - Multiple Sessions - '+task_id, 
                 fontsize=16, fontweight='bold', va='bottom', ha='center', y=1.05)
    
    ax0 = plt.subplot(gd[0,0:4]) #dprime
    ax1 = plt.subplot(gd[1,0:4]) #hit&miss
    ax2 = plt.subplot(gd[2,0:4]) #first licks OR catch/FA 
    ax3 = plt.subplot(gd[0, 4:6]) #datatable
    ax4 = plt.subplot(gd[1:3,4:6]) #heatmap
    ax5 = plt.subplot(gd[3,0:4]) #aborted trials
    #ax6 - extra natural images detectability plot


    
    plot_dprime_multi_sess(df_sum_a, ax=ax0)
    
    trial_types_plot_ms(df_sum_a, ax=ax1, exclude_aborted=True, remove_auto_rew=True)
    
    #ax2 - first licks or catch/fa
    if learning==True:
        first_lick_timing(df_sum_a, ax=ax2, exclude_aborted=exclude_aborted, legend_loc=(.9, 1, 0, 0))
    if learning==False:
        catch_fa_bars(df_sum, ax=ax2)
        
    #ax3 - info table
    make_info_table(df_sum, ax=ax3, datapath=datapath, stim_type=stim_type, frame_type='multi', bbox=[.3,0,.65,1])
    ax3.set_axis_off()
    
    
    #ax4 - heatmap
    # if stim_type=='MNIST':
    #     create_mnist_heatmap(df_sum_a, ax=ax4, multi_ses=True, exclude_aborted=False)
    # if stim_type=='NaturalImage':
    #     make_natim_heatmap(df_sum_a, ax=ax4, multi_ses=True,exclude_aborted=False)
    # if stim_type=='gratings':
    #     create_gratings_heatmap(df_sum_a, ax=ax4, multi_ses=True, exclude_aborted=False)
    # if stim_type=='none':
    #     pass

    if stim_type is not 'None':
        response_prob_heatmap(df_sum_a, ax4, stim_type, cmap='magma', multi_ses=True, first_ses=first_ses)
    else:
        pass

    #ax5 - aborted trials bars
    aborted_trials_bars(df_sum, ax=ax5)

    #ax6 - extra natural images detectability plot
    if stim_type=='NaturalImage':
        ax6 = plt.subplot(gd[3,4:6])
        nat_img_detectability(df_sum_a, exclude_aborted=False, ax=ax6, cmap='magma', multi_ses=True)


    plt.tight_layout(pad=-.5, h_pad=0, w_pad=.5)
    # if stim_type=='gratings':
    #     plt.tight_layout(pad=-.5, h_pad=0, w_pad=.5)
    # else:
    #     plt.tight_layout(pad=-.5, h_pad=-2, w_pad=.5)
    
    return fig