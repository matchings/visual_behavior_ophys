
# coding: utf-8

# In[1]:

import os
import h5py
import scipy.io
import numpy as np
import pandas as pd
from sync import Dataset
import cPickle as pickle
import scipy.stats as stats
import imaging_behavior as ib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ophys.error_rate as er
from toolbox.misc.ophystools import filter_digital
import imaging_behavior.plotting.plotting_2AFC as pa
from aibs.dev.chrisdev import importRawDphys as ird
from aibs.Analysis.InDevelopment.plotTools import saveFigure 

import seaborn as sns 
sns.set_style('darkgrid')
sns.set_context('notebook',font_scale=1.5) 

# In[ ]:


def get_sync(sync_path, pkl):
    d = Dataset(sync_path)
    meta_data = d.meta_data
    sample_freq = meta_data['ni_daq']['counter_output_freq'] 
    # 2P vsyncs
    vs2p_r = d.get_rising_edges('vsync_2p')
    # vs2p_f = d.get_falling_edges('vsync_2p')
    #Convert to seconds
    vs2p_rsec = vs2p_r/sample_freq
    # vs2p_fsec = vs2p_f/sample_freq
    frames_2p = vs2p_rsec
    print("Total 2P frames: %s" % len(frames_2p))
    # stimulus vsyncs
    vs_r = d.get_rising_edges('vsync_stim')
    vs_f = d.get_falling_edges('vsync_stim')
    # convert to seconds
    vs_r_sec = vs_r/sample_freq
    vs_f_sec = vs_f/sample_freq
    print("Detected vsyncs: %s" % len(vs_f_sec))
    # filter out spurious, transient blips in signal
    if len(vs_r_sec) >= len(vs_f_sec): 
        vs_r_sec = vs_r_sec[:len(vs_f_sec)]
    elif len(vs_r_sec) <= len(vs_f_sec):
        vs_f_sec = vs_f_sec[:len(vs_r_sec)]
#        print 'falling',len(vs_f_sec)
#        print 'rising',len(vs_r_sec)
#        print 'vsync',self.pkl['vsynccount']
    if len(vs_f_sec) != pkl['vsynccount']:
        vs_r_sec_f, vs_f_sec_f = filter_digital(vs_r_sec, vs_f_sec)
        print 'Spurious vsyncs:',str(len(vs_f_sec)-len(vs_f_sec_f))
    else: 
        vs_r_sec_f = vs_r_sec
        vs_f_sec_f = vs_f_sec
    if vs_r_sec_f[1] - vs_r_sec_f[0] > 0.2:
        vsyncs = vs_f_sec_f[1:]
    else:
        vsyncs = vs_f_sec_f
    print("Actual vsync frames: %s" % len(vsyncs))  
    #add lick data 
    lick_0 = d.get_rising_edges('lick0')/sample_freq
    lick_1 = d.get_rising_edges('lick1')/sample_freq
    #put sync data in dphys format to be compatible with downstream analysis
    times_2p = {'timestamps':frames_2p}
    times_vsync = {'timestamps':vsyncs}
    times_lick0 = {'timestamps':lick_0}
    times_lick1 = {'timestamps':lick_1}
    sync = {'2PFrames':times_2p,
            'visualFrames':times_vsync,
            'lickTimes_0':times_lick0,
            'lickTimes_1':times_lick1}       
    print 'pkl vsyncs:',pkl['vsynccount']
#            if self.expt_id is '161012-M253178':
#            sync['visualFrames']['timestamps'] = sync['visualFrames']['timestamps'][1:] #hack for problematic first blip
    return sync 
    
# In[ ]:    

def get_traces(expt_dir,trace_file,tech=True):
    trace_path = os.path.join(expt_dir,trace_file)
    print 'loading '+trace_path
    with h5py.File(trace_path,"r") as f:
        traces = f["data"][:] #why do i need to do this? [:]?
        f.close()
    traces=np.asarray(traces)
    if not tech: 
        traces = np.swapaxes(traces,0,1)
    print traces.shape
    return traces
    
# In[ ]:
    
def get_pkl_path(expt_dir):
    mouseID = expt_dir.split('\\')[-2][-7:]
    pkl_file = [file for file in os.listdir(expt_dir) if file.endswith(mouseID+'.pkl')]
    pkl_path = os.path.join(expt_dir,pkl_file[0])   
    return pkl_path 
    
# In[ ]:
    
def get_pkl(expt_dir):
    mouseID = expt_dir.split('\\')[-2][-7:]
    pkl_file = [file for file in os.listdir(expt_dir) if file.endswith(mouseID+'.pkl')]
    pkl_path = os.path.join(expt_dir,pkl_file[0])   
#    print 'loading '+pkl_path
    with open(pkl_path,"rb+") as f:
        pkl = pickle.load(f)
    f.close()
    return pkl  
    
# In[ ]:
    
def get_dphys(expt_dir):
    dphys_file = [file for file in os.listdir(expt_dir) if file.endswith('.dphys')]
    dphys_path = os.path.join(expt_dir,dphys_file[0])
    print 'loading '+dphys_path
    rdphys = ird.importRawDPhys(dphys_path)
    dphys = ird.get_DPhysDict(rdphys)
    return dphys
    
# In[ ]:
    
def get_sync_path(expt_dir):
    expt_id = expt_dir.split('\\')[-2]
    sync_file = [file for file in os.listdir(expt_dir) if file.startswith(expt_id+'-') and file.endswith('.h5')]
    sync_path = os.path.join(expt_dir,sync_file[0])
    return sync_path
    
# In[ ]:    

def get_dphys_from_sync(expt_dir):
    
    pkl = get_pkl(expt_dir)
    
    sync_path = get_sync_path(expt_dir)
    
    d = Dataset(sync_path)
    meta_data = d.meta_data
    sample_freq = meta_data['ni_daq']['counter_output_freq']
    # 2P vsyncs
    vs2p_r = d.get_rising_edges('vsync_2p')
    # vs2p_f = d.get_falling_edges('vsync_2p')
    #Convert to seconds
    vs2p_rsec = vs2p_r/sample_freq
    # vs2p_fsec = vs2p_f/sample_freq
    frames_2p = vs2p_rsec
    print("Total 2P frames: %s" % len(frames_2p))
    # stimulus vsyncs
    vs_r = d.get_rising_edges('vsync_stim')
    vs_f = d.get_falling_edges('vsync_stim')
    # convert to seconds
    vs_r_sec = vs_r/sample_freq
    vs_f_sec = vs_f/sample_freq
    print("Detected vsyncs: %s" % len(vs_f_sec))
    # filter out spurious, transient blips in signal
    if len(vs_r_sec) != pkl['vsynccount']:
#    if len(vs_r_sec) > len(vs_f_sec):
#        vs_r_sec = vs_r_sec[:len(vs_f_sec)]
#    if len(vs_f_sec) > len(vs_r_sec):
#        vs_f_sec = vs_f_sec[:len(vs_r_sec)]
        vs_r_sec_f, vs_f_sec_f = filter_digital(vs_r_sec, vs_f_sec)
        print 'Spurious vsyncs:',str(len(vs_f_sec)-len(vs_f_sec_f))
        # remove first spike when DAQ is initialized, if there is one
    else: 
        vs_r_sec_f = vs_r_sec
        vs_f_sec_f = vs_f_sec
    if vs_r_sec_f[1] - vs_r_sec_f[0] > 0.2:
        vsyncs = vs_f_sec_f[1:]
    else:
        vsyncs = vs_f_sec_f
    print("Actual vsync frames: %s" % len(vsyncs))    
    #make fake dphys
    times_2p = {'timestamps':frames_2p}
    times_vsync = {'timestamps':vsyncs}
    dphys = {'2PFrames':times_2p,
            'visualFrames':times_vsync}
    print 'pkl vsyncs:',pkl['vsynccount']
            
    return dphys    
    
# In[ ]:
    
def get_session(expt_dir):
    mouseID = expt_dir.split('\\')[-2][-7:]
    pkl_file = [file for file in os.listdir(expt_dir) if file.endswith(mouseID+'.pkl')]
    pkl_path = os.path.join(expt_dir,pkl_file[0])  
    session = ib.load_session_from_behavioral_log_file(pkl_path)
    session = pa.addAllToDF(session)
    return session    
  
# In[ ]:
  
def get_save_dir(expt_dir,trace_file):
    save_dir = os.path.join(expt_dir,'analysis')
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir,trace_file[:-3])
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    return save_dir 
    
# In[ ]:
  
def get_fig_save_dir(save_dir):
    fig_save_dir = os.path.join(save_dir)
    if not os.path.exists(fig_save_dir):
        os.mkdir(fig_save_dir)
    return fig_save_dir 
    
# In[ ]:
    
def get_df_file_name(expt_dir):
    expt_name = expt_dir.split('\\')[-1]
    session_name = expt_dir.split('\\')[-2]
#    mouseID = expt_dir.split('\\')[-2][-7:]
    df_file_name = session_name+'-'+expt_name+'_df.pkl'
    return df_file_name
  
# In[ ]:

def load_df(df_path): 
    with open(df_path,"rb+") as f:
        df = pickle.load(f)
    f.close()  
    return df
    

# In[18]:


def get_max_image(expt_dir):  
    
    if os.path.exists(os.path.join(expt_dir,'max_image.png')):
        maxInt_path = os.path.join(expt_dir,'max_image.png')
    elif os.path.exists(os.path.join(expt_dir,'maxInt_a13.png')):
        maxInt_path = os.path.join(expt_dir,'maxInt_a13.png')
    elif os.path.exists(os.path.join(expt_dir,'max_downsample_4Hz_0.tiff')):
        maxInt_path = os.path.join(expt_dir,'max_downsample_4Hz_0.tiff')    
    max_image = mpimg.imread(maxInt_path)
    return max_image
  
# In[ ]:
  
def plot_traces(traces,traces_list):
    fig,ax = plt.subplots(len(traces_list),1,figsize=(20,15))
    ax = ax.ravel()
    for i,roi in enumerate(traces_list): 
        ax[i].plot(traces[roi,:])
#         ax[i].axis('off')
        ax[i].set_ylabel('roi '+str(i))
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_axis_bgcolor('white')
        ax[i].set_xlim([0,traces.shape[1]])
        ax[i].set_ylabel(str(roi))

# In[106]:

def plot_traces_np(traces,traces_list):
    fig,ax = plt.subplots(len(traces_list),1,figsize=(20,15))
    ax = ax.ravel()
    for i,roi in enumerate(traces_list): 
        ax[i].plot(traces[0,roi,:])
#         ax[i].axis('off')
        ax[i].set_ylabel('roi '+str(i))
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_axis_bgcolor('white')
        ax[i].set_xlim([0,traces.shape[2]])
        ax[i].set_ylabel(str(roi))

# In[106]:

def save_traces(traces,fig_save_dir):
    fig_dir = os.path.join(fig_save_dir,'traces')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    for n in range (0,traces.shape[0]):
        fig = plt.figure(figsize=(20,8))
        plt.plot(traces[n,:])
        plt.xlabel('frames',fontsize=18)
        plt.ylabel('fluorescence',fontsize=18)
        plt.title('ROI '+str(n),fontsize=18)
        fig.savefig(os.path.join(fig_dir,'roi_'+str(n)+'.png'), orientation='landscape')
    #        plt.show()
        plt.close(fig)

# In[ ]:

def save_traces_np(traces,fig_save_dir):
    fig_dir = os.path.join(fig_save_dir,'np_subtraction')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    for n in range (0,traces.shape[1]):
        fig = plt.figure(figsize=(20,8))
        plt.plot(traces[0,n,:],label='soma')
        plt.plot(traces[1,n,:],label='neuropil')
        plt.xlabel('frames',fontsize=18)
        plt.ylabel('fluorescence',fontsize=18)
        plt.title('ROI '+str(n),fontsize=18)
        plt.legend()
        fig.savefig(os.path.join(fig_dir,'roi_'+str(n)+'.png'), orientation='landscape')
        plt.close(fig)

# In[ ]:

def save_traces_df(traces,fig_save_dir): 
    fig_dir = os.path.join(fig_save_dir,'traces_df')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    df_traces = np.empty([traces.shape[0],traces.shape[1]])
    for n in range (0,traces.shape[0]):
        trace = traces[n,:]
        df,baseline = er.subtract_baseline(trace)
        df_traces[n,:] = df
        fig = plt.figure(figsize=(20,8))
        plt.plot(df_traces[n,:])
        plt.xlabel('frames',fontsize=18)
        plt.ylabel('dF',fontsize=18)
        plt.title('ROI '+str(n),fontsize=18)
        fig.savefig(os.path.join(fig_dir,'roi_'+str(n)+'.png'), orientation='landscape')
        plt.close(fig)
    
    return df_traces

# In[ ]:

def save_traces_dff(traces,fig_save_dir,method='percentile'): 
    fig_dir = os.path.join(fig_save_dir,'traces_dff')
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    dff_traces = np.empty([traces.shape[0],traces.shape[1]])
    for n in range (0,traces.shape[0]):
        trace = traces[n,:]
        dff,baseline = er.normalize(trace,method=method)
        dff_traces[n,:] = dff
        fig = plt.figure(figsize=(20,8))
        plt.plot(dff_traces[n,:])
        plt.xlabel('frames',fontsize=18)
        plt.ylabel('dFF',fontsize=18)
        plt.title('ROI '+str(n),fontsize=18)
        fig.savefig(os.path.join(fig_dir,'roi_'+str(n)+'.png'), orientation='landscape')
        plt.close(fig)
    
    return dff_traces
       
# In[9]:
        
# plot raw traces
def save_raw_traces(traces,fig_save_dir):
    for cell in range(0,traces.shape[0]):
        figsize=(15,5)
        fig,ax=plt.subplots(figsize=figsize)
        trace = traces[cell,:]
        ax.plot(trace)
        ax.set_xlim(0,traces.shape[1])
        ax.set_title('roi '+str(cell))
        ax.set_ylabel('F')
        ax.set_xlabel('frames')
        fig.tight_layout()
        fig_folder = 'traces'
        fig_title = 'roi_'+str(cell)
        fig_dir = os.path.join(fig_save_dir,fig_folder)
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        saveFigure(fig,os.path.join(fig_dir,fig_title),formats = ['.png'],size=figsize)
        plt.close()
    

## In[9]:
#
#def get_stim_df(pkl,dphys):   
#    stim_order = [(image_item[0]) for image_item in pkl["terrainlog"]] #only 2 images
#    stim_frames = [frame_item[1] for frame_item in pkl["laps"]]
##    stim_frames.insert(0,0.0) #first stim occurs frame 0, this is a bug we need to fix
#    stim_frames_t = []
#    try:
#        for frame_idx,frame_item in enumerate(stim_frames):
#            stim_frames_t.append((frame_item,dphys["visualFrames"]["timestamps"][frame_item]))#hack - add one because first vsync comes from initializing stim
#    except IndexError:
#        pass
#    stim_df_temp  = []
#    trial = 0
#    for (image),(frame,time) in zip(stim_order,stim_frames_t):
#        stim_df_temp.append([trial,time,frame,image])
#        trial = trial+1
#    stim_df = pd.DataFrame(stim_df_temp, columns=['trial','time', 'frame', 'image'])
#    return stim_df
#
## In[11]:
#
#def get_sweep_stim_df(pkl,dphys):
#    stim_order = pkl['bgsweeporder']
#    stim_frames = [frame_item[0] for frame_item in pkl['bgsweepframes']]
##    stim_frames.insert(0,0.0) #first stim occurs frame 0, this is a bug we need to fix
#    # dont want to do this for SweepStim because there is a delay before trials start, they dont start on frame 0
#    stim_frames_t = []
#    try:
#        for frame_idx,frame_item in enumerate(stim_frames):
#            stim_frames_t.append((frame_item,dphys["visualFrames"]["timestamps"][frame_item]))#hack - add one because first vsync comes from initializing stim
#    except IndexError:
#        pass
#    stim_df_temp  = []
#    trial = 0
#    for (code),(frame,time) in zip(stim_order,stim_frames_t):
#        stim_df_temp.append([trial,time,frame,code])
#        trial = trial+1
#    stim_df = pd.DataFrame(stim_df_temp, columns=['trial','time', 'frame', 'stim_code'])
#    return stim_df
#
## In[11]:
#
#def get_stim_codes_DoC(pkl):
#    stim_codes_list = []
#    i = 0
#    for image_name in pkl['image_names']:
#        for size in pkl['sizes']:
#            for position in pkl['positions']:
#                for ori in pkl['oris']:
#                    for contrast in pkl['contrasts']:
#                        stim_codes_list.append([i,image_name,size,position,ori,contrast])        
#                        i +=1 
#    stim_codes = pd.DataFrame(stim_codes_list,columns=['stim_code','image_name','image_size','position','ori','contrast'])
#    return stim_codes
#
#
## In[11]:
##
##def get_response_type_DoC(pkl_df,global_trial):
##    response_latency= pkl_df.iloc[global_trial].response_latency
##    if (pkl_df.iloc[global_trial].rewarded == True) & (np.isnan(response_latency) == False) & (response_latency != np.inf):
##        response = 'HIT'
##    elif (pkl_df.iloc[global_trial].rewarded == True) & ((np.isnan(response_latency) == True) or (response_latency == np.inf)):
##        response = 'MISS'
##    elif (pkl_df.iloc[global_trial].rewarded == False) & (np.isnan(response_latency) == False) & (response_latency != np.inf):
##        response = 'FA'
##    elif (pkl_df.iloc[global_trial].rewarded == False) & ((np.isnan(response_latency) == True) or (response_latency == np.inf)):
##        response = 'CR'
##    return response
#  
## In[11]:
#  
#def get_response_type_DoC(pkl,pkl_df,global_trial):
#    response_latency= pkl_df.iloc[global_trial].response_latency
#    # response = True if response latency is within the response_window
#    if (response_latency > pkl['response_window'][0]) & (response_latency <= pkl['response_window'][1]):
#        response = True
#    else: 
#        response = False
#    #determine trial type
#    if (pkl_df.iloc[global_trial].rewarded == True) & (response == True):
#        response_type = 'HIT'
#    elif (pkl_df.iloc[global_trial].rewarded == True) & (response == False):
#        response_type = 'MISS'
#    elif (pkl_df.iloc[global_trial].rewarded == False) & (response == True):
#        response_type = 'FA'
#    elif (pkl_df.iloc[global_trial].rewarded == False) & (response == False):
#        response_type = 'CR'
#    return response_type
#
#
## In[11]:
#
#def get_stim_df_DoC(pkl,pkl_df,stim_codes,sync):
#    df = pkl_df
#    sdf = stim_codes
#    change_list = []
#    i = 0
#    for trial in df.index:
#        if df.change_image[trial] is not None: #if its a change trial
#            initial_code = sdf[(sdf.image_name==df.initial_image[trial])&(sdf.image_size==df.initial_size[trial])&(sdf.ori==df.initial_ori[trial])].stim_code.values[0]
#            change_code = sdf[(sdf.image_name==df.change_image[trial])&(sdf.image_size==df.change_size[trial])&(sdf.ori==df.change_ori[trial])].stim_code.values[0]
#            change_frame = df.change_frame[trial]
#            change_time = sync["visualFrames"]["timestamps"][change_frame]
#            trial_type = df.loc[trial].trial_type
#            response = df.loc[trial].response
#            response_type = df.loc[trial].response_type
#            change_list.append([i,trial,change_frame,change_time,initial_code,change_code,trial_type,response,response_type])
#            i+=1
#    stim_df = pd.DataFrame(change_list,columns=['trial','global_trial','frame','time','initial_code','change_code','trial_type','response','response_type'])
#    return stim_df
#    
    
# In[11]:

def get_2pframe_nearest(time_point,timestamps):
    #near_val = np.nanmin(p2_timestamps - time_point)+time_point
    #return np.where(p2_timestamps==near_val)[0][0],near_val
    return np.nanargmin(abs(timestamps - time_point))


# In[12]:

def get_snapshot(timepoint,trace,timestamps,snapshot_bounds=[-2,6],inclusive_upp=True):
    lo_bound,upp_bound = [timepoint+bound for bound in snapshot_bounds]
    lo_idx = get_2pframe_nearest(lo_bound,timestamps)
    upp_idx = get_2pframe_nearest(upp_bound,timestamps)
    return timestamps[lo_idx:upp_idx+inclusive_upp],trace[lo_idx:upp_idx+inclusive_upp]

# In[13]:

def get_snapshot_from_stack(timepoint,stack,timestamps,snapshot_bounds=[-1,1],inclusive_upp=True):
    lo_bound,upp_bound = [timepoint+bound for bound in snapshot_bounds]
    lo_idx = get_2pframe_nearest(lo_bound,timestamps)
    upp_idx = get_2pframe_nearest(upp_bound,timestamps)
    return timestamps[lo_idx:upp_idx+inclusive_upp],stack[lo_idx:upp_idx+inclusive_upp,:,:]

# In[13]:

def get_cell_response_dict(traces,stim_df,dphys,window=[-2,6]):
    cell_response_dict = {idx:None for idx in range(len(traces))}
    for roi,trace in enumerate(traces):
        timestamps = dphys["2PFrames"]["timestamps"]#[1:]
        snapshots = {}
        for trial in range(len(stim_df)):
            timepoint = stim_df[(stim_df.trial == trial)].time.values
            snapshots[trial] = get_snapshot(timepoint,trace,timestamps,snapshot_bounds=window) 
        cell_response_dict[roi] = snapshots
    return cell_response_dict

# In[15]:

def get_nearest_idx(time,visualFrame_times):
    try:
        vF = visualFrame_times - time #have to do this because we reuse this mutable object alot, if we -= we change it for future use!
        max_v = np.nanmax(vF) + 1
        vF[vF < 0] = max_v 
        nearest_idx = np.argmin(vF)
    except Exception as e:
        print e
        nearest_idx = None
    #print nearest_idx,len(visualFrame_times)
    return nearest_idx

# In[16]:

def get_lo_hi(array):
    try:
        vals = array[0],array[-1]
    except IndexError as e:
        vals = array,array
        print vals
    return vals

# In[17]:

#chris version
#def get_running_snaps(session,timestamps,inclusive=True):
#    large_value  = 1000000000
#    lo_time, upp_time = get_lo_hi(timestamps)
#    min_idx = get_nearest_idx(lo_time,timestamps)
#    max_idx = get_nearest_idx(upp_time,timestamps)
#    __,_,running = session.timeline.get_inds_times_values('running_speed_cm_per_sec',0,large_value)
#    return timestamps[min_idx:max_idx+inclusive],running[min_idx:max_idx+inclusive]

#marina version
def get_running_snapshot(session,timestamps):
    ind_range,run_times,run_speed = session.timeline.get_inds_times_values('running_speed_cm_per_sec',timestamps[0],timestamps[-1])
    return run_times, run_speed

# In[21]:

def get_final_df(cell_response_dict,session,stim_df):
    # assumes window used for cell_response_dict creation is always -2sec before stimulus to 4sec after stimulus
    new_df_list = []
    normd_low_T = -2
    columns = ["cell","trial","image",
               "v_frame","v_frame_time","response_timestamps",
               "responses","window_avg",
              "correct_line","choice","correct_response","bias_correction","behavioralreport",
              "run_time","run_speed","avg_run_speed"]
    # for cell,snapshots in cell_response_dict.iteritems():
    for cell in cell_response_dict.keys():
#        print 'cell '+str(cell)
        for trial in cell_response_dict[cell].keys():
            snapshot = cell_response_dict[cell][trial]
            tt = snapshot[0][:239]
            rr = snapshot[1][:239]
            new_df_part = [None for col in columns]
            new_df_part[columns.index("response_timestamps")] = tt
            new_df_part[columns.index("responses")] = rr

            run_time,run_speed = get_running_snapshot(session,tt)
            new_df_part[columns.index("run_time")] = run_time
            new_df_part[columns.index("run_speed")] = run_speed
            new_df_part[columns.index("avg_run_speed")] = np.nanmean(run_speed[60:90])#run speed from stim onset to 1sec after

            new_df_part[columns.index("response_timestamps")] = np.append(tt,[np.nan]*(152-len(tt)))
            new_df_part[columns.index("responses")] = np.append(rr,[np.nan]*(152-len(rr)))

#            run_time,run_speed = get_running_snaps(session,tt)
#            new_df_part[columns.index("run_time")] = np.append(run_time,[np.nan]*(152-len(run_time)))
#            new_df_part[columns.index("run_speed")] = np.append(run_speed,[np.nan]*(152-len(run_speed)))
#            new_df_part[columns.index("avg_run_speed")] = np.nanmean(run_speed[:120])#run speed from 2sec before stim to 2sec after
#            
            new_df_part[columns.index("cell")] = cell
            new_df_part[columns.index("trial")] = stim_df.trial[trial]
            new_df_part[columns.index("image")] = stim_df.image[trial]
            new_df_part[columns.index("v_frame")] = stim_df.frame[trial]
            new_df_part[columns.index("v_frame_time")] = stim_df.time[trial]
            try:
                #gets indices for time 0 to 1sec after stimulus onset
                temp_tt = tt-tt[0]+(normd_low_T)
                temp_tt_lo = np.where((temp_tt) > 0)[0][0]
                temp_tt_hi = np.where((temp_tt) <= 1)[0][-1]+1 #+1 so its inclusive
                #print temp_tt_lo,temp_tt_hi
                window_avg = np.nanmean(rr[temp_tt_lo:temp_tt_hi])
            except:
                window_avg = None
            new_df_part[columns.index("window_avg")] = window_avg #avg between t=0, t=1s after stim onset

            df_tmp = session.df[:len(stim_df)]
#            df_tmp = session.df
            new_df_part[columns.index("correct_line")] = df_tmp.correct_line.values[trial]
            new_df_part[columns.index("choice")] = df_tmp.choice.values[trial]
            new_df_part[columns.index("correct_response")] = df_tmp.correct_response.values[trial]
#            new_df_part[columns.index("bias_correction")] = df_tmp.bias_correction.values[trial]
            new_df_part[columns.index("behavioralreport")] = df_tmp.behavioralreport.values[trial]

            new_df_list.append(new_df_part)
    final_df = pd.DataFrame(new_df_list, columns=columns)
    
    return final_df
    
# In[24]:    
    
def get_final_df_sweep_stim(cell_response_dict,stim_df,pkl,session,window=[-1,2]):
    new_df_list = []
    normd_low_T = -1
    columns = ["cell","trial","stim_code"]
    columns = np.hstack((columns,pkl['bgdimnames']))
    columns = np.hstack((columns,["v_frame","v_frame_time","response_timestamps",
               "responses","window_avg","run_time","run_speed","avg_run_speed"]))
    columns = columns.tolist()
    frames_in_window = (window[1]-window[0])*30#fps
    bgsweeptable = pkl['bgsweeptable']
    bgSweep = pkl['bgSweep']
    for cell in cell_response_dict.keys():
        for trial in cell_response_dict[cell].keys():
            snapshot = cell_response_dict[cell][trial]
            tt = snapshot[0]
            rr = snapshot[1]
            new_df_part = [None for col in columns]
            new_df_part[columns.index("response_timestamps")] = tt[:frames_in_window]
            new_df_part[columns.index("responses")] = rr[:frames_in_window]
            
            if 'running_speed_cm_per_sec' in session.timeline.values:
                run_time,run_speed = get_running_snapshot(session,tt[:frames_in_window])
                new_df_part[columns.index("run_time")] = run_time
                new_df_part[columns.index("run_speed")] = run_speed
                new_df_part[columns.index("avg_run_speed")] = np.nanmean(run_speed[60:90])#run speed from stim onset to 1sec after
            else: 
                new_df_part[columns.index("run_time")] = None
                new_df_part[columns.index("run_speed")] = None
                new_df_part[columns.index("avg_run_speed")] = None

            new_df_part[columns.index("cell")] = cell
            new_df_part[columns.index("trial")] = stim_df.trial[trial]
            new_df_part[columns.index("stim_code")] = stim_df.stim_code[trial]
            stim_code = stim_df.stim_code[trial]
            if stim_code == -1:
                for name in pkl['bgdimnames']: 
                    new_df_part[columns.index(name)] = None
            else:
                for name in pkl['bgdimnames']: 
                    if 'Image' in name: 
                        new_df_part[columns.index(name)] = bgsweeptable[stim_code][bgSweep['Image'][1]].split('\\')[-1:][0]
                    else: 
                        new_df_part[columns.index(name)] = bgsweeptable[stim_code][bgSweep[name][1]]

            new_df_part[columns.index("v_frame")] = stim_df.frame[trial]
            new_df_part[columns.index("v_frame_time")] = stim_df.time[trial]
            try:
                temp_tt = tt-tt[0]+(normd_low_T)
                temp_tt_lo = np.where((temp_tt) > 0)[0][0]
                temp_tt_hi = np.where((temp_tt) <= 0.5)[0][-1]+1 #+1 so its inclusive
                #print temp_tt_lo,temp_tt_hi
                window_avg = np.nanmean(rr[temp_tt_lo:temp_tt_hi])
            except:
                window_avg = None
            new_df_part[columns.index("window_avg")] = window_avg

            new_df_list.append(new_df_part)
    final_df = pd.DataFrame(new_df_list, columns=columns)
    return final_df
    
# In[24]:
    
    
def get_final_df_DoC(cell_response_dict,stim_df,session,window=[-2,4]):
    new_df_list = []
    normd_low_T = -1
    columns = ["cell","trial","global_trial","initial_code","change_code","trial_type","response","response_type"]
    columns = np.hstack((columns,["v_frame","v_frame_time","response_timestamps",
               "responses","window_avg","run_time","run_speed","avg_run_speed"]))
    columns = columns.tolist()
    frames_in_window = (window[1]-window[0])*30#fps
    frames_in_run_window = (((window[1])-(window[0]))*60)-30
    mean_window = [np.abs(window[0]),np.abs(window[0])+1]
    for cell in cell_response_dict.keys():
        for trial in cell_response_dict[cell].keys():
            snapshot = cell_response_dict[cell][trial]
            tt = snapshot[0]
            rr = snapshot[1]
            new_df_part = [None for col in columns]
            new_df_part[columns.index("response_timestamps")] = tt[:frames_in_window]
            new_df_part[columns.index("responses")] = rr[:frames_in_window]

            if 'running_speed_cm_per_sec' in session.timeline.values:
                run_time,run_speed = get_running_snapshot(session,tt)
                new_df_part[columns.index("run_time")] = run_time[:frames_in_run_window]
                new_df_part[columns.index("run_speed")] = run_speed[:frames_in_run_window]
                new_df_part[columns.index("avg_run_speed")] = np.nanmean(run_speed[mean_window[0]*30:mean_window[1]*30])#run speed from stim onset to 1sec after
            else: 
                new_df_part[columns.index("run_time")] = None
                new_df_part[columns.index("run_speed")] = None
                new_df_part[columns.index("avg_run_speed")] = None

            new_df_part[columns.index("cell")] = cell
            new_df_part[columns.index("trial")] = stim_df.trial[trial]
            new_df_part[columns.index("global_trial")] = stim_df.global_trial[trial]
            new_df_part[columns.index("initial_code")] = stim_df.initial_code[trial]
            new_df_part[columns.index("change_code")] = stim_df.change_code[trial]
            new_df_part[columns.index("trial_type")] = stim_df.trial_type[trial]
            new_df_part[columns.index("response")] = stim_df.response[trial]
            new_df_part[columns.index("response_type")] = stim_df.response_type[trial]
            new_df_part[columns.index("v_frame")] = stim_df.frame[trial]
            new_df_part[columns.index("v_frame_time")] = stim_df.time[trial]
            try:
                temp_tt = tt-tt[0]+(normd_low_T)
                temp_tt_lo = np.where((temp_tt) > 0)[0][0]
                temp_tt_hi = np.where((temp_tt) <= 0.5)[0][-1]+1 #+1 so its inclusive
                window_avg = np.nanmean(rr[temp_tt_lo:temp_tt_hi])
            except:
                window_avg = None
            new_df_part[columns.index("window_avg")] = window_avg

            new_df_list.append(new_df_part)
    final_df = pd.DataFrame(new_df_list, columns=columns)
    return final_df

    
# In[24]:
    
def get_mean_in_window(trace,timestamps,window=[1,2],get_sem=True): 
    idx0 = np.where(timestamps>=window[0])
    idx1 = np.where(timestamps>=window[1])
    segment = trace[idx0[0][0]:idx1[0][0]]
    mean = np.mean(segment)
    if get_sem: 
        sem = np.std(segment)/np.sqrt(len(segment))
    else: 
        sem = np.std(segment)
    return mean,sem
    
# In[24]:

def get_max_in_window(trace,timestamps,window=[1,2]): 
    idx0 = np.where(timestamps>=window[0])
    idx1 = np.where(timestamps>=window[1])
    segment = trace[idx0[0][0]:idx1[0][0]]
    peak = np.amax(segment)
    return peak
    
# In[24]:

def add_responses_dF(df,window=[1,2]):
    response_df_list = []
    response_dff_list = []
    for cell in df.cell.unique():
        for trial in df.trial.unique():
            trace = df[(df.cell==cell)&(df.trial==trial)].responses.values[0]
            times = df[(df.cell==cell)&(df.trial==trial)].response_timestamps.values[0]
            timestamps = times - times[0]
            baseline,sem = get_mean_in_window(trace,timestamps,window=window)
            trace_df = trace-baseline
            trace_dff = trace_df/baseline
            response_df_list.append(trace_df)
            response_dff_list.append(trace_dff)
    df['responses_dF'] = response_df_list
    df['responses_dFF'] = response_dff_list
    return df   

# In[24]:

def add_mean_response(df,methods=['dF'],window=[2,3]):
    # get mean and peak values for 1 second window after stimulus onset
    for method in methods: 
        mean_list = []
        peak_list = []
        sd_list = []
        for cell in df.cell.unique():
            for trial in df.trial.unique(): 
                if 'd' in method:
                    trace = df[(df.cell==cell)&(df.trial==trial)]['responses_'+method].values[0]
                else:
                    trace = df[(df.cell==cell)&(df.trial==trial)]['responses'].values[0]
                timestamps = df[(df.cell==cell)&(df.trial==trial)].response_timestamps.values[0]
                timestamps = timestamps - timestamps[0]
                mean,sem = get_mean_in_window(trace,timestamps,window=window)
                peak = get_max_in_window(trace,timestamps,window=window)  
                mean_list.append(mean)
                peak_list.append(peak)
                sd_list.append(sem)
        df['mean_onset_'+method] = mean_list
        df['sem_onset_'+method] = sd_list
        df['peak_onset_'+method] = peak_list 
    return df
    
# In[24]:
    
def add_mean_response_offset(df,methods=['dF'],window=[3,4]):
    # get mean and peak values for 1 second window after stimulus offset
    for method in methods: 
        mean_list = []
        peak_list = []
        sem_list = []
        for cell in df.cell.unique():
            for trial in df.trial.unique(): 
                trace = df[(df.cell==cell)&(df.trial==trial)]['responses_'+method].values[0]
                timestamps = df[(df.cell==cell)&(df.trial==trial)].response_timestamps.values[0]
                timestamps = timestamps - timestamps[0]
                mean,sem = get_mean_in_window(trace,timestamps,window=window)
                peak = get_max_in_window(trace,timestamps,window=window)  
                mean_list.append(mean)
                peak_list.append(peak)
                sem_list.append(sem)
        df['mean_offset_'+method] = mean_list
        df['peak_offset_'+method] = peak_list 
        df['sem_offset_'+method] = sem_list
    return df
    
# In[24]:

def plot_error_rate(traces,expt_dir):
    data = traces
    expt_name = expt_dir.split('\\')[-1]
    #detect events and aggregate data
    event_data = er.get_event_data(data)
    event_data = er.add_expt_info(event_data,expt_dir,expt_name) 
    event_data = er.add_error_rate_to_dict(event_data)
    save_path = os.path.join(event_data[0]['expt_dir'],'motion_analysis')
    if not os.path.exists(save_path): 
        os.mkdir(save_path)
    #plot histogram of positive and negative events
    er.plot_event_hist(event_data,save=True)
    #generate error rate plus mask plus trace figure for each ROI
    er.plot_error_rate_by_roi(event_data,expt_dir)
    #plot error rate across all ROIs in the dataset
    er.plot_error_rate_avg(event_data,save=True)
    #plot event rate across all ROIs
    er.plot_event_rate_hist(event_data,save=True)
       
# In[18]:

def add_mean_sd(df,methods=['dF'],period='baseline',window=[1,2]):
    # get mean and peak values for 2 second window after stimulus onset
    for method in methods: 
        mean_list = []
        sd_list = []
        for cell in df.cell.unique():
            for trial in df.trial.unique(): 
                if method == None: 
                    trace = df[(df.cell==cell)&(df.trial==trial)]['responses'].values[0]
                else:
                    trace = df[(df.cell==cell)&(df.trial==trial)]['responses_'+method].values[0]
                timestamps = df[(df.cell==cell)&(df.trial==trial)].response_timestamps.values[0]
                timestamps = timestamps - timestamps[0]
                mean,sd = get_mean_in_window(trace,timestamps,window=window,get_sem=False)
                mean_list.append(mean)
                sd_list.append(sd)
        if method == None: 
            df['mean_'+period] = mean_list
            df['sd_'+period] = sd_list 
        else:
            df['mean_'+period+'_'+method] = mean_list
            df['sd_'+period+'_'+method] = sd_list 
    return df

# In[18]:

def add_mean_sem(df,methods=['dF'],period='baseline',window=[1,2]):
    # get mean and peak values for 2 second window after stimulus onset
    for method in methods: 
        mean_list = []
        sem_list = []
        for cell in df.cell.unique():
            for trial in df.trial.unique(): 
                trace = df[(df.cell==cell)&(df.trial==trial)]['responses_'+method].values[0]
                timestamps = df[(df.cell==cell)&(df.trial==trial)].response_timestamps.values[0]
                timestamps = timestamps - timestamps[0]
                mean,sem = get_mean_in_window(trace,timestamps,window=window,get_sem=True)
                mean_list.append(mean)
                sem_list.append(sem)
        df['mean_'+period+'_'+method] = mean_list
        df['sem_'+period+'_'+method] = sem_list 
    return df
    
# In[18]:

def add_significance(df,factor=5,method='dFF',offset=False):
    sig_resp_list = []
    sig_offset_list = []
    for cell in df.cell.unique():
        for trial in df.trial.unique(): 
            sigma = df[(df.cell==cell)&(df.trial==trial)]['sd_baseline_'+method].values[0]
            sig_thresh = sigma*factor
            stim_mean = df[(df.cell==cell)&(df.trial==trial)]['mean_response_'+method].values[0]
    #         print sigma*5, sig_thresh, stim_mean, offset_mean
            if stim_mean >= sig_thresh: 
                sig_resp_list.append(True)
            else: 
                sig_resp_list.append(False)
            if offset:
                offset_mean = df[(df.cell==cell)&(df.trial==trial)]['mean_offset_'+method].values[0]
                if offset_mean >= sig_thresh: 
                    sig_offset_list.append(True)
                else: 
                    sig_offset_list.append(False)
    df['sig_response'] = sig_resp_list
    if offset: 
        df['sig_offset'] = sig_offset_list
    
    return df

# In[316]:

def p_val(x,mean_window):
    #get frame #s for one second before stim and one second after stim given window
    w = mean_window
    baseline_end = w[0]*30
    baseline_start = (w[0]-1)*30
    stim_start = w[0]*30
    stim_end = (w[0]+1)*30
    (_, p) = stats.f_oneway(x[baseline_start:baseline_end], x[stim_start:stim_end])
    return p
    
# In[279]:

def add_p_vals(df,mean_window):
#add p_vals to response dataframe
    p_list = []
    for row in range(len(df)):
        x = df.iloc[row].responses_dFF
        p = p_val(x,mean_window)
        p_list.append(p)
    df['p_val'] = p_list
    
    return df
    
# In[279]:
    
    # outdated cellSort traces analysis tools


def get_cellSort_masks(expt_dir,mask_file): 
    cs_edited = scipy.io.loadmat(os.path.join(expt_dir,mask_file))
    cellSort_mask = cs_edited['ica_segments']
    cellSort_segcentroids = cs_edited['segcentroid']
    return cellSort_mask, cellSort_segcentroids

def limit_SI(SI):
    for i,val in enumerate(SI): 
        if val < -1:
            SI[i] = -1
        elif val > 1:
            SI[i] = 1
    return SI

def get_binary_mask_2D(cellSort_mask,threshold):
    flat_binary_mask = np.zeros((cellSort_mask.shape[1],cellSort_mask.shape[2]))
    for i in range(0,cellSort_mask.shape[0]):
        ids = np.where(cellSort_mask[i,:,:]>=threshold)
        flat_binary_mask[ids]=1
    return flat_binary_mask
        
def get_binary_mask_3D(cellSort_mask,threshold):
    binary_mask = np.zeros((cellSort_mask.shape[0],cellSort_mask.shape[1],cellSort_mask.shape[2]))
    for roi in range(cellSort_mask.shape[0]):
        tmp = np.zeros((cellSort_mask.shape[1],cellSort_mask.shape[2]))
        ind=np.where(cellSort_mask[roi,:,:]>=threshold)
        tmp[ind]=1
        binary_mask[roi,:,:]=tmp
    return binary_mask
       
def get_metric_mask(binary_mask,metric,roi_list=None):
    # binary mask must be 3D
    # turn zero values into NaNs and positive values into their metric value
    metric_mask = np.zeros((binary_mask.shape[0],binary_mask.shape[1],binary_mask.shape[2]))
    if roi_list is None: 
        roi_list = range(0,binary_mask.shape[0])
    for i,roi in enumerate(roi_list):
        tmp_mask=np.zeros((binary_mask.shape[1],binary_mask.shape[2]))
        ind=np.where(binary_mask[roi,:,:]==0) 
        tmp_mask[ind]=np.nan
        ind=np.where(binary_mask[roi,:,:]>0)
        tmp_mask[ind]=metric[i]
        metric_mask[roi,:,:]=tmp_mask
    return metric_mask
    
def get_metric_mask_2D(label_mask,metric,n_rois,roi_list=None):
    # binary mask must be 2D
    # turn zero values into NaNs and positive values into their metric value
    metric_mask = np.zeros(label_mask.shape)
    metric_mask[:]= np.nan
    if roi_list is None: 
        roi_list = range(0,n_rois)
    for i,roi in enumerate(roi_list):
        metric_mask[label_mask==roi+1]=metric[i]
    return metric_mask

def plot_metric_mask(metric_mask,metric_name,fig_save_dir,fig_type,seg_type='tech',roi_list=None,max_image=None,cmap='jet',ax=None,save=False,label=False,labels=None):
    if ax is None: 
        fig,ax=plt.subplots()
    if max_image is not None: 
        ax.imshow(max_image,cmap='gray')
        alpha = 0.5
    else: 
        alpha = 1
    if seg_type == 'cs':
        if roi_list is None: 
            roi_list = range(0,metric_mask.shape[0])
        for roi in roi_list:
            cax = ax.imshow(metric_mask[roi,:,:],cmap=cmap,alpha=alpha,vmin=-1,vmax=1)
    elif seg_type == 'tech':
        cax = ax.imshow(metric_mask,cmap=cmap,alpha=alpha,vmin=-1,vmax=1)
    ax.grid(False)
    ax.set_title(metric_name)
    if label: 
         ax = label_rois(labels,ax=ax,text_color='y')
    plt.axis('off')
    #     ax.text(curatedROI['segcentroid'][roi][0], curatedROI['segcentroid'][roi][1], roi, fontsize=12,color='blue')
    if save:
        fig.colorbar(cax);
        fig.tight_layout()
        if max_image is not None: 
            fig_title = 'metric_mask_'+fig_type+'_maxInt'
        else:
            fig_title = 'metric_mask_'+fig_type
        if not os.path.exists(fig_save_dir):
            os.mkdir(fig_save_dir)
        saveFigure(fig,os.path.join(fig_save_dir,fig_title),formats = ['.png','.pdf'],size=(8,8))
        plt.close()
    return ax

def label_rois(cellSort_segcentroids,ax,text_color='b'):
    for roi in range(cellSort_segcentroids.shape[0]):
        ax.text(cellSort_segcentroids[roi][0], cellSort_segcentroids[roi][1], roi, fontsize=12,color=text_color)
    return ax
    
    
    

# In[279]:

#everything below replaced by ophys_dataset object 

if __name__ == '__main__':

    baseline_subtraction = False
    # set save path
    save_file = '150902-M182204-circleDiscrimination.pkl'
    save_dir = r'\\aibsdata2\nc-ophys\ImageData\Technology\Marina\150902-M182204\circleDiscrimination\analysis'
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir,save_file)
    # load pkl
    pkl_path = r"\\aibsdata2\nc-ophys\ImageData\Technology\Marina\150902-M182204\circleDiscrimination\150902184702-150902175220-M182204.pkl"
    with open(pkl_path,"rb+") as f:
        pkl = pickle.load(f)
    f.close()
    # load imaging_behavior session
    session = ib.load_session_from_behavioral_log_file(pkl_path)
    session = pa.addAllToDF(session)
    # load dphys
    dphys = r"C:\Data\Behavior + 2P Analysis\150902-M182204\circleDiscrimination\1441244832-1441241510M182204.dphys"
    rdphys = ird.importRawDPhys(dphys)
    dphys = ird.get_DPhysDict(rdphys)
    # load traces
    trace_path = r"\\aibsdata2\nc-ophys\ImageData\Technology\Marina\150902-M182204\circleDiscrimination\trace_31Hz_cellSort_dF.h5"
    with h5py.File(trace_path,"r") as f:
        traces = f["data"][:] #why do i need to do this? [:]?
    traces=np.asarray(traces)
    traces = np.swapaxes(traces,0,1)
    # set figure save path
    fig_save_dir = os.path.join(save_dir,'figures')
    if not os.path.exists(fig_save_dir):
        os.mkdir(fig_save_dir)
    # plot raw traces
    plot_traces(traces,fig_save_dir)
#    # plot baseline subtracted traces
    df_traces = save_traces_df(traces,fig_save_dir)
    # if baseline subtraction, use dF traces
#    if baseline_subtraction:
#        traces = df_traces
    # create stimulus trial dictionary
    stim_df = get_stim_df(pkl,dphys)
#    # create cell responses dictionary - response in every trial for each cell
    cell_response_dict = get_cell_response_dict(traces,stim_df,dphys)
#    # create final dictionary combining stimulus df, cell response df, and behavior session info 
    df = get_final_df(cell_response_dict,session,stim_df)
    # save final dataframe
#    df.to_pickle(save_path)
    # load final dataframe
    f = open(save_path,'rb')
    df = pickle.load(f)
#    # plot average response to each stimulus
#    op.plot_avg_responses(df,session,fig_save_dir)
#    # plot average response to each stimulus broken down by behavior choice (hit vs miss)
#    op.plot_response_by_choice(df,session,fig_save_dir)
#    # plot traces with behavior events overlay 
#    op.plot_trace_behavior_events(roi=38,traces=traces,xlims=[1700,2000],save=False)