# -*- coding: utf-8 -*-
"""
Created on Thursday September 7 11:39:00 2017

@author: marinag
"""
import os
import h5py
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.image as mpimg
from visual_behavior_ophys.dro import utilities as du
from visual_behavior_ophys.utilities.lims_database import LimsDatabase
from visual_behavior_ophys.roi_mask_analysis import roi_mask_analysis as rm


class VisualBehaviorScientificaDataset(object):
    def __init__(self, lims_id, filter_edge_cells=True, analysis_dir=None):
        """initialize visual behavior ophys experiment dataset

			Parameters
			----------
			expt_session_id : ophys experiment session ID
			mouse_id : 6-digit mouse ID
		"""
        self.lims_id = lims_id
        self.analysis_dir = analysis_dir
        self.filter_edge_cells = filter_edge_cells
        self.get_lims_data()
        self.get_directories()
        # self.ophys_session_dir = ophys_session_dir
        # self.session_id = ophys_session_dir.split('_')[-1]
        # self.experiment_id = int(
        #     [dir for dir in os.listdir(ophys_session_dir) if 'ophys_experiment' in dir][0].split('_')[-1])
        self.get_ophys_metadata()
        self.get_sync()
        self.get_pkl()
        self.get_pkl_df()
        self.get_stimulus_type()
        self.get_running_speed()
        self.get_stim_codes()
        self.get_stim_table()
        self.get_roi_metrics()
        self.get_roi_mask_array()
        self.get_motion_correction()
        self.get_max_projection()
        self.get_dff_traces()



    def get_lims_data(self):
        ld = LimsDatabase(self.lims_id)
        lims_data = ld.get_qc_param()
        mouse_id = lims_data.external_specimen_id.values[0]
        self.mouse_id = np.int(mouse_id)
        self.session_id = lims_data.session_id.values[0]
        self.ophys_session_dir = lims_data.datafolder.values[0][:-28]
        self.session_name = lims_data.experiment_name.values[0].split('_')[-1]
        self.experiment_id = lims_data.lims_id.values[0]
        lims_data.insert(loc=2, column='experiment_id', value=self.experiment_id)
        del lims_data['datafolder']
        self.lims_data = lims_data


    def get_directories(self):
        self.ophys_experiment_dir = os.path.join(self.ophys_session_dir, 'ophys_experiment_' + str(self.experiment_id))
        self.demix_dir = os.path.join(self.ophys_experiment_dir, 'demix')
        self.processed_dir = os.path.join(self.ophys_experiment_dir, 'processed')
        segmentation_folder = [dir for dir in os.listdir(self.processed_dir) if 'ophys_cell_segmentation' in dir][0]
        self.segmentation_dir = os.path.join(self.processed_dir, segmentation_folder)
        self.get_analysis_dir()

    def get_analysis_dir(self):
        if self.analysis_dir is None:
            analysis_dir = os.path.join(self.ophys_experiment_dir, 'analysis')
            if not os.path.exists(analysis_dir):
                os.mkdir(analysis_dir)
        else:
            l = self.lims_data
            folder_name = str(l.external_specimen_id.values[0]) + '_' + str(l.lims_id.values[0]) + '_' + \
                          l.structure.values[0] + '_' + str(l.depth.values[0]) + '_' + \
                          l.specimen_driver_line.values[0].split('-')[0] + '_' + self.session_name
            analysis_dir = os.path.join(self.analysis_dir, folder_name)
            if not os.path.exists(analysis_dir):
                os.mkdir(analysis_dir)
        self.analysis_dir = analysis_dir

    def get_ophys_metadata(self):
        metadata = {}
        metadata['ophys_frame_rate'] = 30.
        metadata['stimulus_frame_rate'] = 60.
        self.metadata = metadata
        return self.metadata

    def get_sync_path(self):
        sync_file = [file for file in os.listdir(self.ophys_session_dir) if 'sync' in file][0]
        sync_path = os.path.join(self.ophys_session_dir, sync_file)
        return sync_path

    def get_sync(self):
        from visual_behavior_ophys.temporal_alignment.sync_dataset import Dataset
        sync_path = self.get_sync_path()
        d = Dataset(sync_path)
        sync_data = d
        meta_data = d.meta_data
        sample_freq = meta_data['ni_daq']['counter_output_freq']
        # 2P vsyncs
        vs2p_r = d.get_rising_edges('2p_vsync')
        # Convert to seconds
        vs2p_rsec = vs2p_r / sample_freq
        frames_2p = vs2p_rsec
        # stimulus vsyncs
        vs_r = d.get_rising_edges('stim_vsync')
        vs_f = d.get_falling_edges('stim_vsync')
        # convert to seconds
        vs_r_sec = vs_r / sample_freq
        vs_f_sec = vs_f / sample_freq
        vsyncs = vs_f_sec
        # add lick data
        lick_1 = d.get_rising_edges('lick_1') / sample_freq
        trigger = d.get_rising_edges('2p_trigger') / sample_freq
        cam1_exposure = d.get_rising_edges('cam1_exposure') / sample_freq
        cam2_exposure = d.get_rising_edges('cam2_exposure') / sample_freq
        stim_photodiode = d.get_rising_edges('stim_photodiode') / sample_freq
        # some experiments have 2P frames prior to stimulus start - restrict to timestamps after trigger
        frames_2p = frames_2p[frames_2p > trigger[0]]
        print("Visual frames detected in sync: %s" % len(vsyncs))
        print("2P frames detected in sync: %s" % len(frames_2p))
        # put sync data in dphys format to be compatible with downstream analysis
        times_2p = {'timestamps': frames_2p}
        times_vsync = {'timestamps': vsyncs}
        times_lick_1 = {'timestamps': lick_1}
        times_trigger = {'timestamps': trigger}
        times_cam1_exposure = {'timestamps': cam1_exposure}
        times_cam2_exposure = {'timestamps': cam2_exposure}
        times_stim_photodiode = {'timestamps': stim_photodiode}
        sync = {'2PFrames': times_2p,
                'visualFrames': times_vsync,
                'lickTimes_0': times_lick_1,
                'cam1_exposure': times_cam1_exposure,
                'cam2_exposure': times_cam2_exposure,
                'stim_photodiode': times_stim_photodiode,
                'trigger': times_trigger,
                }
        self.sync = sync
        self.sync_data = sync_data
        return self.sync

    def get_pkl_path(self):
        pkl_file = [file for file in os.listdir(self.ophys_session_dir) if 'M' + str(self.mouse_id) + '.pkl' in file]
        if len(pkl_file) > 0:
            self.pkl_path = os.path.join(self.ophys_session_dir, pkl_file[0])
        else:
            self.expt_date = [file for file in os.listdir(self.ophys_session_dir) if 'sync' in file][0].split('_')[2][
                             2:8]
            print self.expt_date
            pkl_dir = os.path.join(r'\\allen\programs\braintv\workgroups\neuralcoding\Behavior\Data',
                                   'M' + str(self.mouse_id), 'output')
            pkl_file = [file for file in os.listdir(pkl_dir) if file.startswith(self.expt_date)][0]
            self.pkl_path = os.path.join(pkl_dir, pkl_file)
        return self.pkl_path

    def get_pkl(self):
        pkl_path = self.get_pkl_path()
        with open(pkl_path, "rb+") as f:
            pkl = pickle.load(f)
        f.close()
        self.pkl = pkl
        print 'visual frames in pkl file:', self.pkl['vsynccount']
        return self.pkl

    def get_pkl_df(self):
        self.pkl = pd.read_pickle(self.pkl_path)
        pkl_df = du.create_doc_dataframe(self.pkl_path)
        pkl_df = pkl_df.replace(to_replace=np.nan, value=0)  # replace NaNs with 0 in response column
        pkl_df = pkl_df.rename(columns={'change_image_name': 'change_image'})
        pkl_df = pkl_df.rename(columns={'initial_image_name': 'initial_image'})
        pkl_df['trial_num'] = 1
        pkl_df.insert(0, 'trial', np.arange(0, len(pkl_df)).tolist())
        self.pkl_df = pkl_df
        return self.pkl_df

    def get_stimulus_type(self):
        pkl = self.pkl
        if 'stimulus_type' in pkl:
            self.stimulus_type = pkl['stimulus_type']
            print 'stim type is ', self.stimulus_type
        else:
            print 'stimulus_type not specified in pkl'
        return self.stimulus_type

    def get_stimulus_timestamps(self):
        self.timestamps_stimulus = self.sync['visualFrames']['timestamps']
        return self.timestamps_stimulus

    def get_running_speed(self, smooth=True):
        from scipy.signal import medfilt
        dx = self.pkl['dx']
        dx_filt = medfilt(dx, kernel_size=5)  # remove big, single frame spikes in encoder values
        theta_raw = np.cumsum(np.array(dx_filt))  # wheel rotations
        time_array = np.hstack((0, np.cumsum(self.pkl['vsyncintervals']) / 1000.))  # vsync frames
        speed_rad_per_s = np.hstack((np.diff(theta_raw[:len(time_array)]) / np.mean(np.diff(time_array)), 0))
        wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter
        running_radius = 0.5 * (
            2.0 * wheel_diameter / 3.0)  # assume the animal runs at 2/3 the distance from the wheel center
        running_speed_cm_per_sec = np.pi * speed_rad_per_s * running_radius / 180.
        if smooth:
            running_speed_cm_per_sec = pd.rolling_mean(running_speed_cm_per_sec, window=6)
        self.running_speed = running_speed_cm_per_sec
        self.timestamps_stimulus = self.get_stimulus_timestamps()
        return self.running_speed, self.timestamps_stimulus

    def get_stim_codes(self):
        pkl = self.pkl
        stim_codes_list = []
        i = 0
        if 'image_dict' in pkl:
            for image_num in np.sort(pkl['image_dict'].keys()):
                for image_name in pkl['image_dict'][image_num].keys():
                    stim_codes_list.append([i, image_name, image_num])
                    i += 1
            stim_codes = pd.DataFrame(stim_codes_list, columns=['stim_code', 'image_name', 'image_num'])
        else:
            print 'pkl file does not contain image_dict'
        self.stim_codes = stim_codes
        return self.stim_codes

    def get_stim_table(self):
        sync = self.sync
        df = self.pkl_df.copy()
        sdf = self.stim_codes.copy()
        change_list = []
        i = 0
        for trial in df.index:
            if df.change_image[trial] != 0:  # if its a change trial
                initial_code = sdf[(sdf.image_name == df.initial_image[trial])].stim_code.values[0]
                change_code = sdf[(sdf.image_name == df.change_image[trial])].stim_code.values[0]
                initial_image = df.loc[trial].initial_image
                change_image = df.loc[trial].change_image
                change_frame = np.int(df.change_frame[trial])
                change_time = sync["visualFrames"]["timestamps"][change_frame]
                trial_type = df.loc[trial].trial_type
                response = df.loc[trial].response
                response_type = df.loc[trial].response_type
                change_list.append(
                    [i, trial, change_frame, change_time, initial_code, change_code, initial_image, change_image,
                     trial_type, response, response_type])
                i += 1
        self.stim_table = pd.DataFrame(change_list,
                                       columns=['change_trial', 'total_trial', 'change_frame', 'change_time',
                                                'initial_code', 'change_code', 'initial_image', 'change_image',
                                                'trial_type', 'behavioral_response', 'behavioral_response_type'])
        return self.stim_table

    def get_nan_trace_indices(self):
        dff_path = os.path.join(self.ophys_experiment_dir, str(self.experiment_id) + '_dff.h5')
        g = h5py.File(dff_path)
        dff_traces = np.asarray(g['data'])
        nan_trace_indices = []
        for i, trace in enumerate(dff_traces):
            if np.isnan(trace)[0]:
                nan_trace_indices.append(i)
                #     dataset.non_nan_inds = np.asarray([x for x in dataset.objectlist.index.values if x not in nan_trace_indices])
        return nan_trace_indices

    def get_edge_cell_indices(self):
        seg_folder = [file for file in os.listdir(self.processed_dir) if 'segmentation' in file][0]
        objectlist = pd.read_csv(os.path.join(self.processed_dir, seg_folder, 'objectlist.txt'))  # segmentation metrics
        edge_cell_indices = objectlist[objectlist[' traceindex'] == 999].index.values
        return edge_cell_indices

    def get_filtered_roi_indices(self):
        objectlist = self.objectlist
        edge_cell_indices = self.get_edge_cell_indices()
        nan_inds = self.get_nan_trace_indices()
        remove_inds = np.unique(np.hstack((edge_cell_indices, nan_inds)))
        self.filtered_roi_indices = objectlist[(objectlist.index.isin(remove_inds) == False)].index.values
        return self.filtered_roi_indices

    def filter_traces(self, traces):
        filtered_roi_indices = self.get_filtered_roi_indices()
        traces = traces[filtered_roi_indices]
        return traces

    def filter_roi_metrics(self):
        filtered_roi_indices = self.get_filtered_roi_indices()
        self.roi_metrics = self.objectlist[(self.objectlist.index.isin(filtered_roi_indices) == True)]
        self.roi_metrics = self.roi_metrics.reset_index()
        return self.roi_metrics

    def get_roi_metrics(self):
        # objectlist.txt contains metrics associated with segmentation masks
        seg_folder = [file for file in os.listdir(self.processed_dir) if 'segmentation' in file][0]
        objectlist = pd.read_csv(os.path.join(self.processed_dir, seg_folder, 'objectlist.txt'))  # segmentation metrics
        self.objectlist = objectlist
        self.roi_metrics = self.filter_roi_metrics()
        return self.roi_metrics

    def get_roi_mask_array(self):
        f = h5py.File(os.path.join(self.segmentation_dir, "maxInt_masks2.h5"))
        masks = np.asarray(f['data'])
        roi_dict = rm.make_roi_dict(self.roi_metrics, masks)
        f.close()
        self.roi_dict = roi_dict
        tmp = roi_dict[roi_dict.keys()[0]]
        roi_mask_array = np.empty((len(roi_dict.keys()), tmp.shape[0], tmp.shape[1]))
        for i, roi in enumerate(roi_dict.keys()):
            roi_mask_array[i] = roi_dict[roi]
        self.roi_mask_array = roi_mask_array
        return self.roi_mask_array

    def get_motion_correction(self):
        csv_file = os.path.join(self.processed_dir, 'log_0.csv')
        csv = pd.read_csv(csv_file, header=None)
        motion = {}
        motion['x_corr'] = csv[1].values
        motion['y_corr'] = csv[1].values
        self.motion_correction_values = motion
        return self.motion_correction_values

    def get_max_projection(self):
        import matplotlib.image as mpimg
        self.max_projection = mpimg.imread(os.path.join(self.processed_dir, 'max_downsample_4Hz_0.tiff'))
        return self.max_projection

    def get_dff_traces(self):
        dff_path = os.path.join(self.ophys_experiment_dir, str(self.experiment_id) + '_dff.h5')
        g = h5py.File(dff_path)
        self.dff_traces = np.asarray(g['data'])
        self.dff_traces = self.filter_traces(self.dff_traces)
        print 'length of traces:', self.dff_traces.shape[1]
        print 'number of segmented cells:', self.dff_traces.shape[0]
        timestamps = self.get_2p_timestamps()
        return self.dff_traces, timestamps

    def get_2p_timestamps(self):
        timestamps = self.sync['2PFrames']['timestamps']
        # totally unacceptable hack - figure out why this happens!
        if self.dff_traces.shape[1] < timestamps.shape[0]:
            timestamps = timestamps[:self.dff_traces.shape[1]]
        self.timestamps_2p = timestamps
        return self.timestamps_2p

    def get_raw_traces(self):
        file_path = os.path.join(self.processed_dir, 'roi_traces.h5')
        g = h5py.File(file_path)
        self.raw_traces = np.asarray(g['data'])
        self.raw_traces = self.filter_traces(self.raw_traces)
        return self.raw_traces

    def get_neuropil_traces(self):
        file_path = os.path.join(self.processed_dir, 'neuropil_traces.h5')
        g = h5py.File(file_path)
        self.neuropil_traces = np.asarray(g['data'])
        if self.filter_edge_cells:
            self.neuropil_traces = self.filter_traces(self.neuropil_traces)
        return self.neuropil_traces

    def get_neuropil_corrected_traces(self):
        file_path = os.path.join(self.ophys_experiment_dir, 'neuropil_correction.h5')
        g = h5py.File(file_path)
        self.neuropil_corrected_traces = np.asarray(g['FC'])
        if self.filter_edge_cells:
            self.neuropil_corrected_traces = self.filter_traces(self.neuropil_corrected_traces)
        return self.neuropil_corrected_traces

    def get_neuropil_correction_data(self):
        file_path = os.path.join(self.ophys_experiment_dir, 'neuropil_correction.h5')
        g = h5py.File(file_path)
        self.neuropil_RMSE = np.asarray(g['RMSE'])
        self.neuropil_r_values = np.asarray(g['r'])
        if self.filter_edge_cells:
            self.neuropil_RMSE = self.filter_traces(self.neuropil_RMSE)
            self.neuropil_r_values = self.filter_traces(self.neuropil_r_values)
        return self.neuropil_RMSE, self.neuropil_r

    def get_cell_specimen_ids(self):
        file_path = os.path.join(self.ophys_experiment_dir, 'neuropil_correction.h5')
        g = h5py.File(file_path)
        self.cell_specimen_ids = np.asarray(g['roi_names'])
        if self.filter_edge_cells:
            self.cell_specimen_ids = self.filter_traces(self.cell_specimen_ids)
        return self.cell_specimen_ids

    def get_cell_specimen_id_from_index(self, cell_index):
        cell_specimen_ids = self.get_cell_specimen_ids()
        return cell_specimen_ids[cell_index]

    def get_cell_index_from_specimen_id(self, cell_specimen_id):
        cell_index = np.where(self.get_cell_specimen_ids() == cell_specimen_id)[0][0]
        return cell_index

    def get_demixed_traces(self):
        file_path = os.path.join(self.demix_dir, str(self.experiment_id) + '_demixed_traces.h5')
        g = h5py.File(file_path)
        self.demixed_traces = np.asarray(g['data'])
        if self.filter_edge_cells:
            self.demixed_traces = self.filter_traces(self.demixed_traces)
        return self.demixed_traces
