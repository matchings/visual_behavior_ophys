# -*- coding: utf-8 -*-
"""
Created on Wednesday August 30 14:09:00 2017

@author: marinag
"""
import os
import h5py
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
from visual_behavior_ophys import dro.utilities as du
from visual_behavior_ophys.roi_masks import objectlist as ob
# from visual_behavior_ophys.core import stimulus_analysis as sa
# from ophys.ophystools import filter_digital
# import ophys.generate_response_dataframe as rd
# import scipy.ndimage.measurements as measurements



class OphysDataset(object):

    def __init__(self, expt_session_id, expt_dir, expt_info_df_path, filter_masks_and_traces = True):
        """initialize visual behavior ophys experiment dataset

            Parameters
            ----------
            boc: Brain Observatory Cache instance
            expt_session_id : ophys experiment session ID
        """
        self.filter_masks_and_traces = filter_masks_and_traces
        self.expt_info_df_path = expt_info_df_path
        self.expt_session_id = expt_session_id
        self.expt_dir = expt_dir
        self.get_ophys_metadata()
        self.get_pkl()
        self.get_pkl_df()
        self.get_stimulus_type()
        self.get_sync()
        self.get_motion_correction()
        self.get_save_dir()
        self.get_max_projection()
        self.get_roi_metrics()
        if filter_masks_and_traces:
            self.get_filter_params()
        self.get_all_traces_and_masks()


        #
        # self.stim_codes = sa.get_stim_codes_DoC(self.pkl)
        # self.stim_table = sa.get_stim_table_DoC(self.pkl, self.pkl_df, self.stim_codes, self.sync,
        #                                             self.stimulus_type)
        # self.mean_window_duration = 0.5
        # self.window = [-4, 4]
        # self.mean_window = [np.abs(self.window[0]),
        #                     np.abs(self.window[0]) + self.mean_window_duration]  # mean window 500ms after stimulus
        # self.inter_flash_interval = self.pkl['blank_duration_range'][0]
        # self.stimulus_duration = self.pkl['stim_duration']
        # self.previous_flash_start = float(self.mean_window[0]) - (self.inter_flash_interval + self.stimulus_duration)
        # self.previous_flash_window = [self.previous_flash_start, self.previous_flash_start + self.mean_window_duration]
        #


    def get_save_dir(self, suffix=None):
        if suffix:
            save_dir = os.path.join(self.expt_dir, 'analysis_' + suffix)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        else:
            save_dir = os.path.join(self.expt_dir, 'analysis')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        self.save_dir = save_dir


    def get_expt_info(self):
        tmp = pd.read_excel(self.expt_info_df_path)
        self.expt_info = tmp[tmp.expt_session_id == self.expt_session_id]
        return self.expt_info


    def get_ophys_metadata(self):
        expt_info = get_expt_info(self)
        tmp = expt_info
        metadata = {}
        metadata['session_id'] = tmp['session_id'].values[0]
        metadata['container_id'] = tmp['container_id'].values[0]
        metadata['location_id'] = tmp['location_id'].values[0]
        metadata['date'] = tmp['date'].values[0]
        metadata['mouse_id'] = tmp['mouse_id'].values[0]
        metadata['zoom'] = tmp['zoom'].values[0]
        metadata['imaging_depth'] = tmp['imaging_depth'].values[0]
        metadata['layer'] = [5 if metadata['imaging_depth'] >= 300 else None][0]
        metadata['genotype_abbreviated'] = tmp['genotype_abbreviated'].values[0]
        metadata['experiment_name'] = tmp['experiment_name'].values[0]
        metadata['operator'] = tmp['operator'].values[0]
        metadata['session_id'] = tmp['session_id'].values[0]
        return metadata


    def get_pkl_path(self):
        pkl_file = [file for file in os.listdir(self.expt_dir) if file.endswith(self.mouse_id + '.pkl')]
        pkl_path = os.path.join(self.expt_dir, pkl_file[0])
        self.pkl_path = pkl_path
        return self.pkl_path


    def get_pkl(self):
        pkl_path = self.get_pkl_path()
        with open(pkl_path, "rb+") as f:
            pkl = pickle.load(f)
        f.close()
        self.pkl = pkl
        return self.pkl


    def get_stimulus_type(self):
        pkl = self.pkl
        if 'stimulus_type' in pkl:
            self.stimulus_type = pkl['stimulus_type']
            print 'stim type is ',self.stimulus_type
        else:
            print 'stimulus_type not specified in pkl, setting stimulus_type to image'
            self.stimulus_type = 'image'
        if self.stimulus_type == 'grating':
            self.add_ori_columns_to_pkl_df()
        return self.stimulus_type


    def get_pkl_df(self):
        self.pkl = pd.read_pickle(self.pkl_path)
        pkl_df = du.create_doc_dataframe(self.pkl_path)
        pkl_df = pkl_df.replace(to_replace=np.nan, value=0)  # replace NaNs with 1 in response column
        pkl_df = pkl_df.rename(columns={'change_image_name': 'change_image'})
        pkl_df = pkl_df.rename(columns={'initial_image_name': 'initial_image'})
        pkl_df['trial_num'] = 1
        self.pkl_df = pkl_df
        return self.pkl_df


    def add_ori_columns_to_pkl_df(self):
        pkl_df = self.pkl_df
        pkl_df['initial_ori_adj'] = pd.Series(pkl_df.initial_ori % 360., index=pkl_df.index)
        pkl_df['change_ori_adj'] = pd.Series(pkl_df.change_ori % 360., index=pkl_df.index)
        pkl_df['initial_ori_str'] = pkl_df['initial_ori_adj'].astype('str')
        pkl_df['change_ori_str'] = pkl_df['change_ori_adj'].astype('str')
        pkl_df.replace(['0.0', '270.0', 'nan', '180.0', '90.0'],
                       ['vertical', 'horizontal', np.nan, 'vertical', 'horizontal'], inplace=True)
        self.pkl_df = pkl_df
        return self.pkl_df


    def get_sync_path(self):
        sync_file = [file for file in os.listdir(self.expt_dir) if
                     file.startswith(self.expt_id + '-') and file.endswith('.h5') and 'extracted_traces' not in file]
        sync_path = os.path.join(self.expt_dir, sync_file[0])
        self.sync_path = sync_path
        return sync_path


    def get_sync(self):
        from visual_behavior_ophys.sync import process_sync
        self.sync, self.sync_data = process_sync(self.sync_path,self.pkl)
        return self.sync


    def get_motion_correction(self):
        csv_file = os.path.join(self.expt_dir, 'log_0.csv')
        csv = pd.read_csv(csv_file, header=None)
        motion = {}
        motion['x_corr'] = csv[1].values
        motion['y_corr'] = csv[1].values
        self.motion_correction = motion
        return self.motion_correction


    def get_max_projection(self):
        self.max_projection = mpimg.imread(os.path.join(self.expt_dir, 'maxInt_a13.png'))
        return self.max_projection


    def get_roi_metrics(self):
        # objectlist.txt contains metrics associated with segmentation masks, used for filtering
        roi_metrics = pd.read_csv(os.path.join(self.expt_dir, 'objectlist.txt'))  # segmentation metrics
        roi_metrics = roi_metrics[roi_metrics[' traceindex'] != 999]  # remove edge cells, etc
        self.roi_metrics = roi_metrics  # objectlist segmentation metrics dataframe
        return self.roi_metrics


    # filter params
    def get_filter_parameters(self):
        zoom = self.metadata['zoom']
        layer = self.metadata['layer']
        self.filter_params = ob.get_stardard_filter_params(zoom,layer)
        return self.filter_params


    def get_filtered_roi_indices(self):
        self.filtered_roi_inds = ob.get_filtered_indices(self.roi_metrics, self.filter_params)
        return self.filtered_roi_inds


    def get_neuropil_masks(self):
            try:
                g = h5py.File(os.path.join(self.expt_dir, 'masks.h5'))
                unfiltered_neuropil_masks = np.asarray(g['neuropil_masks'])
                g.close()
            except:
                print 'no neuropil masks for', self.expt_session_id
            if self.filter_rois:
                filtered_neuropil_masks = np.empty((len(self.filtered_inds),unfiltered_neuropil_masks.shape))
                for filtered_roi_ind, unfiltered_roi_ind in enumerate(self.filtered_roi_inds):
                    filtered_neuropil_masks[filtered_roi_ind] = unfiltered_neuropil_masks[unfiltered_roi_ind, :]
                self.neuropil_masks = filtered_neuropil_masks
            else:
                self.neuropil_masks = unfiltered_neuropil_masks
            return self.neuropil_masks

    def get_roi_masks(self):
            try:
                g = h5py.File(os.path.join(self.expt_dir, 'masks.h5'))
                unfiltered_roi_masks = np.asarray(g['roi_masks'])
                g.close()
            except:
                print 'no roi masks for', self.expt_session_id
            if self.filter_rois:
                filtered_roi_masks = np.empty((len(self.filtered_inds),unfiltered_roi_masks.shape))
                for filtered_roi_ind, unfiltered_roi_ind in enumerate(self.filtered_roi_inds):
                    filtered_roi_masks[filtered_roi_ind] = unfiltered_roi_masks[unfiltered_roi_ind, :]
                self.roi_masks = filtered_roi_masks
            else:
                self.roi_masks = unfiltered_roi_masks
            return self.roi_masks

    def get_error_and_r_values(self,file_key):
        try:
            g = h5py.File(os.path.join(self.expt_dir, 'traces.h5'))
            unfiltered_error = np.asarray(g['error'])
            unfiltered_r_value = np.asarray(g['r_values'])
            g.close()
        except:
            print('no',file_key,'for', self.expt_session_id)
        if self.filter_rois:
            filtered_error = np.empty((len(self.filtered_inds))
            filtered_r_value = np.empty((len(self.filtered_inds))
            for filtered_ind, unfiltered_ind in enumerate(self.filtered_inds):
                filtered_error[filtered_ind] = unfiltered_error[unfiltered_ind, :]
                filtered_r_value[filtered_ind] = unfiltered_r_value[unfiltered_ind, :]
            self.neuropil_error = filtered_error
            self.neuropil_r_value = filtered_r_value
        else:
            self.neuropil_error = filtered_error
            self.neuropil_r_value = filtered_r_value
        return self.neuropil_error, self.neuropil_r_value


    def extract_traces_from_hdf5(self,file_key):
        try:
            g = h5py.File(os.path.join(self.expt_dir, 'traces.h5'))
            unfiltered_traces = np.asarray(g[file_key])
            g.close()
        except:
            print 'no',file_key,'for', self.expt_session_id
        if self.filter_rois:
            filtered_traces = np.empty((len(self.filtered_inds), unfiltered_traces.shape[1]))
            for filtered_ind, unfiltered_ind in enumerate(self.filtered_traces):
                filtered_traces[filtered_ind] = unfiltered_traces[unfiltered_ind, :]
            traces = filtered_traces
        else:
            traces = unfiltered_traces
        return traces

    def get_dff_traces(self):
        self.dff_traces = extract_traces_from_hdf5('dff_traces')
        return self.dff_traces

    def get_neuropil_traces(self):
        self.neuropil_traces = extract_traces_from_hdf5('neuropil_traces')
        return self.neuropil_traces

    def get_raw_traces(self):
        self.raw_traces = extract_traces_from_hdf5('roi_traces')
        return self.neuropil_traces

    def get_baseline_traces(self):
        self.baseline_traces = extract_traces_from_hdf5('baseline_traces')
        return self.baseline_traces

    def get_neuropil_corrected_traces(self):
        self.neuropil_corrected_traces = extract_traces_from_hdf5('corrected_traces')
        return self.neuropil_corrected_traces

    def get_all_traces_and_masks(self):
        self.get_neuropil_masks()
        self.get_roi_masks()
        self.get_error_and_r_values()
        self.get_dff_traces()
        self.get_neuropil_traces()
        self.get_raw_traces()
        self.get_baseline_traces()
        self.get_neuropil_corrected_traces()







