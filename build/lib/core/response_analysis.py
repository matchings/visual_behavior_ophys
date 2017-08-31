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


def add_reward_rate_to_response_df(self):
    pkl_df = self.pkl_df
    df = self.df
    pkl_df['trial'] = pkl_df.index.values
    tmp = pkl_df[pkl_df.trial.isin(df.global_trial.unique())]
    tmp = tmp[['trial', 'reward_rate']]
    df = df.join(tmp, on='trial', how='left', lsuffix='_pkl')
    del df['trial']
    df = df.rename(columns={'trial_pkl': 'trial'})
    self.df = df


def save_response_df(self, df_path):
    df = self.df
    f = open(df_path, 'wb')
    pickle.dump(df, f)
    f.close()


def add_reward_rate_to_df(self):
    pkl_df = self.pkl_df
    df = self.df
    pkl_df['trial'] = pkl_df.index.values
    tmp = pkl_df[pkl_df.trial.isin(df.global_trial.unique())]
    tmp = tmp[['trial', 'reward_rate']]
    df = df.join(tmp, on='trial', how='left', lsuffix='_pkl')
    del df['trial']
    df = df.rename(columns={'trial_pkl': 'trial'})
    self.df = df


def get_response_df(self, traces=None):
    df_file = [file for file in os.listdir(self.save_dir) if file.endswith('response_df.pkl')]
    if len(df_file) > 0:
        self.df_path = os.path.join(self.save_dir, df_file[0])
        print 'loading response_df from:', self.df_path
        with open(self.df_path, "rb+") as f:
            df = pickle.load(f)
        f.close()
        self.df = df
        if (self.expt_type == 'DetectionOfChange') and ('reward_rate' not in df.columns):
            print 'adding reward reate to df'
            self.add_reward_rate_to_df()
    else:
        print 'response_df not saved, creating it'
        if traces is None:
            cell_response_dict = rd.get_cell_response_dict(self.traces, self.stim_table, self.sync, self.window)
        else:
            cell_response_dict = rd.get_cell_response_dict(traces, self.stim_table, self.sync, self.window)
        if self.expt_type == 'Foraging':
            df = rd.get_final_df(cell_response_dict, self.session, self.stim_table)
        elif self.expt_type == 'SweepStim':
            df = rd.get_final_df_sweep_stim(cell_response_dict, self.stim_table, self.pkl, self.session,
                                            self.window)
        elif self.expt_type == 'DetectionOfChange':
            df = rd.get_final_df_DoC(cell_response_dict, self.stim_table, self.session, self.window)

        if self.global_dff:
            methods = [None]
            df = rd.add_mean_sd(df, methods, period='baseline', window=np.asarray(self.mean_window) - 1)
            df = rd.add_mean_sd(df, methods, period='response', window=self.mean_window)
            df = rd.add_mean_sd(df, methods, period='previous_flash', window=self.previous_flash_window)
        else:
            methods = [None, 'dFF']
            df = rd.add_responses_dF(df, window=np.asarray(self.mean_window) - 1)
            df = rd.add_p_vals(df, self.mean_window)
            df = rd.add_mean_sd(df, methods, period='baseline', window=np.asarray(self.mean_window) - 1)
            df = rd.add_mean_sd(df, methods, period='response', window=self.mean_window)
            for method in methods:
                df = rd.add_significance(df, factor=5, method=method, offset=False)
                #            df = rd.add_mean_response(df,methods,window=self.mean_window)
                #            if self.expt_type == 'SweepStim':
                #                df = rd.add_responses_dF(df,window=np.asarray(self.mean_window)-1)
                #                df = rd.add_mean_sd(df,methods=[None,'dFF'],period='baseline',window=np.asarray(self.mean_window)-1)
                #                df = rd.add_mean_sd(df,methods=[None,'dFF'],period='response',window=self.mean_window)
                #                df = rd.add_significance(df,factor=5,method='dFF',offset=False)
                #                df = rd.add_p_vals(df,self.mean_window)
                #                df = rd.add_mean_response(df,methods=['dFF'],window=self.mean_window)

        self.df = df
        if self.expt_type == 'DetectionOfChange':
            print 'adding reward rate to df'
            self.add_reward_rate_to_df()
        print 'saving response_df'
        self.df_path = os.path.join(self.save_dir, str(self.expt_id) + '_response_df.pkl')
        f = open(self.df_path, 'wb')
        pickle.dump(df, f)
        f.close()
    return self.df


def get_stim_codes(pkl):
    stim_codes_list = []
    i = 0

    if 'image_list' in pkl:
        for image_name in np.sort(pkl['image_list']):
            stim_codes_list.append([i, image_name])
            i += 1
        stim_codes = pd.DataFrame(stim_codes_list, columns=['stim_code', 'image_name'])
    elif 'image_dict' in pkl:
        for image_num in np.sort(pkl['image_dict'].keys()):
            for image_name in pkl['image_dict'][image_num].keys():
                #            for image_name in np.sort(pkl['image_dict'][image_num].keys()): #to make same order as behavior figures
                stim_codes_list.append([i, image_name, image_num])
                i += 1
        stim_codes = pd.DataFrame(stim_codes_list, columns=['stim_code', 'image_name', 'image_num'])
    elif 'image_names' in pkl:
        for image_name in pkl['image_names']:
            for size in pkl['sizes']:
                for position in pkl['positions']:
                    for ori in pkl['oris']:
                        for contrast in pkl['contrasts']:
                            stim_codes_list.append([i, image_name, size, position, ori, contrast])
                            i += 1
        stim_codes = pd.DataFrame(stim_codes_list,
                                  columns=['stim_code', 'image_name', 'image_size', 'position', 'ori', 'contrast'])
    elif 'stimulus_type' in pkl:
        if pkl['stimulus_type'] == 'grating':
            #            ori_0 = 0
            #            ori_list = [ori_0]
            #            for ori in pkl['delta_oris']:
            #                ori_list.append(ori_0+ori)
            ori_list = np.arange(0, 360, pkl['delta_oris'][0])
            for ori in ori_list:
                if (ori == 0) or (ori == 180):
                    image_name = 'vertical'
                elif (ori == 90) or (ori == 270):
                    image_name = 'horizontal'
                stim_codes_list.append([i, ori, image_name])
                i += 1
            stim_codes = pd.DataFrame(stim_codes_list, columns=['stim_code', 'ori', 'image_name'])
    return stim_codes


# returns abbreviated image name (ex: bird rather than bird_black.png)
def get_image_for_code(dataset, stim_code):
    return dataset.stim_codes[dataset.stim_codes.stim_code == stim_code].image_name.values[0].split('_')[0]


# returns full image name
def get_image_name_for_code(dataset, stim_code):
    return dataset.stim_codes[dataset.stim_codes.stim_code == stim_code].image_name.values[0]


# input image name, get back code
def get_code_for_image_name(dataset, image_name):
    return dataset.stim_codes[dataset.stim_codes.image_name == image_name].stim_code.values[0]

