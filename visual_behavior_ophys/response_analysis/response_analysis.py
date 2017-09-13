import os
import numpy as np
import pandas as pd


class ResponseAnalysis(object):
    """ Base class for response analysis code. Contains methods for organizing responses
    by trial in a DataFrame, with a portion of the cell trace and running around the change time for
    every change trial, for every cell. Response DataFrame also contains behavioral metadata such as lick times,
    response type, reward rate, and stimulus information.

    Parameters
    ----------
    data_set: VisualBehaviorSutterDataset or VisualBehaviorScientificaDataset instance
    trial_window: list with elements corresponding to the duration, in seconds, before and after
    the start of a trial (change time) to use when selecting portion of a cell trace for DataFrame
    mean_response_duration: duration, in seconds, to use when computing mean response for a given trial

    """

    def __init__(self, dataset, trial_window=[-4, 4], response_window_duration=0.5):
        self.pkl = dataset.pkl
        self.pkl_df = dataset.pkl_df
        self.stim_codes = dataset.stim_codes
        self.stim_table = dataset.stim_table
        self.metadata = dataset.metadata
        self.save_dir = dataset.analysis_dir
        self.dff_traces, self.ophys_timestamps = dataset.get_dff_traces()
        self.running_speed, self.stimulus_timestamps = dataset.get_running_speed()
        self.trial_window = trial_window
        self.response_window_duration = response_window_duration
        self.response_window = [np.abs(self.trial_window[0]), np.abs(self.trial_window[0]) + self.response_window_duration]
        self.baseline_window = np.asarray(self.response_window) - self.response_window_duration
        self.inter_flash_interval = self.pkl['blank_duration_range'][0]
        self.stimulus_duration = self.pkl['stim_duration']
        self.previous_flash_start = float(self.response_window[0]) - (self.inter_flash_interval + self.stimulus_duration)
        self.previous_flash_window = [self.previous_flash_start, self.previous_flash_start + self.response_window_duration]
        self.get_response_dataframe()


    def get_nearest_frame(self, time_point, timestamps):
        return np.nanargmin(abs(timestamps - time_point))

    def get_trace_around_timepoint(self, timepoint, trace, timestamps, trial_window, inclusive_upp=True):
        lower_bound, upper_bound = [timepoint + bound for bound in trial_window]
        lo_idx = self.get_nearest_frame(lower_bound, timestamps)
        upp_idx = self.get_nearest_frame(upper_bound, timestamps)
        trace = trace[lo_idx:upp_idx + inclusive_upp]
        timepoints = timestamps[lo_idx:upp_idx + inclusive_upp]
        return trace, timepoints

    def get_lick_times_for_trial(self, total_trial):
        lick_frames = self.pkl_df.iloc[total_trial].lick_frames
        if len(lick_frames) != 0:
            lick_times = np.empty(len(lick_frames))
            for i, frame in enumerate(lick_frames):
                lick_times[i] = self.stimulus_timestamps[frame]
        else:
            lick_times = np.nan
        return lick_times

    def get_reward_rate_for_trial(self, total_trial):
        return self.pkl_df[self.pkl_df.trial == total_trial].reward_rate.values[0]

    def get_cell_response_dict(self):
        traces = self.dff_traces
        stim_table = self.stim_table
        trial_window = self.trial_window
        cell_response_dict = {idx: None for idx in range(len(traces))}
        for roi, trace in enumerate(traces):
            responses = {}
            for trial in range(len(stim_table)):
                trial_time = stim_table[(stim_table.change_trial == trial)].change_time.values
                timestamps = self.ophys_timestamps
                responses[trial] = self.get_trace_around_timepoint(trial_time, trace, timestamps, trial_window)
            cell_response_dict[roi] = responses
        return cell_response_dict

    # def generate_response_dataframe(self):
    #     cell_response_dict = self.get_cell_response_dict()
    #     stim_table = self.stim_table
    #     columns = ["cell", "change_trial", "total_trial", "change_frame", "change_frame_time", "initial_code", "change_code",
    #                "trial_type", "behavioral_response", "behavioral_response_type", "lick_times", "reward_rate"
    #                "response", "response_timestamps", "mean_response", "p_val", "significance"
    #                "run_speed", "run_timestamps", "mean_run_speed"]
    #     frames_in_window = (self.trial_window[1] - self.trial_window[0]) * self.metadata['ophys_frame_rate']
    #     frames_in_run_window = (self.trial_window[1] - self.trial_window[0]) * self.metadata['stimulus_frame_rate']
    #     df_list = []
    #     for cell in cell_response_dict.keys():
    #         for trial in stim_table.change_trial:
    #             response,timepoints = cell_response_dict[cell][trial]
    #             df = [None for col in columns]
    #             df[columns.index("cell")] = cell
    #             df[columns.index("trial")] = stim_table.change_trial[trial]
    #             df[columns.index("total_trial")] = stim_table.total_trial[trial]
    #             df[columns.index("change_frame")] = stim_table.change_frame[trial]
    #             df[columns.index("change_frame_time")] = stim_table.change_time[trial]
    #             df[columns.index("initial_code")] = stim_table.initial_code[trial]
    #             df[columns.index("change_code")] = stim_table.change_code[trial]
    #             df[columns.index("initial_image")] = self.pkl_df[self.pkl_df.change_image != 0].initial_image
    #             df[columns.index("change_image")] = self.pkl_df[self.pkl_df.change_image != 0].change_image
    #             df[columns.index("trial_type")] = stim_table.trial_type[trial]
    #             df[columns.index("behavioral_response")] = stim_table.behavioral_response[trial]
    #             df[columns.index("behavioral_response_type")] = stim_table.behavioral_response_type[trial]
    #             df[columns.index("lick_times")] = self.get_lick_times_from_frames(stim_table.total_trial[trial])
    #             df[columns.index("reward_rate")] = self.get_reward_rate_for_trial(stim_table.total_trial[trial])
    #             df[columns.index("response_timestamps")] = timepoints[:frames_in_window]
    #             df[columns.index("response")] = response[:frames_in_window]
    #             df[columns.index("mean_response")] = self.get_mean_in_window(response, self.mean_response_window,
    #                                                     self.metadata['ophys_frame_rate'])
    #             df[columns.index("p_val")] = self.get_pval(response[:frames_in_window])
    #             df[columns.index("significance")] = True
    #             run_speed_trace, run_speed_times = self.get_trace_around_timepoint(stim_table.change_time[trial],
    #                                                 self.running_speed, self.stimulus_timestamps, self.trial_window)
    #             df[columns.index("run_timestamps")] = run_speed_times[:frames_in_run_window]
    #             df[columns.index("run_speed")] = run_speed_trace[:frames_in_run_window]
    #             df[columns.index("mean_run_speed")] = self.get_mean_in_window(run_speed_trace,
    #                                                     self.mean_response_window, self.metadata['stimulus_frame_rate'])
    #             df_list.append(df)
    #     self.response_df = pd.DataFrame(df_list, columns=columns)
    #     return self.response_df

    def generate_response_dataframe(self):
        cell_response_dict = self.get_cell_response_dict()
        stim_table = self.stim_table
        columns = ["cell", "trial", "response", "response_timestamps", "response_window_mean", "baseline_window_mean",
                   "p_value", "sd_over_baseline", "run_speed", "run_timestamps", "mean_run_speed", "change_trial",
                   "lick_times", "reward_rate"]
        frames_in_window = np.int((self.trial_window[1] - self.trial_window[0]) * self.metadata['ophys_frame_rate'])
        frames_in_run_window = np.int(
            (self.trial_window[1] - self.trial_window[0]) * self.metadata['stimulus_frame_rate'])
        df_list = []
        for cell in cell_response_dict.keys():
            for trial in stim_table.change_trial:
                response, timepoints = cell_response_dict[cell][trial]
                df = [None for col in columns]
                df[columns.index("cell")] = cell
                df[columns.index("trial")] = trial
                df[columns.index("response_timestamps")] = timepoints[:frames_in_window]
                df[columns.index("response")] = response[:frames_in_window]
                df[columns.index("response_window_mean")] = self.get_mean_in_window(response, self.response_window,
                                                                             self.metadata['ophys_frame_rate'])
                df[columns.index("baseline_window_mean")] = self.get_mean_in_window(response, self.baseline_window,
                                                                                    self.metadata['ophys_frame_rate'])
                df[columns.index("p_value")] = self.get_p_val(response[:frames_in_window],self.response_window)
                df[columns.index("sd_over_baseline")] = self.get_sd_over_baseline(response[:frames_in_window])
                run_speed_trace, run_speed_times = self.get_trace_around_timepoint(stim_table.change_time[trial],
                                                                                   self.running_speed,
                                                                                   self.stimulus_timestamps,
                                                                                   self.trial_window)
                df[columns.index("run_timestamps")] = run_speed_times[:frames_in_run_window]
                df[columns.index("run_speed")] = run_speed_trace[:frames_in_run_window]
                df[columns.index("mean_run_speed")] = self.get_mean_in_window(run_speed_trace,
                                                                              self.response_window,
                                                                              self.metadata['stimulus_frame_rate'])
                df[columns.index("change_trial")] = trial
                df[columns.index("lick_times")] = self.get_lick_times_for_trial(stim_table.total_trial[trial])
                df[columns.index("reward_rate")] = self.get_reward_rate_for_trial(stim_table.total_trial[trial])

                df_list.append(df)
        df = pd.DataFrame(df_list, columns=columns)
        df = df.merge(stim_table, on='change_trial')
        self.response_df = df
        return self.response_df

    def get_response_dataframe(self):
        response_df_file = [file for file in os.listdir(self.save_dir) if file.endswith('response_dataframe.h5')]
        if len(response_df_file) > 0:
            print 'loading response dataframe'
            self.response_df = pd.read_hdf(os.path.join(self.save_dir, response_df_file[0]))
            print 'done'
        else:
            print 'generating response dataframe'
            self.response_df = self.generate_response_dataframe()
            print 'saving response dataframe'
            response_df_file_path = os.path.join(self.save_dir, 'response_dataframe.h5')
            self.response_df.to_hdf(response_df_file_path, key='df', format='fixed')
            # self.save_df_as_hdf(self.response_df,response_df_file_path)
            print 'done'
            # if self.global_dff:
            #     methods = [None]
            #     df = self.add_mean_sd(df, methods, period='baseline', window=np.asarray(self.mean_response_window) - 1)
            #     df = self.add_mean_sd(df, methods, period='response', window=self.mean_response_window)
            #     df = self.add_mean_sd(df, methods, period='previous_flash', window=self.previous_flash_window)
            # else:
            #     methods = [None, 'dFF']
            #     df = self.add_responses_dF(df, window=np.asarray(self.mean_response_window) - 1)
            #     df = self.add_p_vals(df, self.mean_response_window)
            #     df = self.add_mean_sd(df, methods, period='baseline', window=np.asarray(self.mean_response_window) - 1)
            #     df = self.add_mean_sd(df, methods, period='response', window=self.mean_response_window)
            #     for method in methods:
            #         df = self.add_significance(df, factor=5, method=method, offset=False)

        return self.response_df

    def get_mean_in_window(self, trace, window, frame_rate):
        return np.nanmean(trace[np.int(window[0] * frame_rate): np.int(window[1] * frame_rate)])

    def get_sd_in_window(self, trace, window, frame_rate):
        return np.std(trace[np.int(window[0] * frame_rate): np.int(window[1] * frame_rate)])

    def get_sd_over_baseline(self, trace):
        # baseline_mean = self.get_mean_in_window(trace, self.baseline_window, self.metadata['ophys_frame_rate'])
        baseline_std = self.get_sd_in_window(trace, self.baseline_window, self.metadata['ophys_frame_rate'])
        response_mean = self.get_mean_in_window(trace, self.response_window, self.metadata['ophys_frame_rate'])
        return response_mean / (baseline_std)

    def get_p_val(self, trace, response_window):
        # method borrowed from Brain Observatory analysis in allensdk
        from scipy import stats
        w = response_window
        window = w[1] - w[0]
        frame_rate = 30
        baseline_end = w[0] * frame_rate
        baseline_start = (w[0] - window) * frame_rate
        stim_start = w[0] * frame_rate
        stim_end = (w[0] + window) * frame_rate
        (_, p) = stats.f_oneway(trace[baseline_start:baseline_end], trace[stim_start:stim_end])
        return p

    def ptest(x, num_conditions):
        ptest = len(np.where(x < (0.05 / num_conditions))[0])
        return ptest

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

    def get_unique_values_for_columns(self, columns):
        # gets unique values for a given column in response dataframe.
        # for multiple columns, return list of values with each element corresponding to a column
        # input is list of column names
        df = self.response_df.copy()
        if len(columns) == 1:
            values = [[i] for i in df[columns[0]].unique()]
        elif len(columns) == 2:
            values = [[i, j] for i in df[columns[0]].unique() for j in df[columns[1]].unique()]
        elif len(columns) == 3:
            values = [[i, j, k] for i in df[columns[0]].unique() for j in df[columns[1]].unique()
                      for k in df[columns[2]].unique()]
        elif len(columns) == 4:
            values = [[i, j, k, l] for i in df[columns[0]].unique() for j in df[columns[1]].unique()
                      for k in df[columns[2]].unique() for l in df[columns[3]].unique()]
        elif len(columns) == 5:
            values = [[i, j, k, l, m] for i in df[columns[0]].unique() for j in df[columns[1]].unique()
                      for k in df[columns[2]].unique() for l in df[columns[3]].unique()
                      for m in df[columns[4]].unique()]
        return values

    def filter_df_by_column_values(self, columns, values):
        # filters response dataframe for given column names and column values. limited to 5 columns
        df = self.response_df.copy()
        if len(columns) == 1:
            filtered_df = df[(df[columns[0]] == values[0])]
        elif len(columns) == 2:
            filtered_df = df[(df[columns[0]] == values[0]) & (df[columns[1]] == values[1])]
        elif len(columns) == 3:
            filtered_df = df[
                (df[columns[0]] == values[0]) & (df[columns[1]] == values[1]) & (df[columns[2]] == values[2])]
        elif len(columns) == 4:
            filtered_df = df[
                (df[columns[0]] == values[0]) & (df[columns[1]] == values[1]) & (df[columns[2]] == values[2]) & (
                    df[columns[3]] == values[3])]
        elif len(columns) == 5:
            filtered_df = df[
                (df[columns[0]] == values[0]) & (df[columns[1]] == values[1]) & (df[columns[2]] == values[2])
                & (df[columns[3]] == values[3]) & (df[columns[4]] == values[4])]
        return filtered_df

    def get_stats_for_filtered_response_df(self, filtered_response_df):
        # threshold = mean dF/F value in mean_response_window to
        # significance_factor = # standard deviations over the baseline to call a response signficant
        mean_trace = filtered_response_df.response.mean()  # mean response across trials
        sem_trace = filtered_response_df.response.values.std() / np.sqrt(len(filtered_response_df.response))
        p_value = self.get_p_val(mean_trace, self.response_window)
        sd_over_baseline = self.get_sd_over_baseline(mean_trace)

        response_window_means = filtered_response_df.response_window_mean.values
        baseline_window_means = filtered_response_df.baseline_window_mean.values
        response_window_mean = filtered_response_df.response_window_mean.values.mean()
        baseline_window_mean = filtered_response_df.baseline_window_mean.values.mean()

        stats = [mean_trace, sem_trace, p_value, sd_over_baseline, response_window_means, baseline_window_means,
                 response_window_mean, baseline_window_mean]
        stats_columns = ['mean_trace', 'sem_trace', 'p_value', 'sd_over_baseline',
                         'response_window_means', 'baseline_window_means', 'response_window_mean',
                         'baseline_window_mean']
        return stats, stats_columns

    def get_mean_response_df(self, columns):
        # create dataframe with trial averaged responses & stats for all unique values of a given set of response dtatframe columns
        # columns: names of response df columns. combinations of unique column values will be used as conditions to take the mean response
        unique_values = self.get_unique_values_for_columns(self, columns)
        tmp = []
        for i in range(len(unique_values)):
            values = unique_values[i]
            filtered_response_df = self.filter_df_by_column_values(columns, values)
            if len(filtered_response_df) > 0:
                stats, stats_columns = self.get_stats_for_filtered_response_df(filtered_response_df)
                tmp.append(values + stats)
        column_names = columns + stats_columns
        mean_response_df = pd.DataFrame(tmp, columns=column_names)
        return mean_response_df


    # def get_cell_summary_df_DoC(dataset, df, mdf, SI=True):
    #     sdf = {}  # cell stats dataframe - summary statistics for each cell
    #     if 'response_type' in mdf:
    #         columns = ["cell", "max_response", "pref_stim", "pref_response_type", "p_value",
    #                    "responsive_conds", "suppressed_conds", "sig_conds_thresh", "sig_conds_pval", "sig_conds_sd",
    #                    "stim_SI", "change_SI", "hit_miss_SI", "hit_fa_SI"]  # "run_p_val","run_modulation"
    #     else:
    #         columns = ["cell", "max_response", "pref_stim", "pref_trial_type", "pref_response", "p_value",
    #                    "reliability", "responsive_conds", "suppressed_conds",
    #                    "sig_conds_thresh", "sig_conds_pval", "sig_conds_sd",
    #                    "stim_SI", "change_SI", "hit_miss_SI", "hit_fa_SI"]  # "run_p_val","run_modulation"
    #
    #     df_list = []
    #     for cell in df.cell.unique():
    #         cdf = mdf[mdf.cell == cell]
    #         # get pref_condition
    #         max_response = np.amax(cdf.mean_response)
    #         pref_cond = cdf[(cdf.mean_response == max_response)]
    #         pref_stim = pref_cond.stim_code.values[0]
    #         if 'response_type' in mdf:
    #             pref_response_type = pref_cond.response_type.values[0]
    #         else:
    #             pref_trial_type = pref_cond.trial_type.values[0]
    #             pref_response = pref_cond.response.values[0]
    #         if 'p_val' not in df.keys():
    #             df = rd.add_p_vals(df, dataset.mean_response_window)
    #         p_value = np.nan
    #         if 'response_type' not in mdf:
    #             pref_cond_trials = df[
    #                 (df.cell == cell) & (df.stim_code == pref_stim) & (df.trial_type == pref_trial_type) & (
    #                     df.response == pref_response)]
    #             reliability = len(np.where(pref_cond_trials.p_val < 0.05)[0]) / float(len(pref_cond_trials))
    #         # number of conditions where mean response is greater than 20%
    #         responsive_conditions = len(np.where((cdf.sig_thresh == True) & (cdf.suppressed == False))[0])
    #         suppressed_conditions = len(np.where((cdf.sig_thresh == True) & (cdf.suppressed == True))[0])
    #         # number of conditions where condition trials are significantly different than the blank trials
    #         sig_conds_thresh = len(np.where(cdf.sig_thresh == True)[0])
    #         sig_conds_pval = len(cdf[cdf.p_value < 0.05])
    #         sig_conds_sd = len(np.where(cdf.sig_sd == True)[0])
    #         #        #for pref_stim, compare running trials with stationary trials
    #         #        run_speeds = df[(df.cell==cell)&(df.stim_code==pref_stim)].avg_run_speed.values
    #         #        pref_means = df[(df.cell==cell)&(df.stim_code==pref_stim)].mean_response_dFF.values
    #         #        run_inds = np.where(run_speeds>=1)[0]
    #         #        stationary_inds = np.where(run_speeds<1)[0]
    #         #        run_means = pref_means[run_inds]
    #         #        stationary_means = pref_means[stationary_inds]
    #         #        if (len(run_means>2)) & (len(stationary_means>2)):
    #         #            f,run_p_val = stats.ks_2samp(run_means,stationary_means)
    #         #            run_modulation = np.mean(run_means)/np.mean(stationary_means)
    #         #        else:
    #         #            run_p_val = np.NaN
    #         #            run_modulation = np.NaN
    #         if SI:
    #             if 'response_type' not in mdf:
    #                 stim_0 = cdf[(cdf.stim_code == 0) & (cdf.trial_type == 'go')].mean_response.mean()
    #                 stim_1 = cdf[(cdf.stim_code == 1) & (cdf.trial_type == 'go')].mean_response.mean()
    #                 stim_SI = (stim_0 - stim_1) / (stim_0 + stim_1)
    #
    #                 go = cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'go')].mean_response.mean()
    #                 catch = cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'catch')].mean_response.mean()
    #                 change_SI = (go - catch) / (go + catch)
    #
    #                 hit = \
    #                     cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'go') & (
    #                     cdf.response == 1)].mean_response.values[
    #                         0]
    #                 miss = \
    #                     cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'go') & (
    #                     cdf.response == 0)].mean_response.values[
    #                         0]
    #                 hit_miss_SI = (hit - miss) / (hit + miss)
    #
    #                 hit = \
    #                     cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'go') & (
    #                     cdf.response == 1)].mean_response.values[
    #                         0]
    #                 fa = cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'catch') & (
    #                     cdf.response == 1)].mean_response.values[0]
    #                 hit_fa_SI = (hit - fa) / (hit + fa)
    #             else:
    #                 if len(mdf.stim_code.unique()) > 2:
    #                     stim_0 = cdf[(cdf.stim_code == pref_stim)].mean_response.mean()
    #                     stim_1 = cdf[(cdf.stim_code != pref_stim)].mean_response.mean()
    #                     stim_SI = (stim_0 - stim_1) / (stim_0 + stim_1)
    #                 else:
    #                     stim_0 = cdf[(cdf.stim_code == 0) & (cdf.response_type == 'CR')].mean_response.mean()
    #                     stim_1 = cdf[(cdf.stim_code == 1) & (cdf.response_type == 'CR')].mean_response.mean()
    #                     stim_SI = (stim_0 - stim_1) / (stim_0 + stim_1)
    #
    #                 go = cdf[
    #                     (cdf.stim_code == pref_stim) & (cdf.response_type.isin(['MISS', 'HIT']))].mean_response.mean()
    #                 catch = cdf[
    #                     (cdf.stim_code == pref_stim) & (cdf.response_type.isin(['CR', 'FA']))].mean_response.mean()
    #                 change_SI = (go - catch) / (go + catch)
    #
    #                 hit_df = cdf[(cdf.stim_code == pref_stim) & (cdf.response_type == 'HIT')]
    #                 miss_df = cdf[(cdf.stim_code == pref_stim) & (cdf.response_type == 'MISS')]
    #                 if (len(hit_df) > 0) and (len(miss_df) > 0):
    #                     hit = hit_df.mean_response.values[0]
    #                     miss = miss_df.mean_response.values[0]
    #                     hit_miss_SI = (hit - miss) / (hit + miss)
    #                 else:
    #                     hit_miss_SI = np.nan
    #
    #                 fa_df = cdf[(cdf.stim_code == pref_stim) & (cdf.response_type == 'FA')]
    #                 if (len(fa_df) > 0) and (len(hit_df) > 0):
    #                     fa = fa_df.mean_response.values[0]
    #                     hit = hit_df.mean_response.values[0]
    #                     hit_fa_SI = (hit - fa) / (hit + fa)
    #                 else:
    #                     hit_fa_SI = np.nan
    #         else:
    #             stim_SI = np.nan
    #             change_SI = np.nan
    #             hit_miss_SI = np.nan
    #             hit_fa_SI = np.nan
    #
    #         row = [None for col in columns]
    #         row[columns.index("cell")] = cell
    #         row[columns.index("max_response")] = max_response
    #         row[columns.index("pref_stim")] = pref_stim
    #         if 'response_type' in mdf:
    #             row[columns.index("pref_response_type")] = pref_response_type
    #         else:
    #             row[columns.index("pref_trial_type")] = pref_trial_type
    #             row[columns.index("pref_response")] = pref_response
    #             row[columns.index("reliability")] = reliability
    #         row[columns.index("p_value")] = p_value
    #         row[columns.index("responsive_conds")] = responsive_conditions
    #         row[columns.index("suppressed_conds")] = suppressed_conditions
    #         row[columns.index("sig_conds_thresh")] = sig_conds_thresh
    #         row[columns.index("sig_conds_pval")] = sig_conds_pval
    #         row[columns.index("sig_conds_sd")] = sig_conds_sd
    #         #        row[columns.index("run_p_val")] = run_p_val
    #         #        row[columns.index("run_modulation")] = run_modulation
    #         row[columns.index("stim_SI")] = stim_SI
    #         row[columns.index("change_SI")] = change_SI
    #         row[columns.index("hit_miss_SI")] = hit_miss_SI
    #         row[columns.index("hit_fa_SI")] = hit_fa_SI
    #         df_list.append(row)
    #
    #     sdf = pd.DataFrame(df_list, columns=columns)
    #     return sdf

    def add_pref_stim_to_df(df, sdf):
        pref_stim_list = []
        for row in range(len(df)):
            pref_stim = sdf[sdf.cell == df.iloc[row].cell].pref_stim.values[0]
            if df.iloc[row].change_code == pref_stim:
                pref_stim_list.append(True)
            else:
                pref_stim_list.append(False)
        df['pref_stim'] = pref_stim_list
        return df

    def get_summary_dfs(dataset,response_df):
        mean_window = dataset.mean_response_window
        tmp = dataset.response_df.copy()
        tmp = tmp.replace(to_replace=np.nan, value=0)
        tmp = tmp[tmp.trial_type != 'autorewarded']
        tmp = tmp.rename(columns={'change_code': 'stim_code'})
        tmp = tmp[tmp.trial.isin(tmp.trial.unique()[:-5])]
        columns = ['cell', 'stim_code', 'trial_type', 'response']
        rdf = get_mean_df(tmp, columns)

        conditions = ['cell', 'stim_code', 'response_type']
        mdf = get_mean_df(tmp, conditions)

        sdf = get_cell_summary_df_DoC(dataset, tmp, mdf)

        mdf = add_stuff_to_mdf(dataset, mdf, sdf)

        return mdf, rdf, sdf
