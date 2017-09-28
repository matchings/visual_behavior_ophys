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
        columns = ['cell', 'change_code', 'behavioral_response_type']
        self.mean_response_df = self.get_mean_response_df(columns)
        # self.cell_summary_df = self.get_cell_summary_df(p_val_thresh=0.005, sd_over_baseline_thresh=3)

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
        else:
            print 'generating response dataframe'
            self.response_df = self.generate_response_dataframe()
            print 'saving response dataframe'
            response_df_file_path = os.path.join(self.save_dir, 'response_dataframe.h5')
            self.response_df.to_hdf(response_df_file_path, key='df', format='fixed')
            # self.save_df_as_hdf(self.response_df,response_df_file_path)
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
        # ex: ['cell', 'change_code', 'behavioral_response_type'] gives a dataframe with mean response data for every cell, averaged across
        # each unique combination of change_code and behavioral_response_type
        print 'creating mean response dataframe'
        unique_values = self.get_unique_values_for_columns(columns)
        tmp = []
        for i in range(len(unique_values)):
            values = unique_values[i]
            filtered_response_df = self.filter_df_by_column_values(columns, values)
            if len(filtered_response_df) > 0:
                stats, stats_columns = self.get_stats_for_filtered_response_df(filtered_response_df)
                tmp.append(values + stats)
        column_names = columns + stats_columns
        mdf = pd.DataFrame(tmp, columns=column_names)
        return mdf

    def get_cell_summary_df(self, p_val_thresh=0.005, sd_over_baseline_thresh=3):
        print 'creating cell summary dataframe'
        df = self.response_df
        mdf = self.mean_response_df
        columns = ["cell", "max_response", "pref_stim_code", "pref_image_name", "pref_behavioral_response",
                   "p_value_pref_cond", "sd_over_baseline_pref_cond", "reliability_p_val", "reliability_sd",
                   "responsive_conditions_p_val", "responsive_conditions_sd"]  # "run_p_val","run_modulation"

        df_list = []
        for cell in df.cell.unique():
            cdf = mdf[mdf.cell == cell]
            max_response = np.nanmax(cdf.response_window_mean.values)
            pref_stim_code = cdf[(cdf.response_window_mean == max_response)].change_code.values[0]
            pref_image_name = self.stim_codes[self.stim_codes.stim_code == pref_stim_code].image_name.values[0]
            pref_behavioral_response = cdf[(cdf.response_window_mean == max_response)].behavioral_response_type.values[
                0]
            # get significance of mean response at preferred condition
            p_value_pref_cond = cdf[(cdf.response_window_mean == max_response)].p_value.values[0]
            sd_over_thresh_pref_cond = cdf[(cdf.response_window_mean == max_response)].sd_over_baseline.values[0]
            # get fraction of trials for preferred condition for which the response was significant
            pref_cond_trials = df[(df.cell == cell) & (df.change_code == pref_stim_code) & (
            df.behavioral_response_type == pref_behavioral_response)]
            reliability_p_val = len(np.where(pref_cond_trials.p_value < p_val_thresh)[0]) / float(len(pref_cond_trials))
            reliability_sd = len(np.where(pref_cond_trials.sd_over_baseline > sd_over_baseline_thresh)[0]) / float(
                len(pref_cond_trials))
            # get number of conditions where mean response is signficant
            responsive_conditions_p_val = len(np.where(cdf.p_value < p_val_thresh)[0])
            responsive_conditions_sd = len(np.where(cdf.sd_over_baseline > sd_over_baseline_thresh)[0])
            # make dataframe
            row = [None for col in columns]
            row[columns.index("cell")] = cell
            row[columns.index("max_response")] = max_response
            row[columns.index("pref_stim_code")] = pref_stim_code
            row[columns.index("pref_image_name")] = pref_image_name
            row[columns.index("pref_behavioral_response")] = pref_behavioral_response
            row[columns.index("p_value_pref_cond")] = p_value_pref_cond
            row[columns.index("sd_over_baseline_pref_cond")] = sd_over_thresh_pref_cond
            row[columns.index("reliability_p_val")] = reliability_p_val
            row[columns.index("reliability_sd")] = reliability_sd
            row[columns.index("responsive_conditions_p_val")] = responsive_conditions_p_val
            row[columns.index("responsive_conditions_sd")] = responsive_conditions_sd
            #        row[columns.index("run_p_val")] = run_p_val
            #        row[columns.index("run_modulation")] = run_modulation
            df_list.append(row)
        sdf = pd.DataFrame(df_list, columns=columns)
        return sdf

    def add_pref_stim_to_df(self, df, sdf):
        pref_stim_list = []
        for row in range(len(df)):
            pref_stim = sdf[sdf.cell == df.iloc[row].cell].pref_stim_code.values[0]
            if df.iloc[row].change_code == pref_stim:
                pref_stim_list.append(True)
            else:
                pref_stim_list.append(False)
        df['pref_stim_code'] = pref_stim_list
        return df

    def add_stim_codes_to_pkl_df(self, pkl_df):
        stim_codes = self.stim_codes
        pkl_df['initial_code'] = [
            stim_codes[stim_codes.image_name == pkl_df.iloc[trial].initial_image].stim_code.values[0]
            for trial in range(len(pkl_df))]
        pkl_df['change_code'] = [
            stim_codes[stim_codes.image_name == pkl_df.iloc[trial].change_image].stim_code.values[0]
            for trial in range(len(pkl_df))]
        self.pkl_df = pkl_df
        return pkl_df

    def get_summary_dfs(dataset):
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

        sdf = get_cell_summary_df(dataset, tmp, mdf)

        mdf = add_stuff_to_mdf(dataset, mdf, sdf)

        return mdf, rdf, sdf
