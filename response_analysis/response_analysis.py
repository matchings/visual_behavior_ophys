



class ResponseAnalysis(object):
    """ Base class for all response analysis code. Subclasses are responsible
    for computing metrics and traces relevant to a particular stimulus.
    The base class contains methods for organizing sweep responses row of
    a stimulus stable (get_sweep_response).  Subclasses implement the
    get_response method, computes the mean sweep response to all sweeps for
    a each stimulus condition.

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet instance

    speed_tuning: boolean, deprecated
       Whether or not to compute speed tuning histograms

    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._timestamps = StimulusAnalysis._PRELOAD
        self._celltraces = StimulusAnalysis._PRELOAD
        self._acquisition_rate = StimulusAnalysis._PRELOAD
        self._numbercells = StimulusAnalysis._PRELOAD
        self._roi_id = StimulusAnalysis._PRELOAD
        self._cell_id = StimulusAnalysis._PRELOAD
        self._dfftraces = StimulusAnalysis._PRELOAD
        self._dxcm = StimulusAnalysis._PRELOAD
        self._dxtime = StimulusAnalysis._PRELOAD
        self._binned_dx_sp = StimulusAnalysis._PRELOAD
        self._binned_cells_sp = StimulusAnalysis._PRELOAD
        self._binned_dx_vis = StimulusAnalysis._PRELOAD
        self._binned_cells_vis = StimulusAnalysis._PRELOAD
        self._peak_run = StimulusAnalysis._PRELOAD
        self._binsize = 800

        self._stim_table = StimulusAnalysis._PRELOAD
        self._response = StimulusAnalysis._PRELOAD
        self._sweep_response = StimulusAnalysis._PRELOAD
        self._mean_sweep_response = StimulusAnalysis._PRELOAD
        self._pval = StimulusAnalysis._PRELOAD
        self._peak = StimulusAnalysis._PRELOAD

        # self.mean_window_duration = 0.5
        # self.window = [-4, 4]
        # self.mean_window = [np.abs(self.window[0]),
        #                     np.abs(self.window[0]) + self.mean_window_duration]  # mean window 500ms after stimulus
        # self.inter_flash_interval = self.pkl['blank_duration_range'][0]
        # self.stimulus_duration = self.pkl['stim_duration']
        # self.previous_flash_start = float(self.mean_window[0]) - (self.inter_flash_interval + self.stimulus_duration)
        # self.previous_flash_window = [self.previous_flash_start, self.previous_flash_start + self.mean_window_duration]
        #

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

    def get_p_val(x, mean_window):
        # get frame #s for window before stim and after stim
        w = mean_window
        window = w[1] - w[0]
        frame_rate = 30
        baseline_end = w[0] * frame_rate
        baseline_start = (w[0] - window) * frame_rate
        stim_start = w[0] * frame_rate
        stim_end = (w[0] + window) * frame_rate
        (_, p) = stats.f_oneway(x[baseline_start:baseline_end], x[stim_start:stim_end])
        return p


    def ptest(x, num_conditions):
        ptest = len(np.where(x < (0.05 / num_conditions))[0])
        return ptest


    def get_values_for_conditions(df, conditions):
        if len(conditions) == 1:
            values = [[i] for i in df[conditions[0]].unique()]
        elif len(conditions) == 2:
            values = [[i, j] for i in df[conditions[0]].unique() for j in df[conditions[1]].unique()]
        elif len(conditions) == 3:
            values = [[i, j, k] for i in df[conditions[0]].unique() for j in df[conditions[1]].unique()
                      for k in df[conditions[2]].unique()]
        elif len(conditions) == 4:
            values = [[i, j, k, l] for i in df[conditions[0]].unique() for j in df[conditions[1]].unique()
                      for k in df[conditions[2]].unique() for l in df[conditions[3]].unique()]
        elif len(conditions) == 5:
            values = [[i, j, k, l, m] for i in df[conditions[0]].unique() for j in df[conditions[1]].unique()
                      for k in df[conditions[2]].unique() for l in df[conditions[3]].unique()
                      for m in df[conditions[4]].unique()]
        return values


    def get_cond_for_vals(df, conditions, vals):
        if len(conditions) == 1:
            cond = df[(df[conditions[0]] == vals[0])]
        elif len(conditions) == 2:
            cond = df[(df[conditions[0]] == vals[0]) & (df[conditions[1]] == vals[1])]
        elif len(conditions) == 3:
            cond = df[(df[conditions[0]] == vals[0]) & (df[conditions[1]] == vals[1]) & (df[conditions[2]] == vals[2])]
        elif len(conditions) == 4:
            cond = df[(df[conditions[0]] == vals[0]) & (df[conditions[1]] == vals[1]) & (df[conditions[2]] == vals[2]) & (
            df[conditions[3]] == vals[3])]
        elif len(conditions) == 5:
            cond = df[(df[conditions[0]] == vals[0]) & (df[conditions[1]] == vals[1]) & (df[conditions[2]] == vals[2])
                      & (df[conditions[3]] == vals[3]) & (df[conditions[4]] == vals[4])]
        return cond


    def get_condition_data(cond, mean_window, thresh, factor):
        mean_trace = cond.responses_dFF.mean()  # mean response across trials
        sem_trace = cond.responses_dFF.values.std() / np.sqrt(len(cond.responses_dFF))
        mean_response = cond.mean_response_dFF.mean()
        sem_response = cond.mean_response_dFF.values.std() / float(np.sqrt(len(cond.mean_response_dFF.values)))
        sd_response = cond.mean_response_dFF.values.std()
        mean_baseline = cond.mean_baseline_dFF.mean()
        std_baseline = np.std(cond.mean_baseline_dFF)
        sig_thresh = std_baseline * factor
        # is mean response 5x larger than standard deviation of baseline
        if (mean_response > mean_baseline + sig_thresh) or (mean_response < mean_baseline - sig_thresh):
            sig_sd = True
        else:
            sig_sd = False
        # is mean response greater than threshold value
        if (mean_response > thresh):
            sig_thresh = True
            suppressed = False
        elif (mean_response < -thresh):
            sig_thresh = True
            suppressed = True
        else:
            sig_thresh = False
            suppressed = False
        # p-value comparing stim window to baseline of average trace
        mean_trace = cond.responses_dFF.mean()
        p_value = get_p_val(mean_trace, mean_window)
        #    mean_run_speed = cond.run_speed.mean() #mean response across trials
        #    sem_run_speed = cond.run_speed.std()/np.sqrt(len(cond.run_speed))
        mean_dff = None

        condition_data = [mean_dff, mean_trace, sem_trace, mean_response, sem_response, sd_response, mean_baseline,
                          std_baseline, sig_sd, sig_thresh, suppressed, p_value]  # ,mean_run_speed,sem_run_speed
        return condition_data


    def get_condition_data_global_dFF(cond, mean_window, thresh, factor):
        mean_trace = cond.responses.mean()  # mean response across trials
        sem_trace = cond.responses.values.std() / np.sqrt(len(cond.responses))
        mean_response = cond.mean_response.mean()
        sem_response = cond.mean_response.values.std() / float(np.sqrt(len(cond.mean_response.values)))
        sd_response = cond.mean_response.values.std()
        mean_baseline = cond.mean_baseline.mean()
        std_baseline = np.std(cond.mean_baseline)
        mean_dff = (mean_response - mean_baseline) / mean_baseline
        sig_thresh = std_baseline * factor
        # is mean response 5x larger than standard deviation of baseline
        if (mean_response > mean_baseline + sig_thresh) or (mean_response < mean_baseline - sig_thresh):
            sig_sd = True
        else:
            sig_sd = False
        # is mean response greater than threshold value
        if (mean_response > thresh):
            sig_thresh = True
            suppressed = False
        elif (mean_response < -thresh):
            sig_thresh = True
            suppressed = True
        else:
            sig_thresh = False
            suppressed = False
        # p-value comparing stim window to baseline of average trace
        mean_trace = cond.responses.mean()
        p_value = get_p_val(mean_trace, mean_window)
        #    mean_run_speed = cond.run_speed.mean() #mean response across trials
        #    sem_run_speed = cond.run_speed.std()/np.sqrt(len(cond.run_speed))
        condition_data = [mean_dff, mean_trace, sem_trace, mean_response, sem_response, sd_response, mean_baseline,
                          std_baseline, sig_sd, sig_thresh, suppressed, p_value]
        return condition_data


    def get_mean_df(df, conditions, mean_window, global_dFF=False, thresh=0.3, factor=5):
        values = get_values_for_conditions(df, conditions)
        columns = conditions + ['mean_dff', 'mean_trace', 'sem_trace', 'mean_response', 'sem_response', 'sd_response',
                                'mean_baseline', 'std_baseline', 'sig_sd', 'sig_thresh', 'suppressed', 'p_value']
        mdfl = []
        for i in range(len(values)):
            vals = values[i]
            cond = get_cond_for_vals(df, conditions, vals)
            if len(cond) > 0:
                if global_dFF:
                    condition_data = get_condition_data_global_dFF(cond, mean_window, thresh, factor)
                else:
                    condition_data = get_condition_data(cond, mean_window, thresh, factor)
                mdfl.append(vals + condition_data)
        mdf = pd.DataFrame(mdfl, columns=columns)
        return mdf


    def get_cell_summary_df_DoC(dataset, df, mdf, SI=True):
        sdf = {}  # cell stats dataframe - summary statistics for each cell
        if 'response_type' in mdf:
            columns = ["cell", "max_response", "pref_stim", "pref_response_type", "p_value",
                       "responsive_conds", "suppressed_conds", "sig_conds_thresh", "sig_conds_pval", "sig_conds_sd",
                       "stim_SI", "change_SI", "hit_miss_SI", "hit_fa_SI"]  # "run_p_val","run_modulation"
        else:
            columns = ["cell", "max_response", "pref_stim", "pref_trial_type", "pref_response", "p_value",
                       "reliability", "responsive_conds", "suppressed_conds",
                       "sig_conds_thresh", "sig_conds_pval", "sig_conds_sd",
                       "stim_SI", "change_SI", "hit_miss_SI", "hit_fa_SI"]  # "run_p_val","run_modulation"

        df_list = []
        for cell in df.cell.unique():
            cdf = mdf[mdf.cell == cell]
            # get pref_condition
            max_response = np.amax(cdf.mean_response)
            pref_cond = cdf[(cdf.mean_response == max_response)]
            pref_stim = pref_cond.stim_code.values[0]
            if 'response_type' in mdf:
                pref_response_type = pref_cond.response_type.values[0]
            else:
                pref_trial_type = pref_cond.trial_type.values[0]
                pref_response = pref_cond.response.values[0]
            if 'p_val' not in df.keys():
                df = rd.add_p_vals(df, dataset.mean_window)
            p_value = np.nan
            if 'response_type' not in mdf:
                pref_cond_trials = df[
                    (df.cell == cell) & (df.stim_code == pref_stim) & (df.trial_type == pref_trial_type) & (
                    df.response == pref_response)]
                reliability = len(np.where(pref_cond_trials.p_val < 0.05)[0]) / float(len(pref_cond_trials))
            # number of conditions where mean response is greater than 20%
            responsive_conditions = len(np.where((cdf.sig_thresh == True) & (cdf.suppressed == False))[0])
            suppressed_conditions = len(np.where((cdf.sig_thresh == True) & (cdf.suppressed == True))[0])
            # number of conditions where condition trials are significantly different than the blank trials
            sig_conds_thresh = len(np.where(cdf.sig_thresh == True)[0])
            sig_conds_pval = len(cdf[cdf.p_value < 0.05])
            sig_conds_sd = len(np.where(cdf.sig_sd == True)[0])
            #        #for pref_stim, compare running trials with stationary trials
            #        run_speeds = df[(df.cell==cell)&(df.stim_code==pref_stim)].avg_run_speed.values
            #        pref_means = df[(df.cell==cell)&(df.stim_code==pref_stim)].mean_response_dFF.values
            #        run_inds = np.where(run_speeds>=1)[0]
            #        stationary_inds = np.where(run_speeds<1)[0]
            #        run_means = pref_means[run_inds]
            #        stationary_means = pref_means[stationary_inds]
            #        if (len(run_means>2)) & (len(stationary_means>2)):
            #            f,run_p_val = stats.ks_2samp(run_means,stationary_means)
            #            run_modulation = np.mean(run_means)/np.mean(stationary_means)
            #        else:
            #            run_p_val = np.NaN
            #            run_modulation = np.NaN
            if SI:
                if 'response_type' not in mdf:
                    stim_0 = cdf[(cdf.stim_code == 0) & (cdf.trial_type == 'go')].mean_response.mean()
                    stim_1 = cdf[(cdf.stim_code == 1) & (cdf.trial_type == 'go')].mean_response.mean()
                    stim_SI = (stim_0 - stim_1) / (stim_0 + stim_1)

                    go = cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'go')].mean_response.mean()
                    catch = cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'catch')].mean_response.mean()
                    change_SI = (go - catch) / (go + catch)

                    hit = \
                    cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'go') & (cdf.response == 1)].mean_response.values[
                        0]
                    miss = \
                    cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'go') & (cdf.response == 0)].mean_response.values[
                        0]
                    hit_miss_SI = (hit - miss) / (hit + miss)

                    hit = \
                    cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'go') & (cdf.response == 1)].mean_response.values[
                        0]
                    fa = cdf[(cdf.stim_code == pref_stim) & (cdf.trial_type == 'catch') & (
                    cdf.response == 1)].mean_response.values[0]
                    hit_fa_SI = (hit - fa) / (hit + fa)
                else:
                    if len(mdf.stim_code.unique()) > 2:
                        stim_0 = cdf[(cdf.stim_code == pref_stim)].mean_response.mean()
                        stim_1 = cdf[(cdf.stim_code != pref_stim)].mean_response.mean()
                        stim_SI = (stim_0 - stim_1) / (stim_0 + stim_1)
                    else:
                        stim_0 = cdf[(cdf.stim_code == 0) & (cdf.response_type == 'CR')].mean_response.mean()
                        stim_1 = cdf[(cdf.stim_code == 1) & (cdf.response_type == 'CR')].mean_response.mean()
                        stim_SI = (stim_0 - stim_1) / (stim_0 + stim_1)

                    go = cdf[(cdf.stim_code == pref_stim) & (cdf.response_type.isin(['MISS', 'HIT']))].mean_response.mean()
                    catch = cdf[(cdf.stim_code == pref_stim) & (cdf.response_type.isin(['CR', 'FA']))].mean_response.mean()
                    change_SI = (go - catch) / (go + catch)

                    hit_df = cdf[(cdf.stim_code == pref_stim) & (cdf.response_type == 'HIT')]
                    miss_df = cdf[(cdf.stim_code == pref_stim) & (cdf.response_type == 'MISS')]
                    if (len(hit_df) > 0) and (len(miss_df) > 0):
                        hit = hit_df.mean_response.values[0]
                        miss = miss_df.mean_response.values[0]
                        hit_miss_SI = (hit - miss) / (hit + miss)
                    else:
                        hit_miss_SI = np.nan

                    fa_df = cdf[(cdf.stim_code == pref_stim) & (cdf.response_type == 'FA')]
                    if (len(fa_df) > 0) and (len(hit_df) > 0):
                        fa = fa_df.mean_response.values[0]
                        hit = hit_df.mean_response.values[0]
                        hit_fa_SI = (hit - fa) / (hit + fa)
                    else:
                        hit_fa_SI = np.nan
            else:
                stim_SI = np.nan
                change_SI = np.nan
                hit_miss_SI = np.nan
                hit_fa_SI = np.nan

            row = [None for col in columns]
            row[columns.index("cell")] = cell
            row[columns.index("max_response")] = max_response
            row[columns.index("pref_stim")] = pref_stim
            if 'response_type' in mdf:
                row[columns.index("pref_response_type")] = pref_response_type
            else:
                row[columns.index("pref_trial_type")] = pref_trial_type
                row[columns.index("pref_response")] = pref_response
                row[columns.index("reliability")] = reliability
            row[columns.index("p_value")] = p_value
            row[columns.index("responsive_conds")] = responsive_conditions
            row[columns.index("suppressed_conds")] = suppressed_conditions
            row[columns.index("sig_conds_thresh")] = sig_conds_thresh
            row[columns.index("sig_conds_pval")] = sig_conds_pval
            row[columns.index("sig_conds_sd")] = sig_conds_sd
            #        row[columns.index("run_p_val")] = run_p_val
            #        row[columns.index("run_modulation")] = run_modulation
            row[columns.index("stim_SI")] = stim_SI
            row[columns.index("change_SI")] = change_SI
            row[columns.index("hit_miss_SI")] = hit_miss_SI
            row[columns.index("hit_fa_SI")] = hit_fa_SI
            df_list.append(row)

        sdf = pd.DataFrame(df_list, columns=columns)
        return sdf




