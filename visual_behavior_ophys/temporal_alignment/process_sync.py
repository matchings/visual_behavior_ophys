import numpy as np
from visual_behavior_ophys.temporal_alignment.sync_dataset import Dataset


def filter_digital(rising, falling, threshold=0.0001):
    """
    Removes short transients from digital signal.

    Rising and falling should be same length and units
        in seconds.

    Kwargs:
        threshold (float): transient width
    """
    # forwards (removes low-to-high transients)
    dif_f = falling - rising
    falling_f = falling[np.abs(dif_f) > threshold]
    rising_f = rising[np.abs(dif_f) > threshold]
    # backwards (removes high-to-low transients )
    dif_b = rising_f[1:] - falling_f[:-1]
    dif_br = np.append([threshold * 2], dif_b)
    dif_bf = np.append(dif_b, [threshold * 2])
    rising_f = rising_f[np.abs(dif_br) > threshold]
    falling_f = falling_f[np.abs(dif_bf) > threshold]

    return rising_f, falling_f


def process_sync_sutter(sync_path,pkl):

    d = Dataset(sync_path)
    sync_data = d
    meta_data = d.meta_data
    sample_freq = meta_data['ni_daq']['counter_output_freq']
    # 2P vsyncs
    vs2p_r = d.get_rising_edges('vsync_2p')
    # vs2p_f = d.get_falling_edges('vsync_2p')
    # Convert to seconds
    vs2p_rsec = vs2p_r / sample_freq
    # vs2p_fsec = vs2p_f/sample_freq
    frames_2p = vs2p_rsec
    print("Total 2P frames: %s" % len(frames_2p))
    # stimulus vsyncs
    vs_r = d.get_rising_edges('vsync_stim')
    vs_f = d.get_falling_edges('vsync_stim')
    # convert to seconds
    vs_r_sec = vs_r / sample_freq
    vs_f_sec = vs_f / sample_freq
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
        # threshold 0.0018 works for recent DoC but gives problems for FS
        # threshold 0.0001 is the original and works for FS
        threshold = 0.001
        print 'filter digital threshold', threshold
        #                vs_r_sec_f, vs_f_sec_f = filter_digital(vs_r_sec, vs_f_sec,threshold=0.0018)
        vs_r_sec_f, vs_f_sec_f = filter_digital(vs_r_sec, vs_f_sec, threshold)
        print 'Spurious vsyncs:', str(len(vs_f_sec) - len(vs_f_sec_f))
    else:
        vs_r_sec_f = vs_r_sec
        vs_f_sec_f = vs_f_sec
    if vs_r_sec_f[1] - vs_r_sec_f[0] > 0.2:
        vsyncs = vs_f_sec_f[1:]
    else:
        vsyncs = vs_f_sec_f
    print("Actual vsync frames: %s" % len(vsyncs))
    # add lick data
    lick_0 = d.get_rising_edges('lick0') / sample_freq
    lick_1 = d.get_rising_edges('lick1') / sample_freq
    # put sync data in dphys format to be compatible with downstream analysis
    times_2p = {'timestamps': frames_2p}
    times_vsync = {'timestamps': vsyncs}
    times_lick0 = {'timestamps': lick_0}
    times_lick1 = {'timestamps': lick_1}
    sync = {'2PFrames': times_2p,
            'visualFrames': times_vsync,
            'lickTimes_0': times_lick0,
            'lickTimes_1': times_lick1}
    print 'pkl vsyncs:', pkl['vsynccount']

    return sync, sync_data


def process_sync_scientifica(sync_path,pkl):

    d = Dataset(sync_path)
    sync_data = d
    meta_data = d.meta_data
    sample_freq = meta_data['ni_daq']['counter_output_freq']
    # 2P vsyncs
    vs2p_r = d.get_rising_edges('vsync_2p')
    # vs2p_f = d.get_falling_edges('vsync_2p')
    # Convert to seconds
    vs2p_rsec = vs2p_r / sample_freq
    # vs2p_fsec = vs2p_f/sample_freq
    frames_2p = vs2p_rsec
    print("Total 2P frames: %s" % len(frames_2p))
    # stimulus vsyncs
    vs_r = d.get_rising_edges('vsync_stim')
    vs_f = d.get_falling_edges('vsync_stim')
    # convert to seconds
    vs_r_sec = vs_r / sample_freq
    vs_f_sec = vs_f / sample_freq
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
        # threshold 0.0018 works for recent DoC but gives problems for FS
        # threshold 0.0001 is the original and works for FS
        threshold = 0.001
        print 'filter digital threshold', threshold
        #                vs_r_sec_f, vs_f_sec_f = filter_digital(vs_r_sec, vs_f_sec,threshold=0.0018)
        vs_r_sec_f, vs_f_sec_f = filter_digital(vs_r_sec, vs_f_sec, threshold)
        print 'Spurious vsyncs:', str(len(vs_f_sec) - len(vs_f_sec_f))
    else:
        vs_r_sec_f = vs_r_sec
        vs_f_sec_f = vs_f_sec
    if vs_r_sec_f[1] - vs_r_sec_f[0] > 0.2:
        vsyncs = vs_f_sec_f[1:]
    else:
        vsyncs = vs_f_sec_f
    print("Actual vsync frames: %s" % len(vsyncs))
    # add lick data
    lick_0 = d.get_rising_edges('lick0') / sample_freq
    lick_1 = d.get_rising_edges('lick1') / sample_freq
    # put sync data in dphys format to be compatible with downstream analysis
    times_2p = {'timestamps': frames_2p}
    times_vsync = {'timestamps': vsyncs}
    times_lick0 = {'timestamps': lick_0}
    times_lick1 = {'timestamps': lick_1}
    sync = {'2PFrames': times_2p,
            'visualFrames': times_vsync,
            'lickTimes_0': times_lick0,
            'lickTimes_1': times_lick1}
    print 'pkl vsyncs:', pkl['vsynccount']

    return sync, sync_data
