"""
    compute correlation between IC time series and EVs
"""



import os
import pandas as pd
import matplotlib.pyplot as plt

def read_data(time_series_file, events_file, ic_id):
    # read data from time series files and ev files
    time_series_df= pd.read_csv(time_series_file, header=None)
    events_df = pd.read_csv(events_file, header=0, sep='\t')
    # print(time_series)
    # print(events_df)
    events_list = []
    for trial_type in events_df['trial_type']:
        if trial_type == 3 or trial_type == 4:
            events_list.extend((1,1,1,1,1,1,1,1))
        else:
            events_list.extend((0,0,0,0,0,0,0,0))
    events_series = pd.Series(events_list, index=range(1,233))
    # print(events_series)
    time_series = time_series_df[0]
    time_series.index = range(1,233)
    # print(time_series)
    return events_series, time_series

def compute_corr(events_series, time_series):
    # compute correlation
    corr = time_series.corr(events_series)
    # print(corr)
    return corr

def figure_output(events_series, time_series, corr):
    # create figures about time series and ev series
    plt.figure()
    time_series.plot(kind='line', grid=True, label='S1', style='--', title='r=' + corr)
    events_series.plot(label='S2')
    plt.savefig(os.path.join('/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/melodic/sub-M04_run-4.ica/corr-fig', 'corr-' + str(ic_id) + '.png'))

if __name__ == '__main__':
    for ic_id in range(1,105):
        print('IC-' + str(ic_id) + ':')
        time_series_file = os.path.join('/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/melodic/sub-M04_run-4.ica/report','t' + str(ic_id) + '.txt')
        events_file = '/nfs/e4/function_guided_resection/MotorMapping/sub-M04/ses-01/func/sub-M04_ses-01_task-motor_run-4_events.tsv'
        events_series, time_series = read_data(time_series_file, events_file, ic_id)
        corr = compute_corr(events_series, time_series)
        figure_output(events_series, time_series, corr)