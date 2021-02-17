"""
    compute correlation between IC time series and events series
    create figures of IC time series and events series
"""



import os
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data(ica_dir, events_file, target_ev_list):
    # read data from Tmodes and ev files
    time_series_file = os.path.join(ica_dir, 'melodic_Tmodes')
    time_series_df = pd.read_csv(time_series_file, header=None, sep='  ')
    time_series_df.index = time_series_df.index + 1
    time_series_df.columns = time_series_df.columns + 1
    # print(time_series_df)
    events_df = pd.read_csv(events_file, header=0, sep='\t')
    # print(events_df)

    # prepare events series
    events_list = []
    for trial_type in events_df['trial_type']:
        if trial_type in target_ev_list:
            events_list.extend((1,1,1,1,1,1,1,1))
        else:
            events_list.extend((0,0,0,0,0,0,0,0))
    events_series = pd.Series(events_list, index=range(1,233))

    # prepare time series
    time_series_list = []
    for column in time_series_df.columns:
        time_series = time_series_df[column]
        time_series_list.append(time_series)
    # print(type(time_series_list[0].name))

    return events_series, time_series_list

def compute_corr(events_series, time_series_list, figures_dir):
    # compute correlation
    for time_series in time_series_list:
        corr = time_series.corr(events_series)
        # create figure
        plt.figure()
        time_series.plot(kind='line', grid=True, label='time series', style='--', title='r=' + corr)
        events_series.plot(label='events')
        plt.savefig(os.path.join(figures_dir, 'corr_IC-' + str(time_series.name) + '.png'))


if __name__ == '__main__':
    ica_dir = '/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/melodic/sub-M04_run-4.ica'
    events_file = '/nfs/e4/function_guided_resection/MotorMapping/sub-M04/ses-01/func/sub-M04_ses-01_task-motor_run-4_events.tsv'
    target_ev_list = [3]
    figures_dir = ''
    events_series, time_series_list = prepare_data(ica_dir, events_file, target_ev_list)
    compute_corr(events_series, time_series_list, figures_dir)
