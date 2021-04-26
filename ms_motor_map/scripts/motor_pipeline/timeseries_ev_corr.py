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
    time_series_df = pd.read_csv(time_series_file, header=None, sep='  ', engine='python')
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
        time_series.plot(kind='line', grid=True, label='time series', style='--', title='r=' + str(corr))
        events_series.plot(label='events')
        plt.savefig(os.path.join(figures_dir, 'corr_IC-' + str(time_series.name) + '.png'))


if __name__ == '__main__':

    project_dir = '/nfs/e4/function_guided_resection/MotorMap'
    nifti_dir = os.path.join(project_dir, 'data', 'bold', 'nifti')
    melodic_dir = os.path.join(project_dir, 'data', 'bold', 'derivatives', 'melodic')

    subjid_list = [sub.replace("sub-", "") for sub in os.listdir(nifti_dir) if "sub-" in sub]
    runid_list = [sub.replace("sub-", "") for sub in os.listdir(nifti_dir) if "sub-" in sub]








    ica_dir = '/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/melodic/sub-01/ses-1/run-1.ica'
    events_file = '/nfs/e4/function_guided_resection/MotorMap/data/bold/nifti/sub-01/ses-1/func/sub-01_ses-1_task-motor_run-01_events.tsv'
    """
        -------------------
        we promise:
        0: fixation
        1: toe
        2: ankle
        3: leftleg
        4: rightleg
        5: forearm
        6: upperarm
        7: wrist
        8: finger
        9: eye
        10: jaw
        11: lip
        12: tongue
    """
    target_ev_list = [3, 4]
    # target_ev_list = [9, 10, 11, 12]
    figures_dir = '/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/melodic/sub-01/ses-1/run-1.ica/task_related_report/leg'
    events_series, time_series_list = prepare_data(ica_dir, events_file, target_ev_list)
    compute_corr(events_series, time_series_list, figures_dir)
