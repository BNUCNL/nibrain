"""
    between ICs(N)
    cardiac mri move nonb resp sinus signal sus unclass unknown
    confounds: global_signal, csf, white matter, framewise_displacement, dvars
"""

import os, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)

# read data
def read_ic_ts(mix_file):
    ts_df = pd.read_csv(mix_file, header=None, sep=' ', engine='python')
    ts_df = ts_df.T
    return ts_df

def read_results(results_file):
    results_list = []
    with open(results_file) as f:
        line_list = f.readlines()
        del(line_list[0])
        del(line_list[-1])
        for line in line_list:
            result = re.findall(', (.+?),', line)[0]
            results_list.append(result)
    return results_list

def read_confounds(confounds_file):
    hm_df = pd.read_csv(confounds_file, sep='\t')[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'global_signal', 'csf', 'white_matter', 'framewise_displacement', 'dvars']]
    return hm_df

def read_task(design_file):
    task_df = pd.read_csv(design_file, sep='\t', skiprows=range(0, 5), header=None, usecols=range(0,24,2), names=['Toe', 'Ankle', 'LeftLeg', 'RightLeg', 'Finger', 'Wrist', 'Forearm', 'Upperarm', 'Jaw', 'Lip', 'Tongue', 'Eye'])
    task_df['Head'] = task_df.loc[:, ['Jaw', 'Lip', 'Tongue', 'Eye']].abs().sum(axis=1)
    task_df['Uplimb'] = task_df.loc[:, ['Finger', 'Wrist', 'Forearm', 'Upperarm']].sum(axis=1)
    task_df['Lowlimb'] = task_df.loc[:, ['Toe', 'Ankle', 'LeftLeg', 'RightLeg']].sum(axis=1)
    task_df['All'] = task_df.loc[:,:].sum(axis=1)
    task_df.iloc[:, 0:-4].plot()
    # task_df.iloc[:, -4:].plot()
    plt.show()
    return task_df

# sorting
def sort(ic_ts, results, hm_df, task_df, fig_type):
    # sort classifications, confounds and task
    ic_ts['class'] = results
    ic_ts_sorted = ic_ts.sort_values(by=['class'])
    label_list_orig = ic_ts_sorted['class'].tolist()
    label_num_list = []
    # classifications name
    label_name_list = sorted(set(label_list_orig), key=label_list_orig.index)
    # classifications num
    for label_name in label_name_list:
        label_num_list.append(label_list_orig.count(label_name))
    if fig_type == 'confounds':
        label_name_list.extend(hm_df.columns.values.tolist())
    if fig_type == 'task_related':
        label_name_list.extend(task_df.columns.values.tolist())
    label_tick_list = []
    for index, num in enumerate(label_num_list):
        tick = sum(label_num_list[0:index])
        label_tick_list.append(tick)
    # extend datafram (structure: ICs, confounds, task)
    if fig_type == 'within_run':
        ic_ts_sorted = ic_ts_sorted.T.drop(labels=['class']).apply(lambda x: x.astype(float))
    if fig_type == 'confounds':
        ic_ts_sorted = ic_ts_sorted.T.drop(labels=['class']).join(hm_df).apply(lambda x: x.astype(float))
    if fig_type == 'task_related':
        ic_ts_sorted = ic_ts_sorted.T.drop(labels=['class']).join(task_df).apply(lambda x: x.astype(float))

    # print('*****label_name_list111*****')
    # print(label_name_list)

    # rename some labels
    for index, name in enumerate(label_name_list):
        if name == 'Movement':
            label_name_list[index] = 'Move'
        if name == 'Respiratory':
            label_name_list[index] = 'Resp'
        if name == 'Sagittal sinus':
            label_name_list[index] = 'Sinus'
        if name == 'Susceptability-motion':
            label_name_list[index] = 'Sus-move'
        if name == 'Unclassified Noise':
            label_name_list[index] = 'Unclass'
        if name == 'Non-brain':
            label_name_list[index] = 'Non-b'
        if name == 'global_signal':
            label_name_list[index] = 'signal'
        if name == 'white_matter':
            label_name_list[index] = 'WM'
        if name == 'framewise_displacement':
            label_name_list[index] = 'FD'

    # print('*****ic_ts_sorted*****')
    # print(ic_ts_sorted)
    # print('*****label_name_list*****')
    # print(label_name_list)
    # print('*****label_tick_list*****')
    # print(label_tick_list)

    return ic_ts_sorted, label_name_list, label_tick_list

# calculate corr
def ic_corr(data):
    ic_corr = data.corr()
    return ic_corr

def rater(subject):
    rater_dict = {
        'shd': range(5, 8),
        'ld': range(8, 12),
        'cxy': range(12, 17),
        'zm': range(17, 22),
        'zxr': range(22, 25),
        'gzx': range(25, 30),
        'wys': range(30, 34),
        'dgy': range(34, 38),
        'lmx': range(38, 43),
        'zyj': [],
        'wjy': [],
        'dyx': range(52, 56),
        'lww': range(56, 59),
        'yly': range(65, 69)
    }
    for key, value in rater_dict.items():
        if int(subject.replace('sub-', '')) in value:
            return key
        else:
            return 'ms'

def init_output_dir(subject, fig_type):
    # fig type
    if fig_type == 'within_run':
        save_dir = os.path.join('/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/denoise_validation/within_run', subject)
    if fig_type == 'confounds':
        save_dir = os.path.join('/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/denoise_validation/confounds', subject)
    if fig_type == 'task_related':
        save_dir = os.path.join('/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/denoise_validation/task_related', subject)
    # initiate output dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'suggest.txt')):
        print('#################################DEBUG#############################################')
        with open(os.path.join(save_dir, 'suggest.txt'), 'w') as f:
            f.write(subject)
    return save_dir

def save_fig(fig_type, save_dir):
    if fig_type == 'within_run':
        plt.savefig(os.path.join(save_dir, 'within_run_' + rater_name + '.png'))
    if fig_type == 'confounds':
        plt.savefig(os.path.join(save_dir, 'confounds_' + rater_name + '.png'))
    if fig_type == 'task_related':
        plt.savefig(os.path.join(save_dir, 'task_' + rater_name + '.png'))

def suggest(corr, run, fig_type, save_dir):
    return '1'




if __name__ == '__main__':
    fig_type_list = ['within_run', 'confounds', 'task_related']
    for fig_type in fig_type_list:
        print('##########FIG_TYPE##########')
        print(fig_type)
        subject_list = ['sub-01']
        for subject in subject_list:
            print('##########SUBJECT##########')
            print(subject)
            rater_name = rater(subject)
            melodic_dir = os.path.join('/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/melodic', subject)
            fmriprep_dir = os.path.join('/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/fmriprep', subject)
            design_dir = os.path.join('/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/denoise_validation/design_matrix', subject)
            run_list = os.listdir(os.path.join(melodic_dir, 'ses-1'))
            # run_list = ['sub-01_ses-1_task-motor_run-1.ica']
            fig, ax = plt.subplots(2, 3, figsize=(20, 10))
            plt.subplots_adjust(top=0.9, bottom = 0.1, wspace=0.3, hspace=0.3)
            for run in run_list:
                runid = int(re.findall('run-(.+?).ica', run)[0])
                print('##########RUN ID##########')
                print(runid)
                ica_dir = os.path.join(melodic_dir, 'ses-1', run)
                tmodes_file = os.path.join(ica_dir, 'melodic_mix')
                results_file = os.path.join(ica_dir, 'results_' + rater_name + '.txt')
                confounds_file = os.path.join(fmriprep_dir, 'ses-1', 'func', subject + '_ses-1_task-motor_run-' + str(runid) + '_desc-confounds_timeseries.tsv')
                design_file = os.path.join(design_dir, 'ses-1', 'run-' + str(runid), 'design.mat')
                # read data
                ic_ts = read_ic_ts(tmodes_file)
                results = read_results(results_file)
                hm_df = read_confounds(confounds_file)
                task_df = read_task(design_file)
                # calculate
                ic_ts_sorted, label_name_list, label_tick_list = sort(ic_ts, results, hm_df, task_df, fig_type)
                corr = ic_corr(ic_ts_sorted).replace(1, 0)
                # extract values
                if fig_type == 'confounds':
                    corr = corr.iloc[0:-11, -11:]
                if fig_type == 'task_related':
                    corr = corr.iloc[0:-16, -16:]

                # set up subplot ticks
                if runid < 4:
                    sns.heatmap(abs(corr), cmap='RdBu_r', annot=False, ax=ax[0][runid - 1])
                    ax[0][runid - 1].set_title(run.replace('.ica', '') + '_' + rater_name)
                    if fig_type == 'within_run':
                        ax[0][runid - 1].set_xticks(label_tick_list)
                        ax[0][runid - 1].set_yticks(label_tick_list)
                        ax[0][runid - 1].set_xticklabels(label_name_list, fontsize='small')
                        ax[0][runid - 1].set_yticklabels(label_name_list, fontsize='small')
                    if fig_type == 'confounds':
                        ax[0][runid - 1].set_yticks(label_tick_list)
                        ax[0][runid - 1].set_xticklabels(label_name_list[-11:], fontsize='small')
                        ax[0][runid - 1].set_yticklabels(label_name_list[0:-11], fontsize='small')
                    if fig_type == 'task_related':
                        ax[0][runid - 1].set_yticks(label_tick_list)
                        ax[0][runid - 1].set_xticklabels(label_name_list[-16:], fontsize='small')
                        ax[0][runid - 1].set_yticklabels(label_name_list[0:-16], fontsize='small')
                else:
                    sns.heatmap(abs(corr), cmap='RdBu_r', annot=False, ax=ax[1][runid - 4])
                    ax[1][runid - 4].set_title(run.replace('.ica', '') + '_' + rater_name)
                    if fig_type == 'within_run':
                        ax[1][runid - 4].set_xticks(label_tick_list)
                        ax[1][runid - 4].set_yticks(label_tick_list)
                        ax[1][runid - 4].set_xticklabels(label_name_list, fontsize='small')
                        ax[1][runid - 4].set_yticklabels(label_name_list, fontsize='small')
                    if fig_type == 'confounds':
                        ax[1][runid - 4].set_yticks(label_tick_list)
                        ax[1][runid - 4].set_xticklabels(label_name_list[-11:], fontsize='small')
                        ax[1][runid - 4].set_yticklabels(label_name_list[0:-11], fontsize='small')
                    if fig_type == 'task_related':
                        ax[1][runid - 4].set_yticks(label_tick_list)
                        ax[1][runid - 4].set_xticklabels(label_name_list[-16:], fontsize='small')
                        ax[1][runid - 4].set_yticklabels(label_name_list[0:-16], fontsize='small')
                save_dir = init_output_dir(subject, fig_type)
                # output suggest
                suggest(corr, run, fig_type, save_dir)
            # fig save
            save_fig(fig_type, save_dir)

