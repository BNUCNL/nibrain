"""
    between ICs(N)
    cardiac mri move nonb resp sinus signal sus unclass unknown
    confounds: global_signal, csf, white matter, framewise_displacement, dvars
"""

import os, re
import pandas as pd


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
    # print(hm_df)
    # hm_df.iloc[:, 0:6].plot()
    # plt.show()
    return hm_df

def read_task(design_file):
    task_df = pd.read_csv(design_file, sep='\t', skiprows=range(0, 5), header=None, usecols=range(0,24,2), names=['Toe', 'Ankle', 'LeftLeg', 'RightLeg', 'Finger', 'Wrist', 'Forearm', 'Upperarm', 'Jaw', 'Lip', 'Tongue', 'Eye'])
    task_df['Head'] = task_df.loc[:, ['Jaw', 'Lip', 'Tongue', 'Eye']].sum(axis=1)
    task_df['Uplimb'] = task_df.loc[:, ['Finger', 'Wrist', 'Forearm', 'Upperarm']].sum(axis=1)
    task_df['Lowlimb'] = task_df.loc[:, ['Toe', 'Ankle', 'LeftLeg', 'RightLeg']].sum(axis=1)
    task_df['All'] = task_df.loc[:,:].sum(axis=1)
    # task_df.iloc[:, 0:-4].plot()
    # task_df.iloc[:, -4:].plot()
    # plt.show()
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
        'wjy': range(43, 49),
        'zyj': range(49, 52),
        'dyx': range(52, 56),
        'lww': range(56, 59),
        'yly': range(65, 69),
        'ms': [1, 2, 3, 4, 59, 60, 61, 62, 63]
    }
    for key, value in rater_dict.items():
        # print(key)
        # print(value)
        if int(subject.replace('sub-', '')) in value:
            return key
            break

def suggest(corr, subject, run, fig_type, rater_name):
    # print(corr)
    if fig_type == 'within_run':
        return 1

    if fig_type == 'confounds':
        trans_x_list = corr[corr.trans_x > 0.253].index.tolist()
        trans_y_list = corr[corr.trans_y > 0.253].index.tolist()
        trans_z_list = corr[corr.trans_z > 0.253].index.tolist()
        rot_x_list = corr[corr.rot_x > 0.253].index.tolist()
        rot_y_list = corr[corr.rot_y > 0.253].index.tolist()
        rot_z_list = corr[corr.rot_z > 0.253].index.tolist()
        suspicious_ic_list = list(set(trans_x_list).union(trans_y_list, trans_z_list, rot_x_list, rot_y_list, rot_z_list))

    if fig_type == 'task_related':
        Toe_list = corr[corr.Toe > 0.253].index.tolist()
        # print('##########Toe_list##########')
        # print(Toe_list)
        Ankle_list = corr[corr.Ankle > 0.253].index.tolist()
        # print('##########Ankle_list##########')
        # print(Ankle_list)
        LeftLeg_list = corr[corr.LeftLeg > 0.253].index.tolist()
        # print('##########LeftLeg_list##########')
        # print(LeftLeg_list)
        RightLeg_list = corr[corr.RightLeg > 0.253].index.tolist()
        # print('##########RightLeg_list##########')
        # print(RightLeg_list)
        Finger_list = corr[corr.Finger > 0.253].index.tolist()
        # print('##########Finger_list##########')
        # print(Finger_list)
        Wrist_list = corr[corr.Wrist > 0.253].index.tolist()
        # print('##########Wrist_list##########')
        # print(Wrist_list)
        Forearm_list = corr[corr.Forearm > 0.253].index.tolist()
        # print('##########Forearm_list##########')
        # print(Forearm_list)
        Upperarm_list = corr[corr.Upperarm > 0.253].index.tolist()
        # print('##########Upperarm_list##########')
        # print(Upperarm_list)
        Jaw_list = corr[corr.Jaw > 0.253].index.tolist()
        # print('##########Jaw_list##########')
        # print(Jaw_list)
        Lip_list = corr[corr.Lip > 0.253].index.tolist()
        # print('##########Lip_list##########')
        # print(Lip_list)
        Tongue_list = corr[corr.Tongue > 0.253].index.tolist()
        # print('##########Tongue_list##########')
        # print(Tongue_list)
        Eye_list = corr[corr.Eye > 0.253].index.tolist()
        # print('##########Eye_list##########')
        # print(Eye_list)
        suspicious_ic_list = list(set(Toe_list).union(Ankle_list, LeftLeg_list, RightLeg_list, Finger_list, Wrist_list, Forearm_list, Upperarm_list, Jaw_list, Lip_list, Tongue_list, Eye_list))
    suspicious_ic_list = [i + 1 for i in suspicious_ic_list]

    # write-in results_suggest.txt file
    ica_dir = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/melodic', subject, 'ses-1', run)
    results_df = pd.read_csv(os.path.join(ica_dir, 'results_' + rater_name + '.txt'), skiprows=[0], skipfooter=1, error_bad_lines=False, sep=', ', engine='python', header=None, index_col=0)
    for suspicious_ic in suspicious_ic_list:
        results_df.loc[suspicious_ic, 1] = 'Signal'
        results_df.loc[suspicious_ic, 2] = 'False'
    for index in results_df.loc[~(results_df[1] == 'Signal')].index.tolist():
        results_df.loc[index, 1] = 'Unclassified Noise'
        results_df.loc[index, 2] = 'True'

    last_line = list(set(list(range(1, results_df.shape[0]+1))) - set(suspicious_ic_list))
    results_df.to_csv(os.path.join(ica_dir, 'results_suggest.csv'), sep=',', header=None)

    with open(os.path.join(ica_dir, 'results_' + rater_name + '.txt'), 'r') as f1:
        r = f1.readlines()
        header_line = r[0]
    with open(os.path.join(ica_dir, 'results_suggest.csv'), 'r') as f2:
        content = f2.readlines()
    with open(os.path.join(ica_dir, 'results_suggest.txt'), 'w+') as f3:
        f3.write(header_line)
        for i in content:
            f3.writelines(i)
        f3.write(str(last_line))
        f3.write('\n')



if __name__ == '__main__':
    fig_type = 'task_related'
    subject_list = [sub for sub in os.listdir('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/melodic') if "sub-" in sub]
    # subject_list = ['sub-04', 'sub-23', 'sub-27', 'sub-46']
    for subject in subject_list:
        print('##########SUBJECT##########')
        print(subject)
        rater_name = rater(subject)
        print('##########RATER##########')
        print(rater_name)
        melodic_dir = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/melodic', subject)
        fmriprep_dir = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/fmriprep', subject)
        design_dir = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/denoise_validation/design_matrix', subject)
        run_list = os.listdir(os.path.join(melodic_dir, 'ses-1'))
        for run in run_list:
            runid = int(re.findall('run-(.+?).ica', run)[0])
            print('##########RUN ID##########')
            print(runid)
            ica_dir = os.path.join(melodic_dir, 'ses-1', run)
            tmodes_file = os.path.join(ica_dir, 'melodic_mix')
            results_file = os.path.join(ica_dir, 'results_' + rater_name + '.txt')
            confounds_file = os.path.join(fmriprep_dir, 'ses-1', 'func', subject + '_ses-1_task-motor_run-' + str(runid) + '_desc-confounds_timeseries.tsv')
            design_file = os.path.join(design_dir, 'run-' + str(runid), 'design.mat')
            # read data
            ic_ts = read_ic_ts(tmodes_file)
            results = read_results(results_file)
            hm_df = read_confounds(confounds_file)
            task_df = read_task(design_file)
            # calculate
            ic_ts_sorted, label_name_list, label_tick_list = sort(ic_ts, results, hm_df, task_df, fig_type)
            corr = ic_corr(ic_ts_sorted).replace(1, 0).abs()
            # extract values
            if fig_type == 'confounds':
                corr = corr.iloc[0:-11, -11:]
            if fig_type == 'task_related':
                corr = corr.iloc[0:-16, -16:]
            # output suggest
            suggest(corr, subject, run, fig_type, rater_name)

