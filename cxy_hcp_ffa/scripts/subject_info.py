import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from cxy_hcp_ffa.lib.predefine import proj_dir

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'subj_info')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def basic_info(subj_file):

    subj_ids = [int(i) for i in open(subj_file).read().splitlines()]
    info1_file = '/nfs/m1/hcp/S1200_behavior_restricted.csv'
    info1_df = pd.read_csv(info1_file)

    valid_indices = []
    for idx in info1_df.index:
        if info1_df.loc[idx, 'Subject'] in subj_ids:
            valid_indices.append(idx)
    info1_df = info1_df.loc[valid_indices]
    assert info1_df['Subject'].to_list() == subj_ids

    print('#Male:', np.sum(info1_df['Gender'] == 'M'))
    print('#Female:', np.sum(info1_df['Gender'] == 'F'))
    print('Mean age:', np.mean(info1_df['Age_in_Yrs']))
    print('Age std:', np.std(info1_df['Age_in_Yrs']))
    print(f"Age range: {np.min(info1_df['Age_in_Yrs'])} to "
        f"{np.max(info1_df['Age_in_Yrs'])}")


def make_subj_idx_vec(src_file, out_fname):
    """
    用src_file中的被试号与1080个被试做交集
    得到两个文件：
    1. out_fname.npy: 长度为1080的bool向量，用True标记在交集中的被试
    2. out_fname.txt: 存放交集中的被试号
    """
    subj_ids_1080_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    out_file1 = pjoin(work_dir, f'{out_fname}.npy')
    out_file2 = pjoin(work_dir, f'{out_fname}.txt')

    subj_ids_src = open(src_file).read().splitlines()
    subj_ids_1080 = open(subj_ids_1080_file).read().splitlines()

    subj_ids = set(subj_ids_1080).intersection(subj_ids_src)
    subj_ids = sorted(subj_ids)

    subj_idx_vec = np.zeros(len(subj_ids_1080), bool)
    for sid in subj_ids:
        idx = subj_ids_1080.index(sid)
        subj_idx_vec[idx] = True
    np.save(out_file1, subj_idx_vec)
    open(out_file2, 'w').write('\n'.join(subj_ids))


def screen_subj1():
    """
    用1071个有valid MSMAll的被试与原1080个被试做交集
    并只保留拥有至少一个'ok=(1200, 91282)'的静息run的被试
    得到两个文件：
    1. subject_id1.npy: 长度为1080的bool向量，用True标记选中的被试
    2. subject_id1.txt: 存放选中的被试号
    """
    runs = ['rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR', 'rfMRI_REST2_LR']
    subj_ids_1080_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    subj_ids_1071_file = pjoin(proj_dir, 'data/HCP/subject_id_1071')
    rfMRI_check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    out_file1 = pjoin(work_dir, 'subject_id1.npy')
    out_file2 = pjoin(work_dir, 'subject_id1.txt')

    subj_ids_1080 = open(subj_ids_1080_file).read().splitlines()
    subj_ids_1071 = open(subj_ids_1071_file).read().splitlines()
    df = pd.read_csv(rfMRI_check_file, sep='\t')
    subj_ids_1206 = df['subID'].to_list()
    row_indices = [subj_ids_1206.index(int(i)) for i in subj_ids_1080]
    df = df.loc[row_indices]
    run_idx_vec = np.any(np.array(df[runs] == 'ok=(1200, 91282)'), 1)

    subj_ids = []
    subj_idx_vec = np.zeros(len(subj_ids_1080), bool)
    for sidx, sid in enumerate(subj_ids_1080):
        if run_idx_vec[sidx] and (sid in subj_ids_1071):
            subj_ids.append(sid)
            subj_idx_vec[sidx] = True

    np.save(out_file1, subj_idx_vec)
    open(out_file2, 'w').write('\n'.join(subj_ids))


def screen_subj2():
    """
    只保留subject_id1.txt中拥有4个'ok=(1200, 91282)'的静息run的被试
    得到文件subject_id2.txt存放选中的被试号
    """
    runs = ['rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR', 'rfMRI_REST2_LR']
    subj_ids_1053_file = pjoin(work_dir, 'subject_id1.txt')
    rfMRI_check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    out_file = pjoin(work_dir, 'subject_id2.txt')

    subj_ids_1053 = open(subj_ids_1053_file).read().splitlines()
    df = pd.read_csv(rfMRI_check_file, sep='\t')
    subj_ids_1206 = df['subID'].to_list()
    row_indices = [subj_ids_1206.index(int(i)) for i in subj_ids_1053]
    df = df.loc[row_indices]
    run_idx_vec = np.all(np.array(df[runs] == 'ok=(1200, 91282)'), 1)

    subj_ids = []
    for sidx, sid in enumerate(subj_ids_1053):
        if run_idx_vec[sidx]:
            subj_ids.append(sid)

    open(out_file, 'w').write('\n'.join(subj_ids))


if __name__ == '__main__':
    # basic_info(subj_file=pjoin(proj_dir, 'analysis/s2/subject_id'))
    # make_subj_idx_vec(
    #     src_file=pjoin(proj_dir, 'data/HCP/subject_id_1071'),
    #     out_fname='subject_id_MSMAll')
    # basic_info(subj_file=pjoin(work_dir, 'subject_id_MSMAll.txt'))
    # screen_subj1()
    # basic_info(subj_file=pjoin(work_dir, 'subject_id1.txt'))
    screen_subj2()
