import os
import time
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from matplotlib import pyplot as plt
from cxy_visual_dev.lib.predefine import proj_dir
from nibrain.util.plotfig import auto_bar_width
from magicbox.algorithm.plot import show_bar_value

work_dir = pjoin(proj_dir, 'data/HCP')


def get_subject_info_from_fmriresults01(dataset_name='HCPD'):
    """Get subject information from fmriresults01.txt

    Args:
        dataset_name (str, optional): Project name. Defaults to 'HCPD'.
    """
    # prepare
    src_file = f'/nfs/z1/HCP/{dataset_name}/fmriresults01.txt'
    trg_file = pjoin(work_dir, f'{dataset_name}_SubjInfo.csv')

    # calculate
    df = pd.read_csv(src_file, sep='\t')
    df.drop(labels=[0], axis=0, inplace=True)
    subj_ids = sorted(set(df['src_subject_id']))
    out_dict = {'subID': subj_ids, 'age in months': [],
                'age in years': [], 'gender': []}
    for subj_id in subj_ids:
        idx_vec = df['src_subject_id'] == subj_id
        age = list(set(df.loc[idx_vec, 'interview_age']))
        assert len(age) == 1
        age = int(age[0])
        out_dict['age in months'].append(age)
        age_year = int(age / 12)
        out_dict['age in years'].append(int(age_year))

        gender = list(set(df.loc[idx_vec, 'sex']))
        assert len(gender) == 1
        out_dict['gender'].append(gender[0])

    # save
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def get_subject_info_from_completeness(dataset_name='HCPD'):
    """Get subject information from HCD_LS_2.0_subject_completeness.csv or
    HCA_LS_2.0_subject_completeness.csv.

    已证明这个办法得到的信息和get_subject_info_from_fmriresults01得到的是一样的

    Args:
        dataset_name (str, optional): Project name. Defaults to 'HCPD'.
    """
    # prepare
    name2file = {
        'HCPD': '/nfs/e1/HCPD/HCD_LS_2.0_subject_completeness.csv',
        'HCPA': '/nfs/e1/HCPA/HCA_LS_2.0_subject_completeness.csv'}
    src_file = name2file[dataset_name]
    trg_file = pjoin(work_dir, f'{dataset_name}_SubjInfo_completeness.csv')

    # calculate
    df = pd.read_csv(src_file)
    df.drop(labels=[0], axis=0, inplace=True)
    assert df['src_subject_id'].to_list() == sorted(df['src_subject_id'])
    ages_year = [int(int(age) / 12) for age in df['interview_age']]
    out_dict = {
        'subID': df['src_subject_id'],
        'age in months': df['interview_age'],
        'age in years': ages_year,
        'gender': df['sex']
    }

    # save
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def check_rfMRI_file(subj_par, subj_ids, stem_path, base_path, out_file):
    """
    检查各subject的静息数据文件的缺失，损坏情况
    程序会循环对每个被试做以下操作：
    拼接路径后，搜索所有符合模式的文件，然后检查各文件的状况，并记录于
    被试所在行以“rfMRI dir”命名的列：
    1. 文件不存在：None
    2. 文件存在，但是rfMRI file的开头和rfMRI dir不匹配：rfMRI file
    3. 文件存在，读取数据出错：error
    4. 文件存在，且能读取数据：ok="data.shape"

    注意！！！如果同一个“rfMRI dir”下出现多个匹配的文件（按理来说只有一个）,
    请手动检查，并设计一个更专用的"base_path" pattern

    Args:
        subj_par (str): 被试父目录
        subj_ids (sequence): subject IDs
            a sequence of strings
        stem_path (str): path from 'subj_par' to 'base_path'
        base_path (str): path from rfMRI dir to file
            用于匹配stem path下想要的rfMRI文件的pattern
        out_file (str): TSV file
            除了‘subID’列存的是被试的ID之外，其它列名都是至少会在一个被试中出现的rfMRI dir
    """
    assert out_file.endswith('.tsv')

    df = pd.DataFrame()
    df['subID'] = subj_ids
    n_subj = df.shape[0]
    for subj_idx, idx in enumerate(df.index, 1):
        time1 = time.time()

        subj_id = df.loc[idx, 'subID']
        stem_dir = pjoin(subj_par, subj_id, stem_path)
        n_stem = len(stem_dir)+1
        fpath_ = pjoin(stem_dir, base_path)
        for fpath in glob.iglob(fpath_):
            base_dir = os.path.dirname(fpath)[n_stem:]
            base_file = os.path.basename(fpath)

            if (base_dir in df.columns) and \
               (not np.isnan(df.loc[idx, base_dir])):
                raise RuntimeError(
                    'More than one matched files are '
                    f'found under sub-{subj_id}_baseDir-{base_dir}!'
                )

            if not base_file.startswith(base_dir):
                df.loc[idx, base_dir] = base_file
                continue

            try:
                data = nib.load(fpath).get_fdata()
            except Exception:
                df.loc[idx, base_dir] = 'error'
                continue
            df.loc[idx, base_dir] = f'ok={data.shape}'

        print(f'Finish subj-{subj_idx}/{n_subj}, '
              f'cost {time.time()-time1} seconds')
    df.to_csv(out_file, index=False, sep='\t')


def rfMRI_file_status(fpath):
    """
    统计“check_rfMRI_file”得到的文件中的数据

    Args:
        fpath ([type]): [description]
    """
    df = pd.read_csv(fpath, sep='\t')
    print('total number of subjects:', df.shape[0])
    _, axes = plt.subplots(2, 1)
    ax1, ax2 = axes

    # plot figure1
    xlabel1s = df.columns.to_list()
    xlabel1s.remove('subID')
    n_xlabel1 = len(xlabel1s)
    x1 = np.arange(n_xlabel1)

    item1s = list(set(np.array(df[xlabel1s]).ravel()))
    item1s.remove(np.nan)
    n_item1 = len(item1s)

    width1 = auto_bar_width(x1, n_item1)
    offset1 = -(n_item1 - 1) / 2
    for item1 in item1s:
        y1 = np.zeros(n_xlabel1, int)
        for xlabel1_idx, xlabel1 in enumerate(xlabel1s):
            y1[xlabel1_idx] = np.sum(df[xlabel1] == item1)
        rects1 = ax1.bar(x1+width1*offset1, y1, width1, label=item1)
        show_bar_value(rects1, ax=ax1)
        offset1 += 1
    ax1.legend()
    ax1.set_xticks(x1)
    ax1.set_xticklabels([f'{k}-{v}' for k, v in enumerate(xlabel1s, 1)])
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax1.set_ylabel('#subjects')
    ax1.set_title(os.path.basename(fpath))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # plot figure2
    xlabel2s = []
    for idx in df.index:
        xlabel2 = ''
        for xlabel1_idx, xlabel1 in enumerate(xlabel1s, 1):
            value = df.loc[idx, xlabel1]
            if isinstance(value, str) and value.startswith('ok'):
                if xlabel2 == '':
                    xlabel2 = str(xlabel1_idx)
                else:
                    xlabel2 = f'{xlabel2}+{xlabel1_idx}'
        xlabel2s.append(xlabel2)
    xlabel2s_uniq = np.unique(xlabel2s)
    n_xlabel2 = len(xlabel2s_uniq)
    x2 = np.arange(n_xlabel2)
    width2 = auto_bar_width(x2)
    y2 = np.zeros(n_xlabel2, int)
    for xlabel2_idx, xlabel2 in enumerate(xlabel2s_uniq):
        y2[xlabel2_idx] = xlabel2s.count(xlabel2)
    rects2 = ax2.bar(x2, y2, width2)
    show_bar_value(rects2, ax=ax2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(xlabel2s_uniq)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax2.set_ylabel('#subjects')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def get_subject_info_for_HCPY():
    """
    从1071个Glasser认为具有valid MSMAll的被试中选出
    至少包含一个状态为ok=(1200, 91282)的静息run的被试，
    并证明这1071个被试都包含于S1200 GroupAvg发布的morphology数据的
    1096名被试里。

    得到HCPY_SubjInfo.csv:
        存放被试号，年龄，性别，以及在1096个被试中的索引
    """
    fpath_1206 = '/nfs/z1/HCP/HCPYA/S1200_behavior_restricted.csv'
    fpath_1096 = pjoin(work_dir, 'subject_id_1096')
    fpath_1071 = pjoin(work_dir, 'subject_id_1071')
    fpath_check = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
    out_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')

    df_1206 = pd.read_csv(fpath_1206, index_col='Subject')
    sids_1096 = [int(i) for i in open(fpath_1096).read().splitlines()]
    sids_1071 = [int(i) for i in open(fpath_1071).read().splitlines()]
    df_check = pd.read_csv(fpath_check, sep='\t', index_col='subID')
    assert set(sids_1071).issubset(sids_1096)

    sids = [i for i in sids_1071
            if np.any(df_check.loc[i] == 'ok=(1200, 91282)')]
    s_indices = [sids_1096.index(sid) for sid in sids]
    assert sorted(s_indices) == s_indices

    out_dict = {
        'subID': sids,
        'age in years': df_1206.loc[sids, 'Age_in_Yrs'],
        'gender': df_1206.loc[sids, 'Gender'],
        '1096_idx': s_indices
    }
    pd.DataFrame(out_dict).to_csv(out_file, index=False)


def summary_subj_info(data_flag):
    """
    获取各类数据所涉及的被试信息

    Args:
        data_flag (str):
    """
    if data_flag in ('HCPY-myelin', 'HCPY-thickness'):
        fpath = pjoin(work_dir, 'HCPY_SubjInfo.csv')
        df = pd.read_csv(fpath)
        print('#Subject:', df.shape[0])
        print('#Male:', np.sum(df['gender'] == 'M'))
        print('#Female:', np.sum(df['gender'] == 'F'))
        print('Mean age:', np.mean(df['age in years']))
        print('Age std:', np.std(df['age in years']))
        print(f"Age range: {np.min(df['age in years'])} to "
              f"{np.max(df['age in years'])}")
    elif data_flag in ('HCPD-myelin', 'HCPD-thickness', 'HCPA-myelin', 'HCPA-thickness'):
        dataset_name = data_flag.split('-')[0]
        fpath = pjoin(work_dir, f'{dataset_name}_SubjInfo.csv')
        df = pd.read_csv(fpath)
        print('#Subject:', df.shape[0])
        print('#Male:', np.sum(df['gender'] == 'M'))
        print('#Female:', np.sum(df['gender'] == 'F'))
        print('Mean age:', np.mean(df['age in months']) / 12)
        print('Age std:', np.std(df['age in months']) / 12)
        print(f"Age range: {np.min(df['age in months']) / 12} to "
              f"{np.max(df['age in months']) / 12}")
    elif data_flag == 'HCPY-rfMRI':
        fpath = pjoin(work_dir, 'HCPY_SubjInfo.csv')
        check_file = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
        runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
                'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
        df = pd.read_csv(fpath)
        df_ck = pd.read_csv(check_file, sep='\t')
        subj_ids_1206 = df_ck['subID'].to_list()
        ok_idx_vec = np.all(df_ck[runs] == 'ok=(1200, 91282)', 1)
        indices = []
        for idx in df.index:
            subj_id = df.loc[idx, 'subID']
            subj_idx_1206 = subj_ids_1206.index(subj_id)
            if ok_idx_vec[subj_idx_1206]:
                indices.append(idx)
        df = df.loc[indices]
        print('#Subject:', df.shape[0])
        print('#Male:', np.sum(df['gender'] == 'M'))
        print('#Female:', np.sum(df['gender'] == 'F'))
        print('Mean age:', np.mean(df['age in years']))
        print('Age std:', np.std(df['age in years']))
        print(f"Age range: {np.min(df['age in years'])} to "
              f"{np.max(df['age in years'])}")
    else:
        raise ValueError('not supported data_flag:', data_flag)


if __name__ == '__main__':
    # >>>check_rfMRI_file
    # subj_par = '/nfs/z1/HCP/HCPYA'
    # df = pd.read_csv(pjoin(subj_par, 'S1200_behavior.csv'), usecols=['Subject'])
    # subj_ids = [str(i) for i in df['Subject']]
    # check_rfMRI_file(
    #     subj_par=subj_par, subj_ids=subj_ids,
    #     stem_path='MNINonLinear/Results',
    #     base_path='rfMRI_REST*/'
    #               'rfMRI_REST*_Atlas_MSMAll_hp2000_clean.dtseries.nii',
    #     out_file=pjoin(work_dir, 'HCPY_rfMRI_file_check_z1.tsv')
    # )

    # subj_par = '/nfs/e1/HCPD/fmriresults01'
    # df = pd.read_csv('/nfs/e1/HCPD/HCPD_SubjInfo.csv')
    # subj_ids = [i + '_V1_MR' for i in df['subID']]
    # check_rfMRI_file(
    #     subj_par=subj_par, subj_ids=subj_ids,
    #     stem_path='MNINonLinear/Results',
    #     base_path='rfMRI_REST*/'
    #               'rfMRI_REST*_Atlas_MSMAll_hp0_clean.dtseries.nii',
    #     out_file=pjoin(work_dir, 'HCPD_rfMRI_file_check.tsv')
    # )

    # subj_par = '/nfs/e1/HCPA/fmriresults01'
    # df = pd.read_csv('/nfs/e1/HCPA/HCPA_SubjInfo.csv')
    # subj_ids = [i + '_V1_MR' for i in df['subID']]
    # check_rfMRI_file(
    #     subj_par=subj_par, subj_ids=subj_ids,
    #     stem_path='MNINonLinear/Results',
    #     base_path='rfMRI_REST*/'
    #               'rfMRI_REST*_Atlas_MSMAll_hp0_clean.dtseries.nii',
    #     out_file=pjoin(work_dir, 'HCPA_rfMRI_file_check.tsv')
    # )

    # subj_par = '/nfs/z1/HCP/HCPD/fmriresults01'
    # df = pd.read_csv('/nfs/z1/HCP/HCPD/HCPD_SubjInfo.csv')
    # subj_ids = [i + '_V1_MR' for i in df['subID']]
    # check_rfMRI_file(
    #     subj_par=subj_par, subj_ids=subj_ids,
    #     stem_path='MNINonLinear/Results',
    #     base_path='rfMRI_REST*/'
    #               'rfMRI_REST*_Atlas_MSMAll_hp0_clean.dtseries.nii',
    #     out_file=pjoin(work_dir, 'HCPD_rfMRI_file_check-z1.tsv')
    # )

    # subj_par = '/nfs/z1/HCP/HCPA/fmriresults01'
    # df = pd.read_csv('/nfs/z1/HCP/HCPA/HCPA_SubjInfo.csv')
    # subj_ids = [i + '_V1_MR' for i in df['subID']]
    # check_rfMRI_file(
    #     subj_par=subj_par, subj_ids=subj_ids,
    #     stem_path='MNINonLinear/Results',
    #     base_path='rfMRI_REST*/'
    #               'rfMRI_REST*_Atlas_MSMAll_hp0_clean.dtseries.nii',
    #     out_file=pjoin(work_dir, 'HCPA_rfMRI_file_check-z1.tsv')
    # )

    # HCPA_rfMRI_file_check.tsv和HCPA_rfMRI_file_check-z1.tsv只是列的顺序不一样
    # HCPD_rfMRI_file_check.tsv和HCPD_rfMRI_file_check-z1.tsv只是列的顺序不一样
    # check_rfMRI_file<<<

    # >>>rfMRI_file_status
    # rfMRI_file_status(pjoin(work_dir, 'HCPD_rfMRI_file_check.tsv'))
    # rfMRI_file_status(pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv'))
    # rfMRI_file_status(pjoin(work_dir, 'HCPA_rfMRI_file_check.tsv'))
    # rfMRI_file_status<<<

    # get_subject_info_from_fmriresults01('HCPD')
    # get_subject_info_from_completeness('HCPD')
    # get_subject_info_from_fmriresults01('HCPA')
    # get_subject_info_from_completeness('HCPA')
    # get_subject_info_for_HCPY()

    summary_subj_info(data_flag='HCPY-myelin')
