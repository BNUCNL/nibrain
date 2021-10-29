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


if __name__ == '__main__':
    # >>>check_rfMRI_file
    # subj_par = '/nfs/m1/hcp'
    # df = pd.read_csv(pjoin(subj_par, 'S1200_behavior.csv'), usecols=['Subject'])
    # subj_ids = [str(i) for i in df['Subject']]
    # check_rfMRI_file(
    #     subj_par=subj_par, subj_ids=subj_ids,
    #     stem_path='MNINonLinear/Results',
    #     base_path='rfMRI_REST*/'
    #               'rfMRI_REST*_Atlas_MSMAll_hp2000_clean.dtseries.nii',
    #     out_file=pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
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

    rfMRI_file_status(pjoin(work_dir, 'HCPD_rfMRI_file_check.tsv'))
    rfMRI_file_status(pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv'))
    rfMRI_file_status(pjoin(work_dir, 'HCPA_rfMRI_file_check.tsv'))
