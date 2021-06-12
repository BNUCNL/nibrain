import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from matplotlib import pyplot as plt
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import Atlas, LR_count_32k,\
    mmp_file, dataset_name2dir, dataset_name2info

proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'analysis/outlier')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def select_subject():
    """
    从每个年龄随机选一个被试
    """
    info_file = '/nfs/e1/HCPD/HCPD_SubjInfo.csv'
    out_file = pjoin(work_dir, 'age_subj.csv')

    df = pd.read_csv(info_file)
    age_name = 'age in years'
    age_uniq = np.unique(df[age_name])

    subj_ids = []
    for age in age_uniq:
        row = df.loc[df[age_name] == age].sample(n=1, axis=0)
        subj_ids.append(row['subID'].item())

    out_data = {'age': age_uniq, 'subID': subj_ids}
    out_df = pd.DataFrame(out_data)
    out_df.to_csv(out_file, index=False)


def plot_box_violin(meas_name, roi_name):
    """
    观察上一步选中的被试，左/右视觉系统所有顶点的测量值分布。

    Args:
        meas_name (str): thickness | myelin
        roi_name (str): L_cole_visual | R_cole_visual
    """
    # inputs
    subj_file = pjoin(work_dir, 'age_subj.csv')
    meas2file = {
        'myelin': '/nfs/e1/HCPD/fmriresults01/{sid}_V1_MR/'
                  'MNINonLinear/fsaverage_LR32k/'
                  '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
        'thickness': '/nfs/e1/HCPD/fmriresults01/{sid}_V1_MR/'
                     'MNINonLinear/fsaverage_LR32k/'
                     '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
    }

    # prepare
    df = pd.read_csv(subj_file)
    atlas = Atlas('Cole_visual_LR')
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]

    # plot
    points_list = []
    for subj_id in df['subID']:
        meas_file = meas2file[meas_name].format(sid=subj_id)
        meas_map = nib.load(meas_file).get_fdata()[0]
        meas_vec = meas_map[roi_idx_map]
        points_list.append(meas_vec)
    plt.boxplot(points_list, positions=df['age'])
    plt.violinplot(points_list, positions=df['age'], showextrema=False)
    plt.title(roi_name)
    plt.ylabel(meas_name)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_histgram(meas_name, roi_name):
    """
    观察上一步选中的被试，左/右视觉系统所有顶点的测量值分布。

    Args:
        meas_name (str): thickness | myelin
        roi_name (str): L_cole_visual | R_cole_visual
    """
    # inputs
    iqr_coefs = (1.5, 2, 3)
    iqr_colors = ('r', 'g', 'b')
    subj_file = pjoin(work_dir, 'age_subj.csv')
    meas2file = {
        'myelin': '/nfs/e1/HCPD/fmriresults01/{sid}_V1_MR/'
                  'MNINonLinear/fsaverage_LR32k/'
                  '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
        'thickness': '/nfs/e1/HCPD/fmriresults01/{sid}_V1_MR/'
                     'MNINonLinear/fsaverage_LR32k/'
                     '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
    }

    # prepare
    df = pd.read_csv(subj_file)
    n_subj = df.shape[0]
    atlas = Atlas('Cole_visual_LR')
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]

    # plot
    _, axes = plt.subplots(1, n_subj)
    for subj_idx, subj_id in enumerate(df['subID']):
        ax = axes[subj_idx]
        meas_file = meas2file[meas_name].format(sid=subj_id)
        meas_map = nib.load(meas_file).get_fdata()[0]
        meas_vec = meas_map[roi_idx_map]
        ax.hist(meas_vec, bins=100, orientation='horizontal')
        xmin, xmax = ax.get_xlim()

        Q1 = np.percentile(meas_vec, 25)
        Q3 = np.percentile(meas_vec, 75)
        IQR = Q3 - Q1
        for i, iqr_coef in enumerate(iqr_coefs):
            step = iqr_coef * IQR
            whiskers = [Q1-step, Q3+step]
            ax.hlines(whiskers, xmin, xmax, colors=iqr_colors[i])

        if subj_idx == int(n_subj/2):
            ax.set_title(f"{roi_name}\nage-{df['age'][subj_idx]}")
        else:
            if subj_idx == 0:
                ax.set_ylabel(meas_name)
            ax.set_title(f"age-{df['age'][subj_idx]}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def make_non_outlier_mask_by_fixed_value(dataset_name, meas_name, roi_name,
                                         thr_low, thr_high):
    """
    将低于thr_low和高于thr_high的顶点都认为是outlier，
    制定限制于roi_name指定区域内的非outlier mask

    Args:
        dataset_name (str): HCPD | HCPA
        meas_name (str): thickness | myelin
        roi_name (str): L_cole_visual | R_cole_visual
        thr_low (float): lower threshold
        thr_high (float): higher threshold
    """
    # outputs
    out_fname = f'{dataset_name}_{meas_name}_{roi_name}_{thr_low}-{thr_high}'
    out_file = pjoin(work_dir, f'{out_fname}.npy')
    out_file_cii = pjoin(work_dir, f'{out_fname}.dlabel.nii')

    # prepare
    dataset_dir = dataset_name2dir[dataset_name]
    meas2file = {
        'myelin': pjoin(
            dataset_dir,
            'fmriresults01/{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
            '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
        ),
        'thickness': pjoin(
            dataset_dir,
            'fmriresults01/{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
            '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
        )
    }
    df = pd.read_csv(dataset_name2info[dataset_name])
    n_subj = df.shape[0]

    atlas1 = Atlas('Cole_visual_LR')
    atlas2 = Atlas('Cole_visual_ROI')
    assert atlas1.maps.shape == (1, LR_count_32k)
    assert atlas2.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas1.maps[0] == atlas1.roi2label[roi_name]
    n_vtx = np.sum(roi_idx_map)

    mmp_reader = CiftiReader(mmp_file)
    mmp_lbl_tab = mmp_reader.label_tables()[0]

    non_outlier_mask = roi_idx_map.copy()

    # calculate
    for subj_idx, subj_id in enumerate(df['subID']):
        time1 = time.time()
        meas_file = meas2file[meas_name].format(sid=subj_id)
        meas_map = nib.load(meas_file).get_fdata()[0]
        non_outlier_mask[meas_map < thr_low] = False
        non_outlier_mask[meas_map > thr_high] = False
        print(f'Finished: {subj_idx+1}/{n_subj},'
              f'cost: {time.time() - time1} seconds.')
    n_outlier = np.sum(~non_outlier_mask[roi_idx_map])
    print(f'#outlier/total: {n_outlier}/{n_vtx}')
    cii_data = atlas2.maps.copy()
    cii_data[0, ~non_outlier_mask] = np.nan

    # save
    np.save(out_file, non_outlier_mask)

    lbl_tab = nib.cifti2.cifti2.Cifti2LabelTable()
    if roi_name == 'R_cole_visual':
        prefix = 'R_'
    elif roi_name == 'L_cole_visual':
        prefix = 'L_'
    else:
        raise ValueError("error roi_name:", roi_name)
    for roi, lbl in atlas2.roi2label.items():
        if roi.startswith(prefix):
            lbl_tab[lbl] = mmp_lbl_tab[lbl]
    save2cifti(out_file_cii, cii_data, mmp_reader.brain_models(),
               label_tables=[lbl_tab])


# >>>find outliers by IQR and #subjects threshold (discarded)
def find_outlier_per_subject_IQR(meas_name, roi_name, iqr_coef):
    """
    根据IQR，为所有被试找出左/右视觉系统的outliner

    Args:
        meas_name (str): thickness | myelin
        roi_name (str): L_cole_visual | R_cole_visual
        iqr_coef (float):
            set non-outlier in [Q1 - iqr_coefxIQR, Q3 + iqr_coefxIQR]
    """
    # inputs
    info_file = '/nfs/e1/HCPD/HCPD_SubjInfo.csv'
    meas2file = {
        'myelin': '/nfs/e1/HCPD/fmriresults01/{sid}_V1_MR/'
                  'MNINonLinear/fsaverage_LR32k/'
                  '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
        'thickness': '/nfs/e1/HCPD/fmriresults01/{sid}_V1_MR/'
                     'MNINonLinear/fsaverage_LR32k/'
                     '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
    }

    # outputs
    out_file = pjoin(work_dir, f'{meas_name}_{roi_name}_{iqr_coef}IQR.npy')

    # prepare
    df = pd.read_csv(info_file)
    n_subj = df.shape[0]
    atlas = Atlas('Cole_visual_LR')
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]
    n_vtx = np.sum(roi_idx_map)
    data = np.zeros((n_subj, n_vtx), bool)

    # calclate
    for subj_idx, subj_id in enumerate(df['subID']):
        time1 = time.time()
        meas_file = meas2file[meas_name].format(sid=subj_id)
        meas_map = nib.load(meas_file).get_fdata()[0]
        meas_vec = meas_map[roi_idx_map]
        Q1 = np.percentile(meas_vec, 25)
        Q3 = np.percentile(meas_vec, 75)
        IQR = Q3 - Q1
        step = iqr_coef * IQR
        data[subj_idx] = np.logical_or(meas_vec < Q1-step, meas_vec > Q3+step)
        print(f'Finished: {subj_idx+1}/{n_subj},'
              f'cost: {time.time() - time1} seconds.')

    # save
    np.save(out_file, data)


def plot_outlier_distribution(fpath, title):
    """
    统计在所有N个被试中，同时在至少x个被试中成为outliner的顶点比例
    x属于range(1, N+1)
    """
    data = np.load(fpath)
    n_subj, n_vtx = data.shape
    data = np.sum(data, axis=0)

    x = np.arange(1, n_subj+1)
    y = np.zeros(n_subj, np.float64)
    for i, j in enumerate(x):
        y[i] = np.sum(data >= j) / n_vtx
    print('total vertices:', n_vtx)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('#subjects')
    plt.ylabel('#vertices')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def make_non_outlier_mask(fpath, thr, roi_name, out_file):
    """
    将同时在thr%以上的被试中被认定为outlier的顶点判定为跨被试的outlier
    """
    # outputs
    out_fname = os.path.basename(out_file)
    out_file_cii = pjoin(work_dir, f"{out_fname.rstrip('.npy')}.dlabel.nii")

    # prepare
    data = np.load(fpath)
    n_subj, n_vtx = data.shape
    data = np.sum(data, axis=0)
    atlas1 = Atlas('Cole_visual_LR')
    atlas2 = Atlas('Cole_visual_ROI')
    assert atlas1.maps.shape == (1, LR_count_32k)
    assert atlas2.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas1.maps[0] == atlas1.roi2label[roi_name]
    mmp_reader = CiftiReader(mmp_file)
    mmp_lbl_tab = mmp_reader.label_tables()[0]

    # calculate
    outlier_vec = data > thr/100*n_subj
    print(f'#outliers: {np.sum(outlier_vec)}/{n_vtx}')
    npy_data = np.zeros(LR_count_32k, bool)
    npy_data[roi_idx_map] = ~outlier_vec
    cii_data = atlas2.maps.copy()
    cii_data[0, ~npy_data] = np.nan

    # save
    np.save(out_file, npy_data)

    lbl_tab = nib.cifti2.cifti2.Cifti2LabelTable()
    if roi_name == 'R_cole_visual':
        prefix = 'R_'
    elif roi_name == 'L_cole_visual':
        prefix = 'L_'
    else:
        raise ValueError("error roi_name:", roi_name)
    for roi, lbl in atlas2.roi2label.items():
        if roi.startswith(prefix):
            lbl_tab[lbl] = mmp_lbl_tab[lbl]
    save2cifti(out_file_cii, cii_data, mmp_reader.brain_models(),
               label_tables=[lbl_tab])
# find outliers by IQR and #subjects threshold<<<


if __name__ == '__main__':
    # select_subject()
    # plot_box_violin(meas_name='thickness', roi_name='R_cole_visual')
    # plot_box_violin(meas_name='myelin', roi_name='R_cole_visual')
    # plot_histgram(meas_name='thickness', roi_name='R_cole_visual')
    # plot_histgram(meas_name='myelin', roi_name='R_cole_visual')
    make_non_outlier_mask_by_fixed_value(
        dataset_name='HCPD', meas_name='thickness',
        roi_name='R_cole_visual', thr_low=1, thr_high=4.5
    )
    make_non_outlier_mask_by_fixed_value(
        dataset_name='HCPD', meas_name='myelin',
        roi_name='R_cole_visual', thr_low=0.1, thr_high=3
    )
    # find_outlier_per_subject_IQR(meas_name='thickness', roi_name='R_cole_visual', iqr_coef=2)
    # find_outlier_per_subject_IQR(meas_name='myelin', roi_name='R_cole_visual', iqr_coef=2)
    # plot_outlier_distribution(
    #     fpath=pjoin(work_dir, 'thickness_R_cole_visual_2IQR.npy'),
    #     title='thickness-R_cole_visual-2IQR'
    # )
    # plot_outlier_distribution(
    #     fpath=pjoin(work_dir, 'myelin_R_cole_visual_2IQR.npy'),
    #     title='myelin-R_cole_visual-2IQR'
    # )
    # make_non_outlier_mask(
    #     fpath=pjoin(work_dir, 'thickness_R_cole_visual_2IQR.npy'),
    #     thr=30, roi_name='R_cole_visual',
    #     out_file=pjoin(work_dir, 'thickness_R_cole_visual_2IQR_thr-30_mask.npy')
    # )
    # make_non_outlier_mask(
    #     fpath=pjoin(work_dir, 'myelin_R_cole_visual_2IQR.npy'),
    #     thr=30, roi_name='R_cole_visual',
    #     out_file=pjoin(work_dir, 'myelin_R_cole_visual_2IQR_thr-30_mask.npy')
    # )
