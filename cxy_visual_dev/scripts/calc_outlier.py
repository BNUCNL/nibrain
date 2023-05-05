import os
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from matplotlib import pyplot as plt
from magicbox.io.io import CiftiReader, save2cifti
from magicbox.algorithm.plot import plot_bar
from cxy_visual_dev.lib.predefine import Atlas, LR_count_32k,\
    mmp_map_file, dataset_name2info, proj_dir

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'outlier')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)

# >>>之前为了去除些outlier顶点做的尝试
def select_subject(dataset_name):
    """
    从每个年龄随机选一个被试
    """
    out_file = pjoin(work_dir, f'{dataset_name}_age_subj.csv')

    df = pd.read_csv(dataset_name2info[dataset_name])
    age_name = 'age in years'
    age_uniq = np.unique(df[age_name])

    subj_ids = []
    for age in age_uniq:
        row = df.loc[df[age_name] == age].sample(n=1, axis=0)
        subj_ids.append(row['subID'].item())

    out_data = {'age': age_uniq, 'subID': subj_ids}
    out_df = pd.DataFrame(out_data)
    out_df.to_csv(out_file, index=False)


def plot_box_violin(dataset_name, meas_name, roi_name):
    """
    观察上一步选中的被试，左/右视觉系统所有顶点的测量值分布。

    Args:
        dataset_name (str): HCPD | HCPA
        meas_name (str): thickness | myelin
        roi_name (str): L_cole_visual | R_cole_visual
    """
    # inputs
    subj_file = pjoin(work_dir, f'{dataset_name}_age_subj.csv')
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


def plot_histgram(dataset_name, meas_name, roi_name):
    """
    观察上一步选中的被试，左/右视觉系统所有顶点的测量值分布。

    Args:
        dataset_name (str): HCPD | HCPA
        meas_name (str): thickness | myelin
        roi_name (str): L_cole_visual | R_cole_visual
    """
    # inputs
    iqr_coefs = (1.5, 2, 3)
    iqr_colors = ('r', 'g', 'b')
    subj_file = pjoin(work_dir, f'{dataset_name}_age_subj.csv')
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


def find_outlier_per_subject(dataset_name, meas_name, roi_name, iqr_coef=None,
                             fixed_range=None):
    """
    根据IQR，为所有被试找出左/右视觉系统的outliner

    Args:
        dataset_name (str): HCPD | HCPA
        meas_name (str): thickness | myelin
        roi_name (str): L_cole_visual | R_cole_visual
        iqr_coef (float):
            set non-outlier in [Q1 - iqr_coefxIQR, Q3 + iqr_coefxIQR]
        fixed_range (tuple | list): (lower threshold, higher threshold)
            set non-outlier in [lower threshold, higher threshold]
    Notes:
        iqr_coef和fixed_range必须且只能选用其中一个
    """
    if iqr_coef is None and fixed_range is None:
        raise ValueError("One of iqr_coef and fixed_range should be used!")
    elif iqr_coef is not None and fixed_range is not None:
        raise ValueError("Only one of iqr_coef and fixed_range can be used!")

    # prepare
    meas_file = pjoin(proj_dir,
                      f'data/HCP/{dataset_name}_{meas_name}.dscalar.nii')
    meas_maps = nib.load(meas_file).get_fdata()
    n_subj = meas_maps.shape[0]

    atlas = Atlas('Cole_visual_LR')
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]
    n_vtx = np.sum(roi_idx_map)
    meas_maps = meas_maps[:, roi_idx_map]

    if iqr_coef is None:
        thr_l, thr_h = fixed_range
        out_file = pjoin(work_dir, f'{dataset_name}_{meas_name}_{roi_name}'
                                   f'_outlier_{thr_l}-{thr_h}.npy')
    else:
        Q1 = np.percentile(meas_maps, 25, axis=1, keepdims=True)
        Q3 = np.percentile(meas_maps, 75, axis=1, keepdims=True)
        IQR = Q3 - Q1
        step = iqr_coef * IQR
        thr_l = Q1 - step
        thr_h = Q3 + step
        out_file = pjoin(work_dir, f'{dataset_name}_{meas_name}_{roi_name}'
                                   f'_outlier_{iqr_coef}IQR.npy')

    data = np.zeros((n_subj, n_vtx), bool)

    # calculate
    data[meas_maps > thr_h] = True
    data[meas_maps < thr_l] = True

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


def make_non_outlier_map(fpath, thr, roi_name,
                         out_file_mask=None, out_file_prob=None):
    """
    将同时在thr%以上的被试中被认定为outlier的顶点判定为跨被试的outlier
    If out_file_mask is not None, save mask map. (.dlabel.nii or .npy)
    If out_file_prob is not None, save prob map. (.dscalar.nii)
    """
    # prepare
    data = np.load(fpath)
    n_subj, n_vtx = data.shape

    atlas1 = Atlas('Cole_visual_LR')
    atlas2 = Atlas('Cole_visual_ROI')
    assert atlas1.maps.shape == (1, LR_count_32k)
    assert atlas2.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas1.maps[0] == atlas1.roi2label[roi_name]

    if roi_name == 'R_cole_visual':
        prefix = 'R_'
    elif roi_name == 'L_cole_visual':
        prefix = 'L_'
    else:
        raise ValueError("error roi_name:", roi_name)
    mmp_reader = CiftiReader(mmp_map_file)
    mmp_lbl_tab = mmp_reader.label_tables()[0]

    # calculate
    if out_file_mask is not None:
        data_tmp = np.sum(data, axis=0)
        outlier_vec = data_tmp > thr/100*n_subj
        print(f'#outliers/total: {np.sum(outlier_vec)}/{n_vtx}')
        mask_npy = np.zeros(LR_count_32k, bool)
        mask_npy[roi_idx_map] = ~outlier_vec
        if out_file_mask.endswith('.npy'):
            np.save(out_file_mask, mask_npy)
        elif out_file_mask.endswith('.dlabel.nii'):
            mask_cii = atlas2.maps.copy()
            mask_cii[0, ~mask_npy] = np.nan
            lbl_tab = nib.cifti2.cifti2.Cifti2LabelTable()
            for roi, lbl in atlas2.roi2label.items():
                if roi.startswith(prefix):
                    lbl_tab[lbl] = mmp_lbl_tab[lbl]
            save2cifti(out_file_mask, mask_cii, mmp_reader.brain_models(),
                       label_tables=[lbl_tab])
        else:
            raise ValueError("Not supported file name:", out_file_mask)

    if out_file_prob is not None:
        data_tmp = np.mean(data, axis=0)
        prob_map = np.ones((1, LR_count_32k), dtype=np.float64) * np.nan
        prob_map[0, roi_idx_map] = data_tmp
        assert out_file_prob.endswith('.dscalar.nii')
        save2cifti(out_file_prob, prob_map, mmp_reader.brain_models())
# 之前为了去除些outlier顶点做的尝试<<<


def find_outlier_subj_per_age_RSM(fpaths, out_file, dataset_name='HCPD',
                                  ages='all', iqr_coef=1.5):
    """
    根据各年龄内被试之间thickness或myelin的空间pattern的相似性矩阵
    对相似性矩阵的每一行求平均（不计入对角线的值），以这个值来找到iqr_coef倍IQR以外的被试

    Args:
        fpaths (str): file path with place holders
            占位符关键字分别对应dataset_name, age
        out_file (str): file path for saving out
            存为.npy的numpy bool向量，True为outlier
            长度和数据集被试总数相同，一一对应
        dataset_name (str, optional): Defaults to 'HCPD'.
            HCPD | HCPY | HCPA
        ages (str | sequence, optional): Defaults to 'all'.
            感兴趣的年龄
        iqr_coef (float, optional): Defaults to 1.5.
    """
    info_df = pd.read_csv(dataset_name2info[dataset_name])
    sids = info_df['subID'].to_list()
    age_name = 'age in years'
    if ages == 'all':
        ages = np.unique(info_df[age_name])

    outlier_vec = np.zeros(len(sids), bool)
    n_outliers = np.zeros(len(ages), int)
    for age_idx, age in enumerate(ages):
        fpath = fpaths.format(dataset_name=dataset_name, age=age)
        data = pkl.load(open(fpath, 'rb'))
        n_subj = len(data['row_name'])
        arr = data['r'].copy()
        arr[np.eye(n_subj, dtype=bool)] = np.nan
        corrs = np.nanmean(arr, 1)
        Q1, Q3 = np.percentile(corrs, [25, 75])
        IQR = Q3 - Q1
        step = iqr_coef * IQR
        thr_l = Q1 - step
        thr_h = Q3 + step
        outlier_mask = np.logical_or(corrs < thr_l, corrs > thr_h)
        n_outliers[age_idx] = np.sum(outlier_mask)
        indices = [sids.index(data['row_name'][i]) for i in range(n_subj) if outlier_mask[i]]
        outlier_vec[indices] = True

    title1 = '_'.join(os.path.basename(fpath).split('_')[:-1])
    title2 = f'{title1}\nIQR-{iqr_coef}'
    plot_bar(n_outliers, figsize=(3.2, 2.4), x=ages, fc_ec_flag=True, fc=('w',), ec=('k',),
             show_height='', xlabel=age_name, ylabel='#outliers', title=title2,
             mode=pjoin(work_dir, f'{title1}.jpg'))
    np.save(out_file, outlier_vec)


if __name__ == '__main__':
    # >>>之前为了去除些outlier顶点做的尝试
    # select_subject(dataset_name='HCPD')
    # plot_box_violin(dataset_name='HCPD', meas_name='thickness', roi_name='R_cole_visual')
    # plot_box_violin(dataset_name='HCPD', meas_name='myelin', roi_name='R_cole_visual')
    # plot_histgram(dataset_name='HCPD', meas_name='thickness', roi_name='R_cole_visual')
    # plot_histgram(dataset_name='HCPD', meas_name='myelin', roi_name='R_cole_visual')
    # find_outlier_per_subject(
    #     dataset_name='HCPD', meas_name='myelin', roi_name='R_cole_visual',
    #     fixed_range=(0.1, 5)
    # )
    # plot_outlier_distribution(
    #     fpath=pjoin(work_dir, 'HCPD_myelin_R_cole_visual_outlier_0.1-5.npy'),
    #     title='HCPD_myelin_R_cole_visual_outlier_0.1-5'
    # )
    # make_non_outlier_map(
    #     fpath=pjoin(work_dir, 'HCPD_myelin_R_cole_visual_outlier_0.1-5.npy'),
    #     thr=0, roi_name='R_cole_visual',
    #     out_file_mask=pjoin(work_dir, 'HCPD_myelin_R_cole_visual_0.1-5_thr0_mask.npy'),
    #     out_file_prob=pjoin(work_dir, 'HCPD_myelin_R_cole_visual_0.1-5_thr0_prob.dscalar.nii')
    # )
    # 之前为了去除些outlier顶点做的尝试<<<

    find_outlier_subj_per_age_RSM(
        fpaths=pjoin(anal_dir, 'RSM/RSM_{dataset_name}-myelin_MMP-vis3-R_age-{age}.pkl'),
        out_file=pjoin(work_dir, 'HCPD-myelin_MMP-vis3-R_RSM-IQR3.npy'),
        dataset_name='HCPD', ages='all', iqr_coef=3
    )
    find_outlier_subj_per_age_RSM(
        fpaths=pjoin(anal_dir, 'RSM/RSM_{dataset_name}-thickness_MMP-vis3-R_age-{age}.pkl'),
        out_file=pjoin(work_dir, 'HCPD-thickness_MMP-vis3-R_RSM-IQR3.npy'),
        dataset_name='HCPD', ages='all', iqr_coef=3
    )
