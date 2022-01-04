import os
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    get_rois, hemi2Hemi
from cxy_visual_dev.lib.algo import cat_data_from_cifti,\
    linear_fit1
from magicbox.io.io import CiftiReader

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'fit')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def old_fit():
    """
    分别用枕极和MT做锚点，和距状沟锚点一起描述C1
    这里用这三个map作为三个特征去线性拟合C1（左右脑拼起来之前分别做zscore）

    用HCPD每个被试的myelin和thickness map去拟合C1C2
    这里做拼接左右脑数据的时候会分别做zscore
    """
    mask = Atlas('HCP-MMP').get_mask(
        get_rois('MMP-vis2-L') + get_rois('MMP-vis2-R'))[0]
    mask_L = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis2-L'))[0]
    mask_R = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis2-R'))[0]
    C1C2_maps = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii'
    )).get_fdata()[:2]

    src_files = [
            pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-MT.dscalar.nii')]
    X_list = []
    for src_file in src_files:
        data = cat_data_from_cifti([src_file], (1, 1),
                                   [mask_L, mask_R], zscore1='split')[0]
        X_list.append(data.T)
    Y = C1C2_maps[0, mask][:, None]
    linear_fit1(
        X_list=X_list, feat_names=['CalcarineSulcus', 'OccipitalPole', 'MT'],
        Y=Y, trg_names=['C1'], score_metric='R2',
        out_file=pjoin(work_dir, 'CalcS+OcPole+MT=C1.csv'),
        standard_scale=False
    )

    src_files = [
            pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
            pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')]
    X_list = []
    for src_file in src_files:
        data = cat_data_from_cifti([src_file], (1, 1),
                                   [mask_L, mask_R], zscore1='split')[0]
        X_list.append(data.T)
    Y = C1C2_maps[:, mask].T
    linear_fit1(
        X_list=X_list, feat_names=['Myelination', 'Thickness'],
        Y=Y, trg_names=['C1', 'C2'], score_metric='R2',
        out_file=pjoin(work_dir, 'HCPD-M+T=C1C2.csv'),
        standard_scale=False
    )


def gdist_fit_PC1():
    """
    分别用枕极和MT做锚点，和距状沟锚点一起描述C1
    这里用这三个map作为三个特征去线性拟合C1 (半脑)
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    src_files = [
            # pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-MT.dscalar.nii')]
    feat_names = ['OccipitalPole', 'MT']
    out_file = pjoin(work_dir, 'OcPole+MT=C1.csv')

    C1_map = nib.load(pc_file).get_fdata()[0]
    X_list = []
    for src_file in src_files:
        data = nib.load(src_file).get_fdata()[:, mask]
        X_list.append(data.T)
    Y = np.expand_dims(C1_map[mask], 1)
    linear_fit1(
        X_list=X_list, feat_names=feat_names,
        Y=Y, trg_names=['C1'], score_metric='R2',
        out_file=out_file, standard_scale=True)


def HCPDA_fit_PC12():
    """
    用HCPD/A每个被试的myelin和thickness map去拟合C1C2（半脑）
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    src_files = [
            pjoin(proj_dir, 'data/HCP/HCPA_myelin.dscalar.nii'),
            pjoin(proj_dir, 'data/HCP/HCPA_thickness.dscalar.nii')]
    feat_names = ['Myelination', 'Thickness']
    out_file = pjoin(work_dir, 'HCPA-M+T=C1C2.csv')

    C1C2_maps = nib.load(pc_file).get_fdata()[:2]
    X_list = []
    for src_file in src_files:
        data = nib.load(src_file).get_fdata()[:, mask]
        X_list.append(data.T)
    Y = C1C2_maps[:, mask].T
    linear_fit1(
        X_list=X_list, feat_names=feat_names,
        Y=Y, trg_names=['C1', 'C2'], score_metric='R2',
        out_file=out_file, standard_scale=True)


def mean_tau_diff_fit_PC12():
    """
    用某岁的平均map+tau+diff预测成人PC map
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    src_files = [
        pjoin(anal_dir, 'summary_map/HCPD-myelin_age-map-mean.dscalar.nii'),
        pjoin(anal_dir, 'summary_map/HCPD-thickness_age-map-mean.dscalar.nii'),
        # pjoin(anal_dir, 'dev_trend/HCPD-myelin_MMP-vis3_kendall.dscalar.nii'),
        # pjoin(anal_dir, 'dev_trend/HCPD-thickness_MMP-vis3_kendall.dscalar.nii'),
        # pjoin(anal_dir, 'dev_trend/HCPD-myelin_age-map-mean_21-8.dscalar.nii'),
        # pjoin(anal_dir, 'dev_trend/HCPD-thickness_age-map-mean_21-8.dscalar.nii')
    ]
    # feat_names = ['myelin-8', 'thickness-8', 'myelin-tau', 'thickness-tau',
    #               'myelin-diff', 'thickness-diff']
    # map_indices = [3, 3, 0, 0, 0, 0]
    # out_file = pjoin(work_dir, 'Mean8+Tau+Diff=C1C2.csv')
    feat_names = ['myelin-8', 'thickness-8']
    map_indices = [3, 3]
    out_file = pjoin(work_dir, 'Mean8=C1C2.csv')

    C1C2_maps = nib.load(pc_file).get_fdata()[:2]
    X_list = []
    for i, src_file in enumerate(src_files):
        data = nib.load(src_file).get_fdata()[map_indices[i], mask]
        X_list.append(np.expand_dims(data, 1))
    Y = C1C2_maps[:, mask].T
    linear_fit1(
        X_list=X_list, feat_names=feat_names,
        Y=Y, trg_names=['C1', 'C2'], score_metric='R2',
        out_file=out_file, standard_scale=True)


def HCPDA_fit_PC12_local():
    """
    用HCPD/A每个被试的myelin和thickness map去局部拟合C1C2（半脑）
    """
    data_name = 'HCPD'
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    mask_local_file = pjoin(anal_dir, f'mask_map/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj_N3.dlabel.nii')
    mask_local_lbls = np.arange(1, 4)
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')

    src_files = [
            pjoin(proj_dir, f'data/HCP/{data_name}_myelin.dscalar.nii'),
            pjoin(proj_dir, f'data/HCP/{data_name}_thickness.dscalar.nii')]
    feat_names = ['Myelination', 'Thickness']
    out_file = pjoin(work_dir, f'{data_name}-M+T=C1C2_local-mask1-N3-{Hemi}.csv')

    mask_locals = nib.load(mask_local_file).get_fdata()[:, mask]
    pc_names = ['C1', 'C2']
    n_pc = len(pc_names)
    reader = CiftiReader(pc_file)
    pc_maps = reader.get_data()[:n_pc, mask].T
    assert pc_names == reader.map_names()[:n_pc]

    feat_maps_list = []
    for src_file in src_files:
        data = nib.load(src_file).get_fdata()[:, mask]
        feat_maps_list.append(data.T)

    dfs = []
    for pc_idx in range(n_pc):
        for mask_local_lbl in mask_local_lbls:
            mask_local = mask_locals[pc_idx] == mask_local_lbl
            X_list = [i[mask_local] for i in feat_maps_list]
            Y = np.expand_dims(pc_maps[mask_local, pc_idx], 1)
            df = linear_fit1(
                X_list=X_list, feat_names=feat_names,
                Y=Y, trg_names=[f'{pc_names[pc_idx]}-{mask_local_lbl}'], score_metric='R2',
                out_file='df', standard_scale=True)
            dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    # gdist_fit_PC1()
    # HCPDA_fit_PC12()
    # mean_tau_diff_fit_PC12()
    HCPDA_fit_PC12_local()
