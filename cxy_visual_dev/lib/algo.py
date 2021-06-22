import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from scipy.stats import zscore, sem
from sklearn.decomposition import PCA
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import Atlas, L_offset_32k, L_count_32k,\
    R_offset_32k, R_count_32k, LR_count_32k, mmp_map_file


def ROI_analysis(data_file, atlas_name, out_file, zscore_flag=False):
    """
    为每个被试每个ROI求均值

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_subj, LR_count_32k)
        atlas_name (str): include ROIs' labels and mask map
        out_file (str): output path
        zscore_flag (bool, optional): Defaults to False.
            If True, do zscore across each hemisphere.
    """
    # prepare
    meas_maps = nib.load(data_file).get_fdata()
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    out_df = pd.DataFrame()

    # calculate
    if zscore_flag:
        meas_maps_L = meas_maps[:, L_offset_32k:(L_offset_32k+L_count_32k)]
        meas_maps_R = meas_maps[:, R_offset_32k:(R_offset_32k+R_count_32k)]
        meas_maps_L = zscore(meas_maps_L, 1)
        meas_maps_R = zscore(meas_maps_R, 1)
        meas_maps[:, L_offset_32k:(L_offset_32k+L_count_32k)] = meas_maps_L
        meas_maps[:, R_offset_32k:(R_offset_32k+R_count_32k)] = meas_maps_R
        del meas_maps_L, meas_maps_R

    for roi, lbl in atlas.roi2label.items():
        meas_vec = np.mean(meas_maps[:, atlas.maps[0] == lbl], 1)
        out_df[roi] = meas_vec

    # save
    out_df.to_csv(out_file, index=False)


def pca(data_file, atlas_name, roi_name, n_component, axis, out_name):
    """
    对n_subj x n_vtx形状的矩阵进行PCA降维

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_subj, LR_count_32k)
        atlas_name (str): include ROIs' labels and mask map
        roi_name (str): 决定选用哪个区域内的顶点来参与PCA
        n_component (int): the number of components
        axis (str): vertex | subject
            vertex: 对顶点数量进行降维，得到几个主成分时间序列，
            观察某个主成分在各顶点上的权重，刻画其空间分布。
            subject: 对被试数量进行降维，得到几个主成分map，
            观察某个主成分在各被试上的权重，按年龄排序即可得到时间序列。
        out_name (str): output name
            If axis=vertex, output
            1. n_subj x n_component out_name.csv
            2. n_component x LR_count_32k out_name.dscalar.nii
            3. out_name.pkl with fitted PCA model
    """
    # prepare
    component_names = [f'C{i}' for i in range(1, n_component+1)]

    meas_maps = nib.load(data_file).get_fdata()
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]
    meas_maps = meas_maps[:, roi_idx_map]

    bms = CiftiReader(mmp_map_file).brain_models()

    # calculate
    pca = PCA(n_components=n_component)
    data = np.ones((n_component, LR_count_32k), np.float64) * np.nan
    if axis == 'vertex':
        X = meas_maps
        pca.fit(X)
        Y = pca.transform(X)
        df = pd.DataFrame(data=Y, columns=component_names)
        data[:, roi_idx_map] = pca.components_

    elif axis == 'subject':
        X = meas_maps.T
        pca.fit(X)
        Y = pca.transform(X)
        df = pd.DataFrame(data=pca.components_.T, columns=component_names)
        data[:, roi_idx_map] = Y.T

    else:
        raise ValueError('Invalid axis:', axis)

    # save
    df.to_csv(f'{out_name}.csv', index=False)
    save2cifti(f'{out_name}.dscalar.nii', data, bms, component_names)
    pkl.dump(pca, open(f'{out_name}.pkl', 'wb'))


def ROI_analysis_on_PC(data_file, pca_file, pc_num,
                       mask_atlas, mask_name, roi_atlas, out_file):
    """
    利用指定PC的weights加权各ROI内的值

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_subj, LR_count_32k)
        pca_file (str): file with fitted PCA model
            axis=vertex
        pc_num (int): serial number of the selected principle component
        mask_atlas (str): include ROIs' labels and mask map
        mask_name (str): 指明做PCA时用的是mask_atlas中的那个ROI
        roi_atlas (str): include ROIs' labels and mask map
            指定对那些ROI进行加权操作
        out_file (str): output file
    """
    # prepare measure maps
    meas_maps = nib.load(data_file).get_fdata()
    atlas_mask = Atlas(mask_atlas)
    assert atlas_mask.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas_mask.maps[0] == atlas_mask.roi2label[mask_name]
    meas_maps = meas_maps[:, roi_idx_map]

    # prepare ROI map
    atlas_roi = Atlas(roi_atlas)
    assert atlas_roi.maps.shape == (1, LR_count_32k)
    roi_map = atlas_roi.maps[0][roi_idx_map]
    roi_labels = np.unique(roi_map)
    roi_label2name = {}
    for name, lbl in atlas_roi.roi2label.items():
        roi_label2name[lbl] = name

    # prepare PCA
    pca = pkl.load(open(pca_file, 'rb'))
    eigen_vec = pca.components_.T[:, [pc_num-1]]

    # calculate
    meas_maps = meas_maps - np.expand_dims(pca.mean_, 0)
    out_df = pd.DataFrame()
    for lbl in roi_labels:
        idx_map = roi_map == lbl
        X = meas_maps[:, idx_map]
        eigen_tmp = eigen_vec[idx_map]
        y = np.matmul(X, eigen_tmp)
        out_df[roi_label2name[lbl]] = y[:, 0]

    # save
    out_df.to_csv(out_file, index=False)


def make_age_maps(data_file, info_file, out_name):
    """
    对每个顶点，计算跨同一年龄被试的平均和sem，分别保存在
    out_name-mean.dscalar.nii, out_name-sem.dscalar.nii中

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_subj, LR_count_32k)
        info_file (str): subject info file
        out_name (str): filename to save
    """
    # prepare
    data_maps = nib.load(data_file).get_fdata()
    info_df = pd.read_csv(info_file)
    ages = np.array(info_df['age in years'])
    ages_uniq = np.unique(ages)
    n_age = len(ages_uniq)

    # calculate
    mean_maps = np.ones((n_age, LR_count_32k)) * np.nan
    sem_maps = np.ones((n_age, LR_count_32k)) * np.nan
    for age_idx, age in enumerate(ages_uniq):
        data = data_maps[ages == age]
        mean_maps[age_idx] = np.mean(data, 0)
        sem_maps[age_idx] = sem(data, 0)

    # save
    map_names = [str(i) for i in ages_uniq]
    reader = CiftiReader(mmp_map_file)
    save2cifti(f'{out_name}-mean.dscalar.nii', mean_maps,
               reader.brain_models(), map_names)
    save2cifti(f'{out_name}-sem.dscalar.nii', sem_maps,
               reader.brain_models(), map_names)


def calc_map_corr(data_file, ref_file, atlas_name, out_file):
    """[summary]

    Args:
        data_file ([type]): [description]
        ref_file ([type]): [description]
        atlas_name ([type]): [description]
        out_file ([type]): [description]
    """
    pass
