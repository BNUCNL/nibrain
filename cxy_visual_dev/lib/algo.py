import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from scipy.stats import zscore, sem
from scipy.spatial.distance import cdist
from scipy.signal import detrend
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import Atlas, L_offset_32k, L_count_32k,\
    R_offset_32k, R_count_32k, LR_count_32k, mmp_map_file


def zscore_data(data_file, out_file):
    """
    对每个被试做全脑zscore

    Args:
        data_file (str): .dscalar.nii
        out_file (str): .dscalar.nii
    """
    reader = CiftiReader(data_file)
    data = reader.get_data()
    data = zscore(data, 1)
    save2cifti(out_file, data, reader.brain_models(), reader.map_names())


def zscore_map(data_file, out_file, atlas_name=None, roi_name=None):
    """
    zscore data in the ROI for each map

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_map, LR_count_32k)
        out_file (str): end with .dscalar.nii
            shape=(n_map, LR_count_32k)
        atlas_name (str): include ROIs' labels and mask map
        roi_name (str): 决定选用哪个区域内的顶点来参与zscore
    """
    reader = CiftiReader(data_file)
    maps = reader.get_data()
    n_map = maps.shape[0]
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]
    maps = maps[:, roi_idx_map]

    # calculate
    data = np.ones((n_map, LR_count_32k), np.float64) * np.nan
    maps = zscore(maps, 1)
    data[:, roi_idx_map] = maps

    save2cifti(out_file, data, reader.brain_models(), reader.map_names())


def zscore_map_subj(data_file, out_file):
    """
    zscore data along subjects

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_map, LR_count_32k)
        out_file (str): end with .dscalar.nii
            shape=(n_map, LR_count_32k)
    """
    reader = CiftiReader(data_file)
    maps = reader.get_data()
    maps = zscore(maps, 0)
    save2cifti(out_file, maps, reader.brain_models(), reader.map_names())


def concate_map(data_files, out_file):
    """
    Args:
        data_files (strings): Each string is end with .dscalar.nii
            shape=(n_map, LR_count_32k)
        out_file (str): end with .dscalar.nii
            shape=(n_map_total, LR_count_32k)
    """
    reader = CiftiReader(data_files[0])
    maps = reader.get_data()
    map_names = reader.map_names()
    for data_file in data_files[1:]:
        reader_tmp = CiftiReader(data_file)
        maps_tmp = reader_tmp.get_data()
        maps = np.r_[maps, maps_tmp]
        map_names.extend(reader_tmp.map_names())
    save2cifti(out_file, maps, reader.brain_models(), map_names)


def ROI_analysis(data_file, atlas_name, out_file, zscore_flag=False):
    """
    为每个被试每个ROI求均值

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_subj, LR_count_32k)
        atlas_name (str): include ROIs' labels and mask map
        out_file (str): output path
        zscore_flag (bool, optional): Defaults to False.
            If True, do zscore within each hemisphere.
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


def ROI_scalar(data_file, atlas_name, rois, metric, out_file, out_index=None):
    """
    Upgrade for "ROI_analysis"
    为每个被试每个ROI求scalar value

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_map, n_vtx)
        atlas_name (str): include ROIs' labels and mask map
        rois (None | sequence): ROI names
            If is None, use all ROIs of the atlas.
        metric (str): mean, var, sem
        out_file (str): end with .csv
        out_index (None | str | sequence):
            If None, don't save index to out_file.
            If str, must be 'map name' that means using map names as indices.
            If sequence, its length is equal to n_map.
    """
    # prepare
    reader = CiftiReader(data_file)
    maps = reader.get_data()

    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, maps.shape[1])
    if rois is None:
        rois = atlas.roi2label.keys()

    out_df = pd.DataFrame()

    # calculate
    for roi in rois:
        lbl = atlas.roi2label[roi]
        data = maps[:, atlas.maps[0] == lbl]
        if metric == 'mean':
            vec = np.mean(data, 1)
        elif metric == 'var':
            vec = np.var(data, 1)
        elif metric == 'sem':
            vec = sem(data, 1)
        else:
            raise ValueError('not supported metric')
        out_df[roi] = vec

    # save
    if out_index is None:
        out_df.to_csv(out_file, index=False)
    elif out_index == 'map name':
        out_df.index = reader.map_names()
        out_df.to_csv(out_file, index=True)
    else:
        assert len(out_index) == maps.shape[0]
        out_df.index = out_index
        out_df.to_csv(out_file, index=True)


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


def pca_mf(data_files, atlas_names, roi_names, n_component, axis,
           zscore0, zscore1, csv_files, cii_file, pkl_file, random_state=None):
    """
    对n_subj x n_vtx形状的矩阵进行PCA降维
    adapted for dealing with multi files

    Args:
        data_files (str or 1D array-like): end with .dscalar.nii
            shape=(n_subj, LR_count_32k)
        atlas_names (str or 1D array-like): include ROIs' labels and mask map
        roi_names (str or 1D array-like): 决定选用哪个区域内的顶点来参与PCA
            corresponding to atlas_names
        n_component (int): the number of components
        axis (str): vertex | subject
            vertex: 对顶点数量进行降维，得到几个主成分时间序列，
            观察某个主成分在各顶点上的权重，刻画其空间分布。
            subject: 对被试数量进行降维，得到几个主成分map，
            观察某个主成分在各被试上的权重，按年龄排序即可得到时间序列。
        zscore0 (str): None, split, whole
        zscore1 (str): None, split, whole
        csv_files (str or 1D array-like): n_subj x n_component
            corresponding to rows of data_files
        cii_file (str): n_component x LR_count_32k
        pkl_file (str): fitted PCA model
    """
    # prepare
    if isinstance(data_files, str):
        data_files = np.atleast_1d(data_files)
    else:
        data_files = np.asarray(data_files)
        assert data_files.ndim == 1
    n_row = len(data_files)

    if isinstance(atlas_names, str):
        atlas_names = np.atleast_1d(atlas_names)
    else:
        atlas_names = np.asarray(atlas_names)
        assert atlas_names.ndim == 1
    n_col = len(atlas_names)

    if isinstance(roi_names, str):
        roi_names = np.atleast_1d(roi_names)
    else:
        roi_names = np.asarray(roi_names)
        assert roi_names.ndim == 1
    assert len(roi_names) == n_col

    if isinstance(csv_files, str):
        csv_files = np.atleast_1d(csv_files)
    else:
        csv_files = np.asarray(csv_files)
        assert csv_files.ndim == 1
    assert len(csv_files) == n_row

    component_names = [f'C{i}' for i in range(1, n_component+1)]

    # prepare data
    roi_idx_maps = []
    subj_idx_vecs = []
    n_vertices = [0]
    n_subjects = [0]
    data = []
    for col_idx in range(n_col):
        print(f'---{atlas_names[col_idx]}---')

        # load maps1
        reader = CiftiReader(data_files[0])
        maps1 = reader.get_data()

        # load atlas and mask maps1
        atlas = Atlas(atlas_names[col_idx])
        assert atlas.maps.shape == (1, LR_count_32k)
        roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_names[col_idx]]
        roi_idx_maps.append(roi_idx_map)
        maps1 = maps1[:, roi_idx_map]

        non_nan_idx_arr1 = ~np.isnan(maps1)
        subj_idx_vec1 = np.all(non_nan_idx_arr1, 1)
        assert np.all(subj_idx_vec1 == np.any(non_nan_idx_arr1, 1))
        maps1 = maps1[subj_idx_vec1]
        print(f'{data_files[0]}\n{maps1.shape}\n')

        n_vertices.append(n_vertices[-1] + maps1.shape[1])
        if col_idx == 0:
            n_subjects.append(n_subjects[-1] + maps1.shape[0])
            subj_idx_vecs.append(subj_idx_vec1)
        else:
            assert np.all(subj_idx_vecs[0] == subj_idx_vec1)

        # zscore
        if zscore1 == 'split':
            maps1 = zscore(maps1, 1)
        if zscore0 == 'split':
            maps1 = zscore(maps1, 0)

        if n_row > 1:
            for row_idx in range(1, n_row):
                # load maps2
                maps2 = nib.load(data_files[row_idx]).get_fdata()
                maps2 = maps2[:, roi_idx_map]

                non_nan_idx_arr2 = ~np.isnan(maps2)
                subj_idx_vec2 = np.all(non_nan_idx_arr2, 1)
                assert np.all(subj_idx_vec2 == np.any(non_nan_idx_arr2, 1))
                maps2 = maps2[subj_idx_vec2]
                print(f'{data_files[row_idx]}\n{maps2.shape}\n')

                if col_idx == 0:
                    n_subjects.append(n_subjects[-1] + maps2.shape[0])
                    subj_idx_vecs.append(subj_idx_vec2)
                else:
                    np.all(subj_idx_vecs[row_idx] == subj_idx_vec2)

                # zscore
                if zscore1 == 'split':
                    maps2 = zscore(maps2, 1)
                if zscore0 == 'split':
                    maps2 = zscore(maps2, 0)

                # concatenate
                maps1 = np.r_[maps1, maps2]
        data.append(maps1)
    data = np.concatenate(data, 1)
    # zscore
    if zscore1 == 'whole':
        data = zscore(data, 1)
    if zscore0 == 'whole':
        data = zscore(data, 0)

    # calculate
    pca = PCA(n_components=n_component, random_state=random_state)
    if axis == 'vertex':
        pca.fit(data)
        Y = pca.transform(data)
        csv_data = Y
        cii_data = pca.components_
    elif axis == 'subject':
        data = data.T
        pca.fit(data)
        Y = pca.transform(data)
        csv_data = pca.components_.T
        cii_data = Y.T
    else:
        raise ValueError('Invalid axis:', axis)

    # save
    for row_idx in range(n_row):
        subj_idx_vec = subj_idx_vecs[row_idx]
        csv_data_tmp = np.ones(
            (len(subj_idx_vec), csv_data.shape[1]), np.float64) * np.nan
        s_idx = n_subjects[row_idx]
        e_idx = n_subjects[row_idx + 1]
        csv_data_tmp[subj_idx_vec] = csv_data[s_idx:e_idx]
        df = pd.DataFrame(data=csv_data_tmp, columns=component_names)
        df.to_csv(csv_files[row_idx], index=False)

    maps = np.ones((n_component, LR_count_32k), np.float64) * np.nan
    for col_idx in range(n_col):
        s_idx = n_vertices[col_idx]
        e_idx = n_vertices[col_idx + 1]
        maps[:, roi_idx_maps[col_idx]] = cii_data[:, s_idx:e_idx]
    save2cifti(cii_file, maps, reader.brain_models(), component_names)

    pkl.dump(pca, open(pkl_file, 'wb'))


def pca_mf1(data_files, atlas_names, roi_names, n_component, axis,
            zscore0, zscore1, csv_files, cii_files, pkl_file):
    """
    对n_subj x n_vtx形状的矩阵进行PCA降维
    adapted for dealing with multi files

    Args:
        data_files (str or array-like): end with .dscalar.nii
            shape=(n_subj, LR_count_32k)
            If is array-like, must be 2D.
        atlas_names (str or array-like): include ROIs' labels and mask map
            If is array-like, must be 1D, corresponding to
            columns of data_files.
        roi_names (str or array-like): 决定选用哪个区域内的顶点来参与PCA
            If is array-like, must be 1D, corresponding to atlas_names.
        n_component (int): the number of components
        axis (str): vertex | subject
            vertex: 对顶点数量进行降维，得到几个主成分时间序列，
            观察某个主成分在各顶点上的权重，刻画其空间分布。
            subject: 对被试数量进行降维，得到几个主成分map，
            观察某个主成分在各被试上的权重，按年龄排序即可得到时间序列。
        zscore0 (str): None, split, whole
        zscore1 (str): None, split, whole
        csv_files (str or array-like): n_subj x n_component
            If is array-like, must be 1D, corresponding to rows of data_files.
        cii_files (str or array-like): n_component x LR_count_32k
            If is array-like, must be 1D, corresponding to
            columns of data_files.
        pkl_file (str): fitted PCA model
    """
    # prepare
    if isinstance(data_files, str):
        data_files = np.atleast_2d(data_files)
    else:
        data_files = np.asarray(data_files)
        assert data_files.ndim == 2
    n_row, n_col = data_files.shape

    if isinstance(atlas_names, str):
        atlas_names = np.atleast_1d(atlas_names)
    else:
        atlas_names = np.asarray(atlas_names)
        assert atlas_names.ndim == 1
    assert len(atlas_names) == n_col

    if isinstance(roi_names, str):
        roi_names = np.atleast_1d(roi_names)
    else:
        roi_names = np.asarray(roi_names)
        assert roi_names.ndim == 1
    assert len(roi_names) == n_col

    if isinstance(csv_files, str):
        csv_files = np.atleast_1d(csv_files)
    else:
        csv_files = np.asarray(csv_files)
        assert csv_files.ndim == 1
    assert len(csv_files) == n_row

    if isinstance(cii_files, str):
        cii_files = np.atleast_1d(cii_files)
    else:
        cii_files = np.asarray(cii_files)
        assert cii_files.ndim == 1
    assert len(cii_files) == n_col

    component_names = [f'C{i}' for i in range(1, n_component+1)]

    # prepare data
    bm_list = []
    roi_idx_maps = []
    n_vertices = [0]
    n_subjects = [0]
    data = []
    for col_idx in range(n_col):
        # load maps1
        reader = CiftiReader(data_files[0, col_idx])
        maps1 = reader.get_data()
        bm_list.append(reader.brain_models())

        # load atlas and mask maps1
        atlas = Atlas(atlas_names[col_idx])
        assert atlas.maps.shape == (1, LR_count_32k)
        roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_names[col_idx]]
        maps1 = maps1[:, roi_idx_map]
        roi_idx_maps.append(roi_idx_map)
        n_vertices.append(n_vertices[-1] + maps1.shape[1])
        if col_idx == 0:
            n_subjects.append(n_subjects[-1] + maps1.shape[0])

        # zscore
        if zscore1 == 'split':
            maps1 = zscore(maps1, 1)
        if zscore0 == 'split':
            maps1 = zscore(maps1, 0)

        if n_row > 1:
            for row_idx in range(1, n_row):
                # load maps2
                maps2 = nib.load(data_files[row_idx, col_idx]).get_fdata()
                maps2 = maps2[:, roi_idx_map]
                if col_idx == 0:
                    n_subjects.append(n_subjects[-1] + maps2.shape[0])

                # zscore
                if zscore1 == 'split':
                    maps2 = zscore(maps2, 1)
                if zscore0 == 'split':
                    maps2 = zscore(maps2, 0)

                # concatenate
                maps1 = np.r_[maps1, maps2]
        data.append(maps1)
    data = np.concatenate(data, 1)
    # zscore
    if zscore1 == 'whole':
        data = zscore(data, 1)
    if zscore0 == 'whole':
        data = zscore(data, 0)

    # calculate
    pca = PCA(n_components=n_component)
    if axis == 'vertex':
        pca.fit(data)
        Y = pca.transform(data)
        csv_data = Y
        cii_data = pca.components_
    elif axis == 'subject':
        data = data.T
        pca.fit(data)
        Y = pca.transform(data)
        csv_data = pca.components_.T
        cii_data = Y.T
    else:
        raise ValueError('Invalid axis:', axis)

    # save
    for row_idx in range(n_row):
        s_idx = n_subjects[row_idx]
        e_idx = n_subjects[row_idx + 1]
        df = pd.DataFrame(data=csv_data[s_idx:e_idx, :],
                          columns=component_names)
        df.to_csv(csv_files[row_idx], index=False)
    for col_idx in range(n_col):
        s_idx = n_vertices[col_idx]
        e_idx = n_vertices[col_idx + 1]
        maps = np.ones((n_component, LR_count_32k), np.float64) * np.nan
        maps[:, roi_idx_maps[col_idx]] = cii_data[:, s_idx:e_idx]
        save2cifti(cii_files[col_idx], maps, bm_list[col_idx], component_names)
    pkl.dump(pca, open(pkl_file, 'wb'))


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


def merge_by_age(data_file, info_file, out_name):
    """
    对每个column，计算跨同一年龄被试的平均和sem，分别保存在
    out_name-mean.csv, out_name-sem.csv中

    Args:
        data_file (str): end with .csv
            shape=(n_subj, n_col)
        info_file (str): subject info file
        out_name (str): filename to save
    """
    # prepare
    df = pd.read_csv(data_file)
    n_col = df.shape[1]

    info_df = pd.read_csv(info_file)
    ages = np.array(info_df['age in years'])
    ages_uniq = np.unique(ages)
    n_age = len(ages_uniq)

    # calculate
    means = np.zeros((n_age, n_col), np.float64)
    sems = np.zeros((n_age, n_col), np.float64)
    for age_idx, age in enumerate(ages_uniq):
        data = np.array(df.loc[ages == age])
        means[age_idx] = np.mean(data, 0)
        sems[age_idx] = sem(data, 0)

    # save
    mean_df = pd.DataFrame(means, ages_uniq, df.columns)
    mean_df.to_csv(f'{out_name}-mean.csv')
    sem_df = pd.DataFrame(sems, ages_uniq, df.columns)
    sem_df.to_csv(f'{out_name}-sem.csv')


def calc_map_corr(data_file1, data_file2, atlas_name, roi_name, out_file,
                  map_names1=None, map_names2=None, index=False):
    """
    计算指定atlas的ROI内的data1 maps和data2 maps的相关
    存出的CSV文件的index是map_names1, column是map_names2

    Args:
        data_file1 (str): end with .dscalar.nii
            shape=(n_map1, LR_count_32k)
        data_file2 (str): end with .dscalar.nii
            shape=(n_map2, LR_count_32k)
        atlas_name (str):
        roi_name (str):
        out_file (str):
        map_names1 (list, optional): Defaults to None.
            If is None, use map names in data1_file.
        map_names2 (list, optional): Defaults to None.
            If is None, use map names in data2_file.
        index (bool, optional): Defaults to False.
            Save index of DataFrame or not
    """
    # prepare
    reader1 = CiftiReader(data_file1)
    reader2 = CiftiReader(data_file2)
    data_maps1 = reader1.get_data()
    data_maps2 = reader2.get_data()
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]
    data_maps1 = data_maps1[:, roi_idx_map]
    data_maps2 = data_maps2[:, roi_idx_map]

    if map_names1 is None:
        map_names1 = reader1.map_names()
    else:
        assert len(map_names1) == data_maps1.shape[0]

    if map_names2 is None:
        map_names2 = reader2.map_names()
    else:
        assert len(map_names2) == data_maps2.shape[0]

    # calculate
    corrs = 1 - cdist(data_maps1, data_maps2, 'correlation')

    # save
    df = pd.DataFrame(corrs, map_names1, map_names2)
    df.to_csv(out_file, index=index)


def vtx_corr_col(data_file1, atlas_name, roi_name,
                 data_file2, column, idx_col, out_file):
    """
    计算data_file1中指定atlas ROI内的所有顶点序列和data_file2中指定column序列的相关

    Args:
        data_file1 (str): end with .dscalar.nii
            shape=(N, LR_count_32k)
        atlas_name (str):
        roi_name (str):
        data_file2 (str): end with .csv
            shape=(N, n_col)
        column (str):
        idx_col (int): specify index column of csv file
            If None, means no index column.
    """
    # prepare
    reader = CiftiReader(data_file1)
    maps = reader.get_data()
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]
    maps = maps[:, roi_idx_map]

    df = pd.read_csv(data_file2, index_col=idx_col)
    col_vec = np.array(df[column])
    col_vec = np.expand_dims(col_vec, 0)

    # calculate
    data = np.ones((1, LR_count_32k)) * np.nan
    data[0, roi_idx_map] = 1 - cdist(col_vec, maps.T, 'correlation')[0]

    # save
    save2cifti(out_file, data, reader.brain_models())


def row_corr_row(data_file1, cols1, idx_col1,
                 data_file2, cols2, idx_col2,
                 out_file, index=False, columns=None):
    """
    Calculate the correlation between each row in data_file1 and data_file2
    存出的CSV文件的index是index of data_file1, column是index of data_file2

    Args:
        data_file1 (str): end with .csv
            shape=(N, n_col)
        cols1 (sequence): columns of data_file1
            If None, use all columns.
        idx_col1 (int, None): specify index column of csv file
            If None, means no index column.
        data_file2 (str): end with .csv
            shape=(N, n_col)
        cols2 (sequence): columns of data_file2
            If None, use all columns.
        idx_col2 (int, None): specify index column of csv file
            If None, means no index column.
        out_file (str): end with .csv
        index (bool, optional): Defaults to False.
            Save index of DataFrame or not
        columns (sequence, optional): Defaults to None.
            If None, use index of data_file2 as columns to save out.
    """
    # prepare
    df1 = pd.read_csv(data_file1, index_col=idx_col1)
    if cols1 is None:
        data1 = np.array(df1)
    else:
        data1 = np.array(df1[cols1])
    df2 = pd.read_csv(data_file2, index_col=idx_col2)
    if cols2 is None:
        data2 = np.array(df2)
    else:
        data2 = np.array(df2[cols2])

    # calculate
    data = 1 - cdist(data1, data2, 'correlation')
    if columns is None:
        columns = df2.index
    else:
        assert len(columns) == len(df2.index)
    df = pd.DataFrame(data, index=df1.index, columns=columns)

    # save
    df.to_csv(out_file, index=index)


def col_operate_col(data_file, cols, idx_col,
                    operation_type, operation_method,
                    out_file, index=False):
    """
    Do operation between each two columns. For example, col2-col1.

    Args:
        data_file (str): end with .csv
            shape=(N, n_col)
        cols (sequence): columns of data_file
            If None, use all columns.
        idx_col (int, None): specify index column of csv file
            If None, means no index column.
        operation_type (str): adjacent_pair, all_pair
            The latter (+, -, *, /) the former
        operation_method (str): +, -, *, /
        out_file (str): end with .csv
        index (bool, optional): Defaults to False.
            Save index of DataFrame or not
    """
    # prepare
    df = pd.read_csv(data_file, index_col=idx_col)
    if cols is None:
        cols = df.columns.to_list()

    # calculate
    out_df = pd.DataFrame(index=df.index)
    if operation_type == 'adjacent_pair':
        for i, col1 in enumerate(cols[:-1]):
            col2 = cols[i+1]
            if operation_method == '+':
                col = f'{col2}+{col1}'
                out_df[col] = df[col2] + df[col1]
            elif operation_method == '-':
                col = f'{col2}-{col1}'
                out_df[col] = df[col2] - df[col1]
            elif operation_method == '*':
                col = f'{col2}*{col1}'
                out_df[col] = df[col2] * df[col1]
            elif operation_method == '/':
                col = f'{col2}/{col1}'
                out_df[col] = df[col2] / df[col1]
            else:
                raise ValueError('Not supported operation_method:',
                                 operation_method)
    elif operation_type == 'all_pair':
        for i, col1 in enumerate(cols[:-1]):
            for col2 in cols[i + 1:]:
                if operation_method == '+':
                    col = f'{col2}+{col1}'
                    out_df[col] = df[col2] + df[col1]
                elif operation_method == '-':
                    col = f'{col2}-{col1}'
                    out_df[col] = df[col2] - df[col1]
                elif operation_method == '*':
                    col = f'{col2}*{col1}'
                    out_df[col] = df[col2] * df[col1]
                elif operation_method == '/':
                    col = f'{col2}/{col1}'
                    out_df[col] = df[col2] / df[col1]
                else:
                    raise ValueError('Not supported operation_method:',
                                     operation_method)
    else:
        raise ValueError('Not supported operation_type:', operation_type)

    # save
    out_df.to_csv(out_file, index=index)


def map_operate_map(data_file1, data_file2, operation_method, out_file):
    """
    Do operation between two CIFTI data.

    Args:
        data_file1 (str): end with .dscalar.nii
            shape=(n_map1, LR_count_32k)
        data_file2 (str): end with .dscalar.nii
            shape=(n_map2, LR_count_32k)
            If n_map2 == n_map1, do operation between
            one-to-one corresponding maps.
            If n_map2 == 1, do operation between each map1 and the map2.
            else, error.
        operation_method (str): +, -, *, /
        out_file (str): end with .dscalar.nii
            shape=(n_map1, LR_count_32k)
    """
    # prepare
    reader1 = CiftiReader(data_file1)
    reader2 = CiftiReader(data_file2)
    data_maps1 = reader1.get_data()
    data_maps2 = reader2.get_data()

    # calculate
    if operation_method == '+':
        data = data_maps1 + data_maps2
    elif operation_method == '-':
        data = data_maps1 - data_maps2
    elif operation_method == '*':
        data = data_maps1 * data_maps2
    elif operation_method == '/':
        data = data_maps1 / data_maps2
    else:
        raise ValueError('not supported operation_method:', operation_method)

    # save
    save2cifti(out_file, data, reader1.brain_models(), reader1.map_names())


def mask_maps(data_file, atlas_name, roi_names, out_file):
    """
    把data map在指定atlas的ROI以外的部分全赋值为nan

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_map, LR_count_32k)
        atlas_name (str):
        roi_name (str):
        out_file (str):
    """
    # prepare
    reader = CiftiReader(data_file)
    data = reader.get_data()
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    mask_arr = np.zeros(LR_count_32k, bool)
    for roi_name in roi_names:
        mask_arr = np.logical_or(mask_arr, atlas.maps[0] == atlas.roi2label[roi_name])

    # calculate
    data[:, ~mask_arr] = np.nan

    # save
    save2cifti(out_file, data, reader.brain_models(), reader.map_names())


def polyfit(data_file, info_file, deg, out_file):
    """
    对时间序列进行多项式拟合

    Args:
        data_file (str): .csv | .dscalar.nii
            If is .csv, fit each column with ages.
            If is .dscalar.nii, fit each vertex with ages.
        info_file (str): .csv
        deg (int): degree of polynomial
        out_file (str): file to output
            same postfix with data_file
    """
    # prepare
    if data_file.endswith('.csv'):
        assert out_file.endswith('.csv')
        reader = pd.read_csv(data_file)
        data = np.array(reader)
    elif data_file.endswith('.dscalar.nii'):
        assert out_file.endswith('.dscalar.nii')
        reader = CiftiReader(data_file)
        data = reader.get_data()
    else:
        raise ValueError(1)

    # calculate
    ages = np.array(pd.read_csv(info_file)['age in years'])
    coefs = np.polyfit(ages, data, deg)
    n_row = coefs.shape[0]
    if deg == 1:
        assert n_row == 2
        row_names = ['coef', 'intercept']
    else:
        row_names = [''] * n_row
        raise Warning("Can't provide row names for degree:", deg)

    # save
    if out_file.endswith('.csv'):
        df = pd.DataFrame(coefs, row_names, reader.columns)
        df.to_csv(out_file)
    else:
        save2cifti(out_file, coefs, reader.brain_models(), row_names)


def calc_alff(x, tr, axis=0, low_freq_band=(0.01, 0.08),
              linear_detrend=True):
    """
    Calculate amplitude of low-frequency fluctuation (ALFF) and
    Fractional ALFF (fALFF)

    Parameters:
    ----------
        x (array-like): Array to Fourier transform.
        tr (float): Repetition time (second)
        axis (int): Default is the first axis (i.e., axis=0).
            Axis along which the fft is computed.
        low_freq_band (tuple):  low frequency band (Hz)
        linear_detrend (bool): do linear detrend or not

    Returns:
    -------
        alff (ndarray):
        falff (ndarray):
    """
    x = np.asarray(x)
    if axis != 0:
        x = np.swapaxes(x, 0, axis)
    if linear_detrend:
        x = detrend(x, axis=0, type='linear')

    # fast fourier transform
    fft_array = fft(x, axis=0)
    # get fourier transform sample frequencies
    freq_scale = fftfreq(x.shape[0], tr)
    # calculate power of half frequency bands
    half_band_idx = (0.0 <= freq_scale) & (freq_scale <= (1 / tr) / 2)
    half_band_array = fft_array[half_band_idx]
    half_band_power = np.sqrt(np.absolute(half_band_array))
    half_band_scale = freq_scale[half_band_idx]
    low_freq_band_idx = (low_freq_band[0] <= half_band_scale) & \
                        (half_band_scale <= low_freq_band[1])

    # calculate ALFF or fALFF
    # ALFF: sum of low band power
    # fALFF: ratio of Alff to total power
    alff = np.sum(half_band_power[low_freq_band_idx], axis=0)
    falff = alff / np.sum(half_band_power, axis=0)

    return alff, falff
