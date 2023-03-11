import time
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from scipy.signal import detrend
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA, FactorAnalysis, DictionaryLearning,\
    FastICA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from magicbox.io.io import CiftiReader, save2cifti, GiftiReader
from magicbox.graph.triangular_mesh import get_n_ring_neighbor
from cxy_visual_dev.lib.predefine import Atlas, LR_count_32k, proj_dir,\
    hemi2stru, s1200_midthickness_L, s1200_midthickness_R


def cat_data_from_cifti(fpaths, cat_shape, vtx_masks=None, map_mask=None,
                        zscore0=None, zscore1=None):
    """
    对来自多个cifti文件中的数据进行拼接，同时可以进行zscore

    Args:
        fpaths (list): a list of file paths
            shape=(n_map, n_vtx)
        cat_shape (tuple): (n_row, n_col)
            按照这个形状，以行优先的顺序拼接数据
        vtx_masks (list, optional): a list of 1D index arraies
            指定大脑区域
        map_mask (ndarray, optional): 1D index array
            指定使用的map
            如果是bool类型，则长度为n_map
            如果是int类型，则值域为[0, n_map-1]
        zscore0 (str, optional): split, whole
            split: do zscore across subjects of each row
            whole: do zscore across subjects of all rows
            split-minmax: do MinMaxScale across subjects of each row
            whole-minmax: do MinMaxScale across subjects of all rows
        zscore1 (str, optional): split, whole
            split: do zscore across vertices of each mask of each column
            whole: do zscore across vertices of all columns
            split-minmax: do MinMaxScale across vertices of each mask of each column
            whole-minmax: do MinMaxScale across vertices of all columns

    Returns:
        [tuple]: (data, n_vertices, n_maps, reader)
            data: 拼接后的数据
            n_vertices：记录每次纵向拼接的顶点数量与上一次记录的加和
            n_maps: 记录每次横向拼接的map数量与上一次记录的加和
            reader: 返回第一个cifti文件的CiftiReader实例
    """
    # check
    n_row, n_col = cat_shape
    assert len(fpaths) == n_row * n_col

    # prepare data
    reader = None
    n_vertices = [0]
    n_maps = [0]
    data = []
    f_idx = 0
    for _ in range(n_row):
        data1 = []
        for _ in range(n_col):
            fpath = fpaths[f_idx]

            # load maps
            if f_idx == 0:
                reader = CiftiReader(fpath)
                maps = reader.get_data()
            else:
                maps = nib.load(fpath).get_fdata()

            if vtx_masks is None:
                data2 = maps
                n_vertices.append(n_vertices[-1] + data2.shape[1])
                if zscore1 == 'split':
                    data2 = zscore(data2, 1)
                elif zscore1 == 'split-minmax':
                    data2 = MinMaxScaler(
                        feature_range=(0, 1)).fit_transform(data2.T).T
            else:
                # extract masked data
                data2 = []
                for vtx_mask in vtx_masks:
                    maps_mask = maps[:, vtx_mask]
                    n_vertices.append(n_vertices[-1] + maps_mask.shape[1])
                    if zscore1 == 'split':
                        maps_mask = zscore(maps_mask, 1)
                    elif zscore1 == 'split-minmax':
                        maps_mask = MinMaxScaler(
                            feature_range=(0, 1)).fit_transform(maps_mask.T).T
                    data2.append(maps_mask)
                data2 = np.concatenate(data2, 1)

            # update
            data1.append(data2)
            f_idx += 1

        data1 = np.concatenate(data1, 1)
        if map_mask is not None:
            data1 = data1[map_mask]
        if zscore0 == 'split':
            data1 = zscore(data1, 0)
        elif zscore0 == 'split-minmax':
            data1 = MinMaxScaler(
                feature_range=(0, 1)).fit_transform(data1)
        n_maps.append(n_maps[-1] + data1.shape[0])

        # update
        data.append(data1)

    data = np.concatenate(data, 0)
    if zscore1 == 'whole':
        data = zscore(data, 1)
    elif zscore1 == 'whole-minmax':
        data = MinMaxScaler(
            feature_range=(0, 1)).fit_transform(data.T).T

    if zscore0 == 'whole':
        data = zscore(data, 0)
    elif zscore0 == 'whole-minmax':
        data = MinMaxScaler(
            feature_range=(0, 1)).fit_transform(data)

    return data, n_vertices, n_maps, reader


def zscore_cii_masked(src_file, mask, out_file):
    """
    zscore maps across vertices in the mask

    Args:
        src_file (str): end with .dscalar.nii
            shape=(n_map, n_vtx)
        mask (1D array)
        out_file (str): end with .dscalar.nii
            shape=(n_map, n_vtx)
    """
    reader = CiftiReader(src_file)
    maps = reader.get_data()

    data = np.ones_like(maps) * np.nan
    maps = maps[:, mask]
    data[:, mask] = zscore(maps, 1)

    save2cifti(out_file, data, reader.brain_models(),
               reader.map_names(), reader.volume)


def zscore_cii(src_file, axis, out_file):
    """
    zscore across maps (axis=0) or vertices (axis=1)

    Args:
        src_file (str): end with .dscalar.nii
            shape=(n_map, n_vtx)
        axis (int): 0 | 1
        out_file (str): end with .dscalar.nii
            shape=(n_map, n_vtx)
    """
    reader = CiftiReader(src_file)
    maps = zscore(reader.get_data(), axis)
    save2cifti(out_file, maps, reader.brain_models(),
               reader.map_names(), reader.volume)


def stack_cii(src_files, out_file):
    """
    Stack maps from multiple CIFTI files.

    Args:
        src_files (strings): Each string is end with .dscalar.nii
            shape=(n_map, n_vtx)
        out_file (str): end with .dscalar.nii
            shape=(n_map_all, n_vtx)
    """
    reader = CiftiReader(src_files[0])
    maps = reader.get_data()
    map_names = reader.map_names()
    for src_file in src_files[1:]:
        reader_tmp = CiftiReader(src_file)
        maps = np.r_[maps, reader_tmp.get_data()]
        map_names.extend(reader_tmp.map_names())
    save2cifti(out_file, maps, reader.brain_models(), map_names, reader.volume)


def smooth_cii(src_file, hemi, n_ring, out_file):
    """
    smooth cerebral cortex
    忽略值为NAN的顶点，将非NAN顶点赋值为其本身与非NAN近邻的均值

    Args:
        src_file (str): .dscalar.nii
        hemi (str): lh or rh
        n_ring (int): smoothness
        out_file (str): .dscalar.nii
    """
    hemi2geo_file = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}

    # get map info
    reader = CiftiReader(src_file)
    full_shape = reader.full_data.shape
    bm = reader.brain_models([hemi2stru[hemi]])[0]
    offset, count = bm.index_offset, bm.index_count
    src_maps = reader.get_data(hemi2stru[hemi], True)
    idx2vtx = reader.get_stru_pos(hemi2stru[hemi])[-1]

    # get vertex neighbors
    faces = GiftiReader(hemi2geo_file[hemi]).faces
    vtx2neighbors = get_n_ring_neighbor(faces, n_ring)

    # calculating
    n_map, n_vtx = src_maps.shape
    out_maps = np.ones(full_shape) * np.nan
    hemi_maps = np.ones((n_map, n_vtx)) * np.nan
    for vtx in range(n_vtx):
        vertices = list(vtx2neighbors[vtx])
        vertices.append(vtx)
        for map_idx in range(n_map):
            if np.isnan(src_maps[map_idx, vtx]):
                continue
            hemi_maps[map_idx, vtx] = np.nanmean(src_maps[map_idx, vertices])
    out_maps[:, offset:(offset+count)] = hemi_maps[:, idx2vtx]

    # save out
    save2cifti(out_file, out_maps, reader.brain_models(),
               reader.map_names(), reader.volume)


def decompose(fpaths, cat_shape, method, axis, csv_files, cii_files,
              pkl_file, vtx_masks=None, map_mask=None, zscore0=None,
              zscore1=None, n_component=None, random_state=None):
    """
    将来自多个cifti文件n_map x n_vtx的数据拼接，并进行成分分解

    Args:
        fpaths (list): a list of file paths
            详见cat_data_from_cifti
        cat_shape (tuple): (n_row, n_col)
            详见cat_data_from_cifti
        method (str): PCA | FA | FA1 | DicL | ICA
            'PCA': Principal Component Analysis
            'FA': Factor Analysis
            'FA1': Factor Analysis with 'varimax' rotation
            'DicL': Dictionary Learning
            'ICA': Independent Component Analysis
        axis (int): 0 | 1
            0: 对map数量进行降维，得到几个主成分map，
                观察某个主成分在各map上的权重
            1: 对顶点数量进行降维，得到几个主成分时间序列，
                观察某个主成分在各顶点上的权重，可刻画其空间分布
        csv_files (list): a list of CSV files
            shape=(n_map, n_component)
            len(csv_files)=n_row
        cii_files (list): a list of .dscalar.nii files
            shape=(n_component, n_vtx)
            len(cii_files)=n_col
        pkl_file (str): fitted model
        vtx_masks (list, optional): a list of 1D index arraies
            详见cat_data_from_cifti
        map_mask (ndarray, optional): 1D index array
            详见cat_data_from_cifti
        zscore0 (str, optional): split, whole
            详见cat_data_from_cifti
        zscore1 (str, optional): split, whole
            详见cat_data_from_cifti
        n_component (int, optional): the number of components
        random_state (int, optional):
    """
    # prepare
    n_row, n_col = cat_shape
    assert len(csv_files) == n_row
    assert len(cii_files) == n_col
    data, n_vertices, n_maps, reader = cat_data_from_cifti(
        fpaths, cat_shape, vtx_masks, map_mask, zscore0, zscore1)
    n_map, n_vtx = reader.full_data.shape

    # calculate
    if method == 'PCA':
        transformer = PCA(n_components=n_component, random_state=random_state)
    elif method == 'FA1':
        transformer = FactorAnalysis(
            n_components=n_component, rotation='varimax', random_state=random_state)
    elif method == 'FA':
        transformer = FactorAnalysis(
            n_components=n_component, rotation=None, random_state=random_state)
    elif method == 'DicL':
        transformer = DictionaryLearning(n_components=n_component, random_state=random_state)
    elif method == 'ICA':
        transformer = FastICA(n_components=n_component, random_state=random_state)
    else:
        raise ValueError('not supported method:', method)
    if axis == 1:
        transformer.fit(data)
        Y = transformer.transform(data)
        csv_data = Y
        cii_data = transformer.components_
    elif axis == 0:
        data = data.T
        transformer.fit(data)
        Y = transformer.transform(data)
        csv_data = transformer.components_.T
        cii_data = Y.T
    else:
        raise ValueError('Invalid axis:', axis)

    # save
    if n_component is None:
        n_component = csv_data.shape[1]
    else:
        assert n_component == csv_data.shape[1]
    component_names = [f'C{i}' for i in range(1, n_component+1)]

    for row_idx in range(n_row):
        s_idx = n_maps[row_idx]
        e_idx = n_maps[row_idx + 1]
        if map_mask is None:
            csv_data_tmp = csv_data[s_idx:e_idx]
        else:
            csv_data_tmp = np.ones((n_map, n_component), np.float64) * np.nan
            csv_data_tmp[map_mask] = csv_data[s_idx:e_idx]
        df = pd.DataFrame(data=csv_data_tmp, columns=component_names)
        df.to_csv(csv_files[row_idx], index=False)

    i = 0
    for col_idx in range(n_col):
        if vtx_masks is None:
            s_idx = n_vertices[i]
            e_idx = n_vertices[i + 1]
            cii_data_tmp = cii_data[:, s_idx:e_idx]
            i += 1
        else:
            cii_data_tmp = np.ones((n_component, n_vtx), np.float64) * np.nan
            for vtx_mask in vtx_masks:
                s_idx = n_vertices[i]
                e_idx = n_vertices[i + 1]
                cii_data_tmp[:, vtx_mask] = cii_data[:, s_idx:e_idx]
                i += 1
        save2cifti(cii_files[col_idx], cii_data_tmp, reader.brain_models(),
                   component_names)

    pkl.dump(transformer, open(pkl_file, 'wb'))


def transform(fpaths, cat_shape, model_file, csv_files,
              vtx_masks=None, map_mask=None, zscore0=None):
    """
    将来自多个cifti文件n_map x n_vtx的数据拼接，并用现成的model做transform
    目前只支持对顶点数量的降维

    Args:
        fpaths (list): a list of file paths
            详见cat_data_from_cifti
        cat_shape (tuple): (n_row, n_col)
            详见cat_data_from_cifti
        model_file (str):
        csv_files (list): a list of CSV files
            shape=(n_map, n_component)
            len(csv_files)=n_row
        vtx_masks (list, optional): a list of 1D index arraies
            详见cat_data_from_cifti
        map_mask (ndarray, optional): 1D index array
            详见cat_data_from_cifti
        zscore0 (str, optional): split, whole
            详见cat_data_from_cifti
    """
    n_row, n_col = cat_shape
    assert len(csv_files) == n_row
    data, n_vertices, n_maps, reader = cat_data_from_cifti(
        fpaths, cat_shape, vtx_masks, map_mask, zscore0)
    n_map, n_vtx = reader.full_data.shape

    transformer = pkl.load(open(model_file, 'rb'))
    csv_data = transformer.transform(data)
    n_component = csv_data.shape[1]
    component_names = [f'C{i}' for i in range(1, n_component+1)]
    for row_idx in range(n_row):
        s_idx = n_maps[row_idx]
        e_idx = n_maps[row_idx + 1]
        if map_mask is None:
            csv_data_tmp = csv_data[s_idx:e_idx]
        else:
            csv_data_tmp = np.ones((n_map, n_component), np.float64) * np.nan
            csv_data_tmp[map_mask] = csv_data[s_idx:e_idx]
        df = pd.DataFrame(data=csv_data_tmp, columns=component_names)
        df.to_csv(csv_files[row_idx], index=False)


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


def linear_fit1(X_list, feat_names, Y, trg_names, score_metric,
                out_file, standard_scale=True):
    """
    每个X的形状相同，有多少列就迭代多少次
    每次迭代用所有X中对应的列作为features，去拟合Y的各列。
    得到每次迭代对每个target的拟合分数，系数，截距

    Args:
        X_list (list): a list of 2D arrays
        feat_names (strings): feature names
        Y (2D array): target array
        trg_names (strings): target names
        score_metric (str): 目前只支持R2
        out_file (str):
            If 'df', return df
            If ends with '.csv', save to CSV file
        standard_scale (bool, optional):
            是否在线性回归之前做特征内的标准化
    """
    n_feat = len(X_list)
    n_iter = X_list[0].shape[1]
    assert n_feat == len(feat_names)

    n_trg = Y.shape[1]
    assert n_trg == len(trg_names)
    n_sample = Y.shape[0]

    # fitting
    coefs = np.zeros((n_iter, n_trg, n_feat), np.float64)
    intercepts = np.zeros((n_iter, n_trg), np.float64)
    scores = np.zeros((n_iter, n_trg), np.float64)
    for iter_idx in range(n_iter):
        time1 = time.time()
        X = np.zeros((n_sample, n_feat), np.float64)
        for feat_idx in range(n_feat):
            X[:, feat_idx] = X_list[feat_idx][:, iter_idx]

        if standard_scale:
            model = Pipeline([('preprocesser', StandardScaler()),
                             ('regressor', LinearRegression())])
            model.fit(X, Y)
            coefs[iter_idx] = model.named_steps['regressor'].coef_
            intercepts[iter_idx] = model.named_steps['regressor'].intercept_
        else:
            model = LinearRegression()
            model.fit(X, Y)
            coefs[iter_idx] = model.coef_
            intercepts[iter_idx] = model.intercept_

        Y_pred = model.predict(X)
        if score_metric == 'R2':
            scores[iter_idx] = [
                r2_score(Y[:, i], Y_pred[:, i]) for i in range(n_trg)]
        else:
            raise ValueError('not supported score metric')

        print(f'Finished {iter_idx + 1}/{n_iter}, '
              f'cost {time.time() - time1} seconds.')

    # save
    df = pd.DataFrame()
    for trg_idx, trg_name in enumerate(trg_names):
        for feat_idx, feat_name in enumerate(feat_names):
            df[f'coef_{trg_name}_{feat_name}'] = coefs[:, trg_idx, feat_idx]
        df[f'score_{trg_name}'] = scores[:, trg_idx]
        df[f'intercept_{trg_name}'] = intercepts[:, trg_idx]
    if out_file == 'df':
        return df
    elif out_file.endswith('.csv'):
        df.to_csv(out_file, index=False)
    else:
        raise ValueError('not supported out_file')


class AgeSlideWindow:
    """
    按照岁数升序排序，将被试划分到不同年龄窗口中。
    """

    def __init__(self, dataset_name, width, step, merge_remainder):
        """
        Args:
            dataset_name (str): "HCPD", "HCPA", or "HCPY"
            width (int): window width
                with units of subjects
            step (int): step size
                with uints of subjects
            merge_remainder (bool):
                Merge remainder subjects whose number is less than step
                into the last window or not.
                If the number of remainder subjects is 0, this parameter
                will be ignored (set as False forcibly).
        """
        # subject information
        self.dataset_name = dataset_name
        self.subj_info = pd.read_csv(pjoin(
            proj_dir, f'data/HCP/{dataset_name}_SubjInfo.csv'))
        self.n_subj = self.subj_info.shape[0]
        if dataset_name == 'HCPY':
            self.sorted_indices = np.argsort(self.subj_info['age in years'])
        else:
            self.sorted_indices = np.argsort(self.subj_info['age in months'])

        # width and step information
        assert width < self.n_subj
        self.width = width
        self.step = step

        # get window start and end indices
        step_space = self.n_subj - self.width
        self.n_remainder = step_space % self.step
        self.merge_remainder = False if self.n_remainder == 0 else merge_remainder
        self.start_indices = list(range(0, step_space, step))
        self.end_indices = [i + width for i in self.start_indices]
        if self.merge_remainder:
            self.end_indices[-1] = self.n_subj
        else:
            self.start_indices.append(step_space)
            self.end_indices.append(self.n_subj)
        self.n_win = len(self.start_indices)

    def get_subj_indices(self, win_id):
        """
        Get subject indices according to window ID

        Args:
            win_id (int): Count from 1
        """
        idx = win_id - 1
        return self.sorted_indices[self.start_indices[idx]:self.end_indices[idx]]

    def get_ages(self, win_id, age_type):
        """
        Get ages according to window ID

        Args:
            win_id (int): Count from 1
            age_type (str): 'month' or 'year'
        """
        indices = self.get_subj_indices(win_id)
        if self.dataset_name == 'HCPY':
            assert age_type == 'year', "HCPY only has age in years."
            ages = self.subj_info.loc[indices, 'age in years']
        else:
            ages = self.subj_info.loc[indices, 'age in months']
            if age_type == 'year':
                ages = ages / 12
            elif age_type == 'month':
                pass
            else:
                raise ValueError('not supported age_type:', age_type)

        return ages

    def plot_sw_age_range(self, age_type='year', figsize=None, out_file='show'):
        """
        Plot age range of each window

        Args:
            age_type (str, optional): Defaults to 'year'.
                'month' or 'year', the unit of the x axis
        """
        win_ids = np.arange(1, self.n_win + 1)
        fig, ax = plt.subplots(figsize=figsize)
        for win_id in win_ids:
            ages = self.get_ages(win_id, age_type)
            age_range = [ages.min(), ages.max()]
            if age_range[0] == age_range[1]:
                ax.scatter(age_range[0], win_id, s=3, c='k')
            else:
                ax.plot(age_range, [win_id, win_id], c='k')
        xticks = (self.get_ages(1, age_type).min(),
                  self.get_ages(int(self.n_win/2), age_type).median(),
                  self.get_ages(self.n_win, age_type).max())
        xticks = [int(i) for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        yticks = (1, self.n_win)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_xlabel(f'Age ({age_type})')
        ax.set_ylabel('Window')
        title = f'{self.dataset_name}_width-{self.width}_setp-{self.step}'
        if self.merge_remainder:
            title += '\nmerge remainder'
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        if out_file == 'show':
            fig.show()
        else:
            fig.savefig(out_file)
