import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from scipy.stats import zscore, sem
from scipy.spatial.distance import cdist
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
    data1 = np.array(df1[cols1])
    df2 = pd.read_csv(data_file2, index_col=idx_col2)
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


def mask_maps(data_file, atlas_name, roi_name, out_file):
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
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]

    # calculate
    data[:, ~roi_idx_map] = np.nan

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
