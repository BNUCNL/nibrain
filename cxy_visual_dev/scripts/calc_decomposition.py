import os
import numpy as np
import pandas as pd
import pickle as pkl
from os.path import join as pjoin
from scipy.stats.stats import zscore
from sklearn.decomposition import PCA
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir,\
    s1200_1096_thickness, s1200_1096_myelin, Atlas,\
    get_rois, LR_count_32k, mmp_map_file
from cxy_visual_dev.lib.algo import decompose, transform

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'decomposition')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def pca_csv(
    fpath, out_csv1, out_csv2, out_pkl, columns=None, row_mask=None,
    zscore0=False, n_component=None, random_state=None
):
    """
    Args:
        fpath (str): a CSV file
        out_csv1 (str): a CSV file
            shape=(n_row, n_component)
        out_csv2 (str): a CSV file
            shape=(n_component, n_col)
        out_pkl (str): a pkl file
            fitted model
        columns (list, optional): a list of column names
            If None, use all columns
        row_mask (ndarray, optional): 1D index array
            If None, use all rows
        zscore0 (bool, optional): Default is False
            If True, do zscore for each column
        n_component (int, optional): the number of components
        random_state (int, optional):
    """
    # prepare
    df = pd.read_csv(fpath)

    if columns is None:
        data = np.array(df)
        columns = df.columns
    else:
        data = np.array(df[columns])
    n_row, n_col = data.shape

    if row_mask is not None:
        data = data[row_mask]

    if zscore0:
        data = zscore(data, 0)

    # calculate
    transformer = PCA(n_components=n_component, random_state=random_state)
    transformer.fit(data)
    Y = transformer.transform(data)
    csv_data1 = Y
    csv_data2 = transformer.components_

    # save
    if n_component is None:
        n_component = csv_data1.shape[1]
    else:
        assert n_component == csv_data1.shape[1]
    component_names = [f'C{i}' for i in range(1, n_component+1)]

    if row_mask is not None:
        csv_data1_tmp = np.ones((n_row, n_component), np.float64) * np.nan
        csv_data1_tmp[row_mask] = csv_data1
        csv_data1 = csv_data1_tmp
    out_df1 = pd.DataFrame(data=csv_data1, columns=component_names)
    out_df1.to_csv(out_csv1, index=False)

    out_df2 = pd.DataFrame(data=csv_data2, columns=columns, index=component_names)
    out_df2.to_csv(out_csv2, index=True)

    pkl.dump(transformer, open(out_pkl, 'wb'))


def csv2cifti(src_file, rois, atlas_name, out_file):
    """
    pca_csv的后续
    把ROI的权重存成cifti格式 方便可视化在大脑上
    """
    # prepare
    df = pd.read_csv(src_file, index_col=0)
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    out_data = np.ones((df.shape[0], LR_count_32k), np.float64) * np.nan

    if rois == 'all':
        rois = df.columns

    # calculate
    for roi in rois:
        mask = atlas.maps[0] == atlas.roi2label[roi]
        for i, idx in enumerate(df.index):
            out_data[i, mask] = df.loc[idx, roi]

    # save
    save2cifti(out_file, out_data, CiftiReader(mmp_map_file).brain_models(),
               df.index.to_list())


if __name__ == '__main__':
    # ===左右V1~3拼一起做PCA，在此之前各ROI内部要做zscore===
    # atlas = Atlas('HCP-MMP')
    # rois = ('L_V1', 'L_V2', 'L_V3', 'R_V1', 'R_V2', 'R_V3')
    # decompose(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness], cat_shape=(2, 1),
    #     method='PCA', axis=0,
    #     csv_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj_M.csv'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj_T.csv')],
    #     cii_files=[pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-LR-V123_zscore1-split_PCA-subj.pkl'),
    #     vtx_masks=[atlas.get_mask(roi)[0] for roi in rois], map_mask=None,
    #     zscore0=None, zscore1='split', n_component=20, random_state=7
    # )
    # ===左右V1~3拼一起做PCA，在此之前各ROI内部要做zscore===

    # 左右视觉皮层拼一起，然后把myelin和thickness的数据沿顶点轴拼起来。做完zscore之后做tPCA
    # 不做跨顶点的zscore，会抹掉被试间的主要差异。
    # 做跨被试的zscore，虽然PCA第一步是为各特征减去自己的均值，但是它会记录并用在后续我要用的HCPD数据上
    # 先做跨被试的zscore可以让它记录的均值全是为0，然后transform HCPD的时候先自行对HCPD的数据做跨被试的zscore
    # atlas = Atlas('HCP-MMP')
    # decompose(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness], cat_shape=(1, 2),
    #     method='PCA', axis=1,
    #     csv_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx.csv')],
    #     cii_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_M.dscalar.nii'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_T.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx.pkl'),
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]],
    #     map_mask=None, zscore0='split', zscore1=None, n_component=20, random_state=7
    # )
    # transform(
    #     fpaths=[
    #         pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #         pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')],
    #     cat_shape=(1, 2),
    #     model_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx.pkl'),
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]],
    #     map_mask=None, zscore0='split',
    #     csv_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore0_PCA-vtx_trans-HCPD.csv')]
    # )

    # 把HCPD的myelin和thickness数据沿顶点轴拼起来（左右视觉皮层拼一起）。
    # 做完跨被试的zscore之后做tPCA
    # atlas = Atlas('HCP-MMP')
    # decompose(
    #     fpaths=[
    #         pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #         pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')],
    #     cat_shape=(1, 2),
    #     method='PCA', axis=1,
    #     csv_files=[pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx.csv')],
    #     cii_files=[
    #         pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx_M.dscalar.nii'),
    #         pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx_T.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPD-M+T_MMP-vis2-LR_zscore0_PCA-vtx.pkl'),
    #     vtx_masks=[
    #         atlas.get_mask(get_rois('MMP-vis2-L'))[0],
    #         atlas.get_mask(get_rois('MMP-vis2-R'))[0]],
    #     map_mask=None, zscore0='split', zscore1=None, n_component=20, random_state=7
    # )

    # 在成人数据上，对右脑HCP-MMP1_visual-cortex3做zscore
    # 联合myelin和thickness做空间PCA
    # atlas = Atlas('HCP-MMP')
    # decompose(
    #     fpaths=[s1200_1096_myelin, s1200_1096_thickness], cat_shape=(2, 1),
    #     method='PCA', axis=0,
    #     csv_files=[
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_M.csv'),
    #         pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_T.csv')],
    #     cii_files=[pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.pkl'),
    #     vtx_masks=[atlas.get_mask(get_rois('MMP-vis3-R'))[0]],
    #     map_mask=None, zscore0=None, zscore1='split', n_component=20, random_state=7
    # )

    # 在成人数据上，对右脑HCP-MMP1_visual-cortex3做zscore
    # 分别对myelin做空间PCA
    atlas = Atlas('HCP-MMP')
    decompose(
        fpaths=[s1200_1096_myelin], cat_shape=(1, 1),
        method='PCA', axis=0,
        csv_files=[pjoin(work_dir, 'HCPY-M_MMP-vis3-R_zscore1_PCA-subj.csv')],
        cii_files=[pjoin(work_dir, 'HCPY-M_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')],
        pkl_file=pjoin(work_dir, 'HCPY-M_MMP-vis3-R_zscore1_PCA-subj.pkl'),
        vtx_masks=[atlas.get_mask(get_rois('MMP-vis3-R'))[0]],
        map_mask=None, zscore0=None, zscore1='split', n_component=20, random_state=7
    )

    # 在成人数据上，对右脑HCP-MMP1_visual-cortex3做zscore
    # 分别对thickness做空间PCA
    atlas = Atlas('HCP-MMP')
    decompose(
        fpaths=[s1200_1096_thickness], cat_shape=(1, 1),
        method='PCA', axis=0,
        csv_files=[pjoin(work_dir, 'HCPY-T_MMP-vis3-R_zscore1_PCA-subj.csv')],
        cii_files=[pjoin(work_dir, 'HCPY-T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')],
        pkl_file=pjoin(work_dir, 'HCPY-T_MMP-vis3-R_zscore1_PCA-subj.pkl'),
        vtx_masks=[atlas.get_mask(get_rois('MMP-vis3-R'))[0]],
        map_mask=None, zscore0=None, zscore1='split', n_component=20, random_state=7
    )

    # >>>在HCPD数据上，去除5~7岁，做tPCA对MMP-vis3-R的顶点降维（分模态）
    # info_df = pd.read_csv(pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo.csv'))
    # ages = np.array(info_df['age in years'])
    # map_mask = np.ones_like(ages, bool)
    # for age in [5, 6, 7]:
    #     idx_vec = ages == age
    #     print(np.sum(idx_vec))
    #     map_mask[idx_vec] = False
    # atlas = Atlas('HCP-MMP')
    # decompose(
    #     fpaths=[pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')],
    #     cat_shape=(1, 1), method='PCA', axis=1,
    #     csv_files=[pjoin(work_dir, 'HCPD-T_MMP-vis3-R_zscore0_PCA-vtx.csv')],
    #     cii_files=[pjoin(work_dir, 'HCPD-T_MMP-vis3-R_zscore0_PCA-vtx.dscalar.nii')],
    #     pkl_file=pjoin(work_dir, 'HCPD-T_MMP-vis3-R_zscore0_PCA-vtx.pkl'),
    #     vtx_masks=[atlas.get_mask(get_rois('MMP-vis3-R'))[0]], zscore0='split',
    #     map_mask=map_mask, n_component=20, random_state=7
    # )
    # 在HCPD数据上，去除5~7岁，做tPCA对MMP-vis3-R的顶点降维（分模态）<<<

    # >>>在HCPD数据上，去除5~7岁，做tPCA对MMP-vis3-R的ROI降维（分模态）
    # info_df = pd.read_csv(pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo.csv'))
    # ages = np.array(info_df['age in years'])
    # row_mask = np.ones_like(ages, bool)
    # for age in [5, 6, 7]:
    #     idx_vec = ages == age
    #     print(np.sum(idx_vec))
    #     row_mask[idx_vec] = False
    # pca_csv(
    #     fpath=pjoin(anal_dir, 'ROI_scalar/HCPD-thickness_HCP-MMP.csv'),
    #     out_csv1=pjoin(work_dir, 'HCPD-T_MMP-vis3-R_zscore0_PCA-ROI_Y.csv'),
    #     out_csv2=pjoin(work_dir, 'HCPD-T_MMP-vis3-R_zscore0_PCA-ROI_W.csv'),
    #     out_pkl=pjoin(work_dir, 'HCPD-T_MMP-vis3-R_zscore0_PCA-ROI.pkl'),
    #     columns=get_rois('MMP-vis3-R'), row_mask=row_mask, zscore0=True,
    #     n_component=20, random_state=7
    # )
    # csv2cifti(
    #     src_file=pjoin(work_dir, 'HCPD-T_MMP-vis3-R_zscore0_PCA-ROI_W.csv'),
    #     rois='all', atlas_name='HCP-MMP',
    #     out_file=pjoin(work_dir, 'HCPD-T_MMP-vis3-R_zscore0_PCA-ROI_W.dscalar.nii')
    # )
    # 在HCPD数据上，去除5~7岁，做tPCA对MMP-vis3-R的ROI降维（分模态）<<<
