import os
import numpy as np
import pandas as pd
import pymannkendall as mk
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import Atlas, LR_count_32k,\
    mmp_map_file, proj_dir, get_rois

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'dev_trend')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_mann_kendall_csv(src_file, info_file, out_file):
    """
    用 kendall tau刻画每个column的发育趋势
    """
    # load
    df = pd.read_csv(src_file)
    info_df = pd.read_csv(info_file)
    ages = np.array(info_df['age in years'])
    age_uniq = np.unique(ages).tolist()
    print(age_uniq)
    for i in [5, 6, 7]:
        age_uniq.remove(i)
    print(age_uniq)

    # calculate
    out_df = pd.DataFrame(index=('tau', 'p'), columns=df.columns)
    for col in out_df.columns:
        meas_vec = np.array(df[col])
        y = np.zeros_like(age_uniq, dtype=np.float64)
        for age_idx, age in enumerate(age_uniq):
            y[age_idx] = np.mean(meas_vec[ages == age])
        mk_test = mk.original_test(y, 0.05)
        out_df.loc['tau', col] = mk_test.Tau
        out_df.loc['p', col] = mk_test.p

    # save
    out_df.to_csv(out_file)


def kendall2cifti(src_file, rois, atlas_name, out_file):
    """
    calc_mann_kendall_csv的后续
    把指定ROI的Tau值和p值存成cifti格式 方便可视化在大脑上
    """
    # prepare
    df = pd.read_csv(src_file, index_col=0)
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    out_data = np.ones((2, LR_count_32k)) * np.nan

    if rois == 'all':
        rois = atlas.roi2label.keys()

    # calculate
    for roi in rois:
        mask = atlas.maps[0] == atlas.roi2label[roi]
        out_data[0, mask] = df.loc['tau', roi]
        out_data[1, mask] = -np.log10(df.loc['p', roi])

    # save
    save2cifti(out_file, out_data, CiftiReader(mmp_map_file).brain_models(),
               ['tau', '-lg(p)'])


def calc_mann_kendall_cii(src_file, vtx_mask, ages, out_file):
    """
    基于age maps，为每个点计算kendall tau
    """
    reader = CiftiReader(src_file)
    ages_all = reader.map_names()
    row_indices = [ages_all.index(i) for i in ages]
    print(row_indices)
    src_maps = reader.get_data()[row_indices, :]
    print(src_maps.shape)
    n_vtx = src_maps.shape[1]

    out_maps = np.ones((2, n_vtx), np.float64) * np.nan
    for idx in range(n_vtx):
        if vtx_mask[idx]:
            y = src_maps[:, idx]
            mk_test = mk.original_test(y, 0.05)
            out_maps[0, idx] = mk_test.Tau
            out_maps[1, idx] = -np.log10(mk_test.p)

    save2cifti(out_file, out_maps, reader.brain_models(),
               ['tau', '-lg(p)'], reader.volume)


def diff_between_age_cii(src_file, age1, age2):
    fname = os.path.basename(src_file)
    fname = fname.split('.')[0]
    out_file = pjoin(work_dir, f'{fname}_{age1}-{age2}.dscalar.nii')
    reader = CiftiReader(src_file)
    src_maps = reader.get_data()
    ages_all = reader.map_names()
    age1_idx = ages_all.index(age1)
    age2_idx = ages_all.index(age2)
    diff_map = src_maps[[age1_idx]] - src_maps[[age2_idx]]
    save2cifti(out_file, diff_map, reader.brain_models(),
               volume=reader.volume)


if __name__ == '__main__':
    # calc_mann_kendall_csv(
    #     src_file=pjoin(anal_dir, 'ROI_scalar/HCPD-myelin_HCP-MMP.csv'),
    #     info_file=pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo.csv'),
    #     out_file=pjoin(work_dir, 'HCPD-myelin_HCP-MMP_kendall.csv')
    # )
    # kendall2cifti(
    #     src_file=pjoin(work_dir, 'HCPD-myelin_HCP-MMP_kendall.csv'),
    #     rois=get_rois('MMP-vis3-L')+get_rois('MMP-vis3-R'),
    #     atlas_name='HCP-MMP',
    #     out_file=pjoin(work_dir, 'HCPD-myelin_MMP-vis3-ROI_kendall.dscalar.nii')
    # )
    # calc_mann_kendall_csv(
    #     src_file=pjoin(anal_dir, 'ROI_scalar/HCPD-thickness_HCP-MMP.csv'),
    #     info_file=pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo.csv'),
    #     out_file=pjoin(work_dir, 'HCPD-thickness_HCP-MMP_kendall.csv')
    # )
    # kendall2cifti(
    #     src_file=pjoin(work_dir, 'HCPD-thickness_HCP-MMP_kendall.csv'),
    #     rois=get_rois('MMP-vis3-L')+get_rois('MMP-vis3-R'),
    #     atlas_name='HCP-MMP',
    #     out_file=pjoin(work_dir, 'HCPD-thickness_MMP-vis3-ROI_kendall.dscalar.nii')
    # )

    # calc_mann_kendall_cii(
    #     src_file=pjoin(anal_dir, 'summary_map/HCPD-myelin_age-map-mean.dscalar.nii'),
    #     vtx_mask=Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-L')+get_rois('MMP-vis3-R'))[0],
    #     ages=[str(i) for i in range(8, 22)],
    #     out_file=pjoin(work_dir, 'HCPD-myelin_MMP-vis3_kendall.dscalar.nii')
    # )
    # calc_mann_kendall_cii(
    #     src_file=pjoin(anal_dir, 'summary_map/HCPD-thickness_age-map-mean.dscalar.nii'),
    #     vtx_mask=Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-L')+get_rois('MMP-vis3-R'))[0],
    #     ages=[str(i) for i in range(8, 22)],
    #     out_file=pjoin(work_dir, 'HCPD-thickness_MMP-vis3_kendall.dscalar.nii')
    # )

    diff_between_age_cii(
        src_file=pjoin(anal_dir, 'summary_map/HCPD-myelin_age-map-mean.dscalar.nii'),
        age1='21', age2='8'
    )
    diff_between_age_cii(
        src_file=pjoin(anal_dir, 'summary_map/HCPD-thickness_age-map-mean.dscalar.nii'),
        age1='21', age2='8'
    )
