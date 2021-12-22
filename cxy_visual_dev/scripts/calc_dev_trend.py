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
    age_uniq = np.unique(ages)

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


if __name__ == '__main__':
    calc_mann_kendall_csv(
        src_file=pjoin(anal_dir, 'ROI_scalar/HCPD-myelin_HCP-MMP.csv'),
        info_file=pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo.csv'),
        out_file=pjoin(work_dir, 'HCPD-myelin_HCP-MMP_kendall.csv')
    )
    kendall2cifti(
        src_file=pjoin(work_dir, 'HCPD-myelin_HCP-MMP_kendall.csv'),
        rois=get_rois('MMP-vis3-L')+get_rois('MMP-vis3-R'),
        atlas_name='HCP-MMP',
        out_file=pjoin(work_dir, 'HCPD-myelin_MMP-vis3-ROI_kendall.dscalar.nii')
    )
    calc_mann_kendall_csv(
        src_file=pjoin(anal_dir, 'ROI_scalar/HCPD-thickness_HCP-MMP.csv'),
        info_file=pjoin(proj_dir, 'data/HCP/HCPD_SubjInfo.csv'),
        out_file=pjoin(work_dir, 'HCPD-thickness_HCP-MMP_kendall.csv')
    )
    kendall2cifti(
        src_file=pjoin(work_dir, 'HCPD-thickness_HCP-MMP_kendall.csv'),
        rois=get_rois('MMP-vis3-L')+get_rois('MMP-vis3-R'),
        atlas_name='HCP-MMP',
        out_file=pjoin(work_dir, 'HCPD-thickness_MMP-vis3-ROI_kendall.dscalar.nii')
    )
