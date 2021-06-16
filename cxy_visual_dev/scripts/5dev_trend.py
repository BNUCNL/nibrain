import os
import numpy as np
import pandas as pd
import pymannkendall as mk
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import Atlas, LR_count_32k,\
    mmp_map_file, mmp_name2label, get_parcel2label_by_ColeName,\
    dataset_name2info, proj_dir

work_dir = pjoin(proj_dir, 'analysis/dev_trend')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_mann_kendall(data_file, info_file, out_file):
    """
    用 kendall tau刻画每个column的发育趋势
    """
    # load
    df = pd.read_csv(data_file)
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


def kendall2cifti(data_file, rois, atlas_name, out_file):
    """
    把指定ROI的Tau值和p值存成cifti格式 方便可视化在大脑上
    """
    # prepare
    df = pd.read_csv(data_file, index_col=0)
    atlas = Atlas(atlas_name)
    assert atlas.maps.shape == (1, LR_count_32k)
    out_data = np.ones((2, LR_count_32k)) * np.nan

    # calculate
    for roi in rois:
        mask = atlas.maps[0] == atlas.roi2label[roi]
        out_data[0, mask] = df.loc['tau', roi]
        out_data[1, mask] = -np.log10(df.loc['p', roi])

    # save
    save2cifti(out_file, out_data, CiftiReader(mmp_map_file).brain_models(),
               ['tau', '-lg(p)'])


if __name__ == '__main__':
    # calculate HCP_MMP1's kendall
    calc_mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_myelin_HCP_MMP1.csv'),
        info_file=dataset_name2info['HCPD'],
        out_file=pjoin(work_dir, 'HCPD_myelin_HCP_MMP1_kendall.csv')
    )
    calc_mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_thickness_HCP_MMP1.csv'),
        info_file=dataset_name2info['HCPD'],
        out_file=pjoin(work_dir, 'HCPD_thickness_HCP_MMP1_kendall.csv')
    )
    calc_mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_myelin_4mm_R_cole_visual_ROI-PC1.csv'),
        info_file=dataset_name2info['HCPD'],
        out_file=pjoin(work_dir, 'HCPD_myelin_4mm_R_cole_visual_ROI-PC1_kendall.csv')
    )
    calc_mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_thickness_4mm_R_cole_visual_ROI-PC1.csv'),
        info_file=dataset_name2info['HCPD'],
        out_file=pjoin(work_dir, 'HCPD_thickness_4mm_R_cole_visual_ROI-PC1_kendall.csv')
    )

    # map HCP_MMP1's kendall to cifti
    # rois = list(mmp_name2label.keys())
    # kendall2cifti(
    #     data_file=pjoin(work_dir, 'HCPD_myelin_HCP_MMP1_kendall.csv'),
    #     rois=rois, atlas_name='HCP_MMP1',
    #     out_file=pjoin(work_dir, 'HCPD_myelin_HCP_MMP1_kendall.dscalar.nii')
    # )
    # kendall2cifti(
    #     data_file=pjoin(work_dir, 'HCPD_thickness_HCP_MMP1_kendall.csv'),
    #     rois=rois, atlas_name='HCP_MMP1',
    #     out_file=pjoin(work_dir, 'HCPD_thickness_HCP_MMP1_kendall.dscalar.nii')
    # )

    # map Cole_visual_ROI's kendall to cifti
    # net_names = ['Primary Visual', 'Secondary Visual',
    #              'Posterior Multimodal', 'Ventral Multimodal']
    # rois = list(get_parcel2label_by_ColeName(net_names).keys())
    # kendall2cifti(
    #     data_file=pjoin(work_dir, 'HCPD_myelin_HCP_MMP1_kendall.csv'),
    #     rois=rois, atlas_name='Cole_visual_ROI',
    #     out_file=pjoin(work_dir, 'HCPD_myelin_Cole_visual_ROI_kendall.dscalar.nii')
    # )
    # kendall2cifti(
    #     data_file=pjoin(work_dir, 'HCPD_thickness_HCP_MMP1_kendall.csv'),
    #     rois=rois, atlas_name='Cole_visual_ROI',
    #     out_file=pjoin(work_dir, 'HCPD_thickness_Cole_visual_ROI_kendall.dscalar.nii')
    # )
