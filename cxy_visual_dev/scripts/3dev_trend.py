import os
import numpy as np
import pandas as pd
from os.path import join as pjoin

proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'analysis/dev_trend')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def mann_kendall(data_file, info_file, rois, out_file, atlas_name=None):
    import pymannkendall as mk
    from magicbox.io.io import CiftiReader, save2cifti
    from cxy_visual_dev.lib.predefine import Atlas, LR_count_32k
    from cxy_visual_dev.lib.predefine import mmp_file

    if out_file.endswith('.dscalar.nii'):
        assert atlas_name is not None
    elif out_file.endswith('.csv'):
        pass
    else:
        raise ValueError('only support: .dscalar.nii and .csv')

    # load
    df = pd.read_csv(data_file)
    info_df = pd.read_csv(info_file)
    ages = np.array(info_df['age in years'])
    age_uniq = np.unique(ages)

    # calculate
    out_df = pd.DataFrame(index=rois, columns=('tau', 'p'))
    for roi in rois:
        meas_vec = np.array(df[roi])
        y = np.zeros_like(age_uniq, dtype=np.float64)
        for age_idx, age in enumerate(age_uniq):
            y[age_idx] = np.mean(meas_vec[ages == age])
        mk_test = mk.original_test(y, 0.05)
        out_df.loc[roi, 'tau'] = mk_test.Tau
        out_df.loc[roi, 'p'] = mk_test.p

    # save
    if out_file.endswith('.dscalar.nii'):
        atlas = Atlas(atlas_name)
        assert atlas.maps.shape == (1, LR_count_32k)
        out_data = np.ones((2, LR_count_32k)) * np.nan
        for roi in out_df.index:
            mask = atlas.maps[0] == atlas.roi2label[roi]
            out_data[0, mask] = out_df.loc[roi, 'tau']
            out_data[1, mask] = -np.log10(out_df.loc[roi, 'p'])
        save2cifti(out_file, out_data, CiftiReader(mmp_file).brain_models(),
                   ['tau', '-lg(p)'])
    else:
        out_df.to_csv(out_file)


if __name__ == '__main__':
    # calculate mann kendall for all HCP MMP1.0 ROIs
    from cxy_visual_dev.lib.predefine import mmp_name2label
    rois = list(mmp_name2label.keys())
    mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_myelin_HCP_MMP1.csv'),
        info_file='/nfs/e1/HCPD/HCPD_SubjInfo.csv',
        rois=rois, atlas_name='HCP_MMP1',
        out_file=pjoin(work_dir, 'HCPD_myelin_HCP_MMP1_kendall.dscalar.nii')
    )
    mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_thickness_HCP_MMP1.csv'),
        info_file='/nfs/e1/HCPD/HCPD_SubjInfo.csv',
        rois=rois, atlas_name='HCP_MMP1',
        out_file=pjoin(work_dir, 'HCPD_thickness_HCP_MMP1_kendall.dscalar.nii')
    )
    mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_myelin_HCP_MMP1_zscore.csv'),
        info_file='/nfs/e1/HCPD/HCPD_SubjInfo.csv',
        rois=rois, atlas_name='HCP_MMP1',
        out_file=pjoin(work_dir, 'HCPD_myelin_HCP_MMP1_zscore_kendall.dscalar.nii')
    )
    mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_thickness_HCP_MMP1_zscore.csv'),
        info_file='/nfs/e1/HCPD/HCPD_SubjInfo.csv',
        rois=rois, atlas_name='HCP_MMP1',
        out_file=pjoin(work_dir, 'HCPD_thickness_HCP_MMP1_zscore_kendall.dscalar.nii')
    )

    # calculate mann kendall for Cole_visual_ROI
    from cxy_visual_dev.lib.predefine import get_parcel2label_by_ColeName
    net_names = ['Primary Visual', 'Secondary Visual',
                 'Posterior Multimodal', 'Ventral Multimodal']
    rois = list(get_parcel2label_by_ColeName(net_names).keys())
    mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_myelin_HCP_MMP1.csv'),
        info_file='/nfs/e1/HCPD/HCPD_SubjInfo.csv',
        rois=rois, atlas_name='HCP_MMP1',
        out_file=pjoin(work_dir, 'HCPD_myelin_Cole_visual_ROI_kendall.dscalar.nii')
    )
    mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_thickness_HCP_MMP1.csv'),
        info_file='/nfs/e1/HCPD/HCPD_SubjInfo.csv',
        rois=rois, atlas_name='HCP_MMP1',
        out_file=pjoin(work_dir, 'HCPD_thickness_Cole_visual_ROI_kendall.dscalar.nii')
    )
    mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_myelin_HCP_MMP1_zscore.csv'),
        info_file='/nfs/e1/HCPD/HCPD_SubjInfo.csv',
        rois=rois, atlas_name='HCP_MMP1',
        out_file=pjoin(work_dir, 'HCPD_myelin_Cole_visual_ROI_zscore_kendall.dscalar.nii')
    )
    mann_kendall(
        data_file=pjoin(proj_dir, 'analysis/structure/HCPD_thickness_HCP_MMP1_zscore.csv'),
        info_file='/nfs/e1/HCPD/HCPD_SubjInfo.csv',
        rois=rois, atlas_name='HCP_MMP1',
        out_file=pjoin(work_dir, 'HCPD_thickness_Cole_visual_ROI_zscore_kendall.dscalar.nii')
    )
