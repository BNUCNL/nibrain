from os.path import join as pjoin

proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'analysis/structure')


def calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='LR',
            zscore_flag=False):
    """
    Calculate thickness or myelination
    """
    import time
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from scipy.stats import zscore
    from cxy_visual_dev.lib.predefine import Atlas, L_offset_32k, L_count_32k
    from cxy_visual_dev.lib.predefine import R_offset_32k, R_count_32k, LR_count_32k

    # inputs
    info_file = f'/nfs/e1/{proj_name}/{proj_name}_SubjInfo.csv'
    proj2par = {
        'HCPD': '/nfs/e1/HCPD/fmriresults01',
        'HCPA': '/nfs/e1/HCPA/fmriresults01'}
    proj_par = proj2par[proj_name]

    meas2file = {
        'myelin': pjoin(proj_par,
                        '{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
                        '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'),
        'thickness': pjoin(proj_par,
                           '{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
                           '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii')}
    meas_file = meas2file[meas_name]

    # outputs
    if zscore_flag:
        out_file = pjoin(work_dir, f'{proj_name}_{meas_name}_{atlas_name}_zscore.csv')
    else:
        out_file = pjoin(work_dir, f'{proj_name}_{meas_name}_{atlas_name}.csv')

    # prepare
    atlas = Atlas(atlas_name)
    df = pd.read_csv(info_file)
    n_subj = df.shape[0]
    out_df = pd.DataFrame()
    assert atlas.maps.shape == (1, LR_count_32k)

    # calculate
    for i, idx in enumerate(df.index, 1):
        time1 = time.time()
        sid = df.loc[idx, 'subID']
        meas_map = nib.load(meas_file.format(sid=sid)).get_fdata()
        if zscore_flag:
            meas_map_L = meas_map[:, L_offset_32k:(L_offset_32k+L_count_32k)]
            meas_map_R = meas_map[:, R_offset_32k:(R_offset_32k+R_count_32k)]
            meas_map_L = zscore(meas_map_L, 1)
            meas_map_R = zscore(meas_map_R, 1)
            meas_map[:, L_offset_32k:(L_offset_32k+L_count_32k)] = meas_map_L
            meas_map[:, R_offset_32k:(R_offset_32k+R_count_32k)] = meas_map_R
        for roi, lbl in atlas.roi2label.items():
            meas = np.mean(meas_map[atlas.maps == lbl])
            out_df.loc[idx, roi] = meas
        print(f'Finished {i}/{n_subj}: cost {time.time() - time1}')

    # save
    out_df.to_csv(out_file, index=False)


if __name__ == '__main__':

    # calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='Cole_visual_ROI')
    # calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='Cole_visual_ROI')
    calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='HCP_MMP1')
    calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='HCP_MMP1')
    # calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='LR')
    # calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='LR')
    # calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='Cole_visual_LR')
    # calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='Cole_visual_LR')
    # calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='Cole_visual_ROI', zscore_flag=True)
    # calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='Cole_visual_ROI', zscore_flag=True)
    calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='HCP_MMP1', zscore_flag=True)
    calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='HCP_MMP1', zscore_flag=True)
    # HCP_MMP1 atlas 包含 Cole_visual_ROI
