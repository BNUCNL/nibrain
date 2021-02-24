from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/grouping/structure')


def calc_meas_individual(gid=1, hemi='lh', morph='thickness'):
    """
    Calculate morphology using individual ROIs.
    """
    import nibabel as nib
    import numpy as np
    import pickle as pkl
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label
    from commontool.io.io import CiftiReader

    morph2file = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
    }
    meas_file = morph2file[morph]
    gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     f'grouping/group_id_{hemi}.npy')
    roi_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     f'rois_v3_{hemi}.nii.gz')
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    trg_file = pjoin(work_dir, f'individual_G{gid}_{morph}_{hemi}.pkl')

    gid_idx_vec = np.load(gid_file) == gid
    subj_ids = open(subj_file).read().splitlines()
    subj_ids = [subj_ids[i] for i in np.where(gid_idx_vec)[0]]
    n_subj = len(subj_ids)
    roi_maps = nib.load(roi_file).get_data().squeeze().T[gid_idx_vec]
    meas_reader = CiftiReader(meas_file)
    meas_ids = [name.split('_')[0] for name in meas_reader.map_names()]
    meas_indices = [meas_ids.index(i) for i in subj_ids]
    meas = meas_reader.get_data(hemi2stru[hemi], True)[meas_indices]

    out_dict = {'shape': 'n_roi x n_subj',
                'roi': list(roi2label.keys()),
                'subject': subj_ids,
                'meas': np.ones((len(roi2label), n_subj)) * np.nan}
    for roi_idx, roi in enumerate(out_dict['roi']):
        lbl_idx_arr = roi_maps == roi2label[roi]
        for subj_idx in range(n_subj):
            lbl_idx_vec = lbl_idx_arr[subj_idx]
            if np.any(lbl_idx_vec):
                out_dict['meas'][roi_idx, subj_idx] = np.mean(
                    meas[subj_idx][lbl_idx_vec])
    pkl.dump(out_dict, open(trg_file, 'wb'))


def pre_ANOVA(gid=1, morph='thickness'):
    """
    准备好二因素被试间设计方差分析需要的数据。
    半球x脑区
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'individual_G{}_{}_{}.pkl')
    trg_file = pjoin(work_dir, f'individual_G{gid}_{morph}_preANOVA.csv')

    out_dict = {'hemi': [], 'roi': [], 'meas': []}
    for hemi in hemis:
        data = pkl.load(open(src_file.format(gid, morph, hemi), 'rb'))
        for roi in rois:
            roi_idx = data['roi'].index(roi)
            meas_vec = data['meas'][roi_idx]
            meas_vec = meas_vec[~np.isnan(meas_vec)]
            n_valid = len(meas_vec)
            out_dict['hemi'].extend([hemi] * n_valid)
            out_dict['roi'].extend([roi.split('-')[0]] * n_valid)
            out_dict['meas'].extend(meas_vec)
            print(f'{hemi}_{roi}:', n_valid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


if __name__ == '__main__':
    # calc_meas_individual(gid=1, hemi='lh', morph='thickness')
    # calc_meas_individual(gid=1, hemi='lh', morph='myelin')
    # calc_meas_individual(gid=1, hemi='rh', morph='thickness')
    # calc_meas_individual(gid=1, hemi='rh', morph='myelin')
    # calc_meas_individual(gid=2, hemi='lh', morph='thickness')
    # calc_meas_individual(gid=2, hemi='lh', morph='myelin')
    # calc_meas_individual(gid=2, hemi='rh', morph='thickness')
    # calc_meas_individual(gid=2, hemi='rh', morph='myelin')
    pre_ANOVA(gid=1, morph='thickness')
    pre_ANOVA(gid=1, morph='myelin')
    pre_ANOVA(gid=2, morph='thickness')
    pre_ANOVA(gid=2, morph='myelin')
