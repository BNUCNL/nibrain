from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/structure')


def calc_morphology_individual(hemi='lh', morph='thickness'):
    import nibabel as nib
    import numpy as np
    import pickle as pkl
    from commontool.io.io import CiftiReader
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label

    rois_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                      f'refined_with_Kevin/rois_v3_{hemi}.nii.gz')
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    morph2meas = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
    }
    meas_file = morph2meas[morph]
    trg_file = pjoin(work_dir, f'individual_{morph}_{hemi}.pkl')

    rois = nib.load(rois_file).get_data().squeeze().T
    n_roi = len(roi2label)
    subj_ids = open(subj_file).read().splitlines()
    n_subj = len(subj_ids)
    meas_reader = CiftiReader(meas_file)
    meas_ids = [name.split('_')[0] for name in meas_reader.map_names()]
    meas_indices = [meas_ids.index(i) for i in subj_ids]
    meas = meas_reader.get_data(hemi2stru[hemi], True)[meas_indices]

    roi_meas = {'shape': 'n_roi x n_subj', 'roi': list(roi2label.keys()),
                'meas': np.ones((n_roi, n_subj)) * np.nan}
    for roi_idx, roi in enumerate(roi_meas['roi']):
        lbl_idx_arr = rois == roi2label[roi]
        for subj_idx in range(n_subj):
            lbl_idx_vec = lbl_idx_arr[subj_idx]
            if np.any(lbl_idx_vec):
                roi_meas['meas'][roi_idx, subj_idx] = np.mean(
                    meas[subj_idx][lbl_idx_vec])
    pkl.dump(roi_meas, open(trg_file, 'wb'))


def pre_ANOVA(morph='thickness'):
    """
    准备好二因素被试间设计方差分析需要的数据。
    半球x脑区
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'individual_{}_{}.pkl')
    trg_file = pjoin(work_dir, f'individual_{morph}_preANOVA.csv')

    out_dict = {'hemi': [], 'roi': [], 'meas': []}
    for hemi in hemis:
        data = pkl.load(open(src_file.format(morph, hemi), 'rb'))
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
    # calc_morphology_individual(hemi='lh', morph='thickness')
    # calc_morphology_individual(hemi='lh', morph='myelin')
    # calc_morphology_individual(hemi='rh', morph='thickness')
    # calc_morphology_individual(hemi='rh', morph='myelin')
    pre_ANOVA(morph='thickness')
    pre_ANOVA(morph='myelin')
