from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/structure')


def calc_meas_individual(hemi='lh', meas='thickness'):
    import nibabel as nib
    import numpy as np
    import pickle as pkl
    from magicbox.io.io import CiftiReader
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label

    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    rois_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                      f'refined_with_Kevin/rois_v3_{hemi}.nii.gz')
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    meas_id_file = pjoin(proj_dir, 'data/HCP/subject_id_1096')
    meas2file = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
        'va': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
              'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
              'S1200.All.midthickness_MSMAll_va.32k_fs_LR.dscalar.nii'
    }
    meas_file = meas2file[meas]
    trg_file = pjoin(work_dir, f'rois_v3_{hemi}_{meas}.pkl')

    roi_maps = nib.load(rois_file).get_fdata().squeeze().T
    n_roi = len(rois)
    subj_ids = open(subj_file).read().splitlines()
    n_subj = len(subj_ids)
    meas_reader = CiftiReader(meas_file)
    meas_ids = open(meas_id_file).read().splitlines()
    meas_indices = [meas_ids.index(i) for i in subj_ids]
    meas_maps = meas_reader.get_data(hemi2stru[hemi], True)[meas_indices]

    roi_meas = {'shape': 'n_roi x n_subj', 'roi': rois,
                'meas': np.ones((n_roi, n_subj)) * np.nan}
    for roi_idx, roi in enumerate(roi_meas['roi']):
        lbl_idx_arr = roi_maps == roi2label[roi]
        for subj_idx in range(n_subj):
            lbl_idx_vec = lbl_idx_arr[subj_idx]
            meas_map = meas_maps[subj_idx]
            if np.any(lbl_idx_vec):
                if meas == 'va':
                    tmp = np.sum(meas_map[lbl_idx_vec])
                else:
                    tmp = np.mean(meas_map[lbl_idx_vec])
                roi_meas['meas'][roi_idx, subj_idx] = tmp
    pkl.dump(roi_meas, open(trg_file, 'wb'))


def calc_meas_group(hemi='lh', meas='thickness'):
    import nibabel as nib
    import numpy as np
    import pickle as pkl
    from magicbox.io.io import CiftiReader
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label

    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    rois_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/'
                      f'refined_with_Kevin/MPM_v3_{hemi}_0.25.nii.gz')
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    meas_id_file = pjoin(proj_dir, 'data/HCP/subject_id_1096')
    meas2file = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
        'va': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
              'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
              'S1200.All.midthickness_MSMAll_va.32k_fs_LR.dscalar.nii'
    }
    meas_file = meas2file[meas]
    trg_file = pjoin(work_dir, f'MPM_v3_{hemi}_0.25_{meas}_new.pkl')

    roi_map = nib.load(rois_file).get_fdata().squeeze()
    n_roi = len(rois)
    subj_ids = open(subj_file).read().splitlines()
    n_subj = len(subj_ids)
    meas_reader = CiftiReader(meas_file)
    meas_ids = open(meas_id_file).read().splitlines()
    meas_indices = [meas_ids.index(i) for i in subj_ids]
    meas_maps = meas_reader.get_data(hemi2stru[hemi], True)[meas_indices]

    roi_meas = {'shape': 'n_roi x n_subj', 'roi': rois,
                'meas': np.ones((n_roi, n_subj)) * np.nan}
    for roi_idx, roi in enumerate(roi_meas['roi']):
        lbl_idx_vec = roi_map == roi2label[roi]
        for subj_idx in range(n_subj):
            meas_map = meas_maps[subj_idx]
            if meas == 'va':
                tmp = np.sum(meas_map[lbl_idx_vec])
            else:
                tmp = np.mean(meas_map[lbl_idx_vec])
            roi_meas['meas'][roi_idx, subj_idx] = tmp
    pkl.dump(roi_meas, open(trg_file, 'wb'))


def pre_ANOVA_individual(meas='thickness'):
    """
    准备好二因素被试间设计方差分析需要的数据。
    半球x脑区
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'rois_v3_{}_{}.pkl')
    trg_file = pjoin(work_dir, f'rois_v3_{meas}_preANOVA.csv')

    out_dict = {'hemi': [], 'roi': [], 'meas': []}
    for hemi in hemis:
        data = pkl.load(open(src_file.format(hemi, meas), 'rb'))
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


def pre_ANOVA_rm_individual(meas='thickness'):
    """
    Preparation for two-way repeated-measures ANOVA
    半球x脑区
    Only the subjects who have all four ROIs will be used.
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'rois_v3_{}_{}.pkl')
    roi_idx_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                   'rois_v3_idx_vec.csv')

    # outputs
    trg_file = pjoin(work_dir, f'rois_v3_{meas}_preANOVA-rm.csv')

    # load data
    roi_idx_df = pd.read_csv(roi_idx_file)
    roi_cols = [f'{i}_{j}' for i in hemis for j in rois]
    roi_idx_vec = np.all(roi_idx_df.loc[:, roi_cols], 1)

    out_dict = {}
    nan_idx_vec = np.zeros_like(roi_idx_vec, bool)
    for hemi in hemis:
        data = pkl.load(open(src_file.format(hemi, meas), 'rb'))
        for roi in rois:
            roi_idx = data['roi'].index(roi)
            meas_vec = data['meas'][roi_idx]
            out_dict[f"{hemi}_{roi.split('-')[0]}"] = meas_vec[roi_idx_vec]
            nan_idx_vec = np.logical_or(nan_idx_vec, np.isnan(meas_vec))
    assert np.all(roi_idx_vec == ~nan_idx_vec)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def plot_bar(meas='thickness'):
    import numpy as np
    import pickle as pkl
    from scipy.stats import sem
    from nibrain.util.plotfig import auto_bar_width
    from matplotlib import pyplot as plt

    src_file = pjoin(work_dir, 'rois_v3_{}_{}.pkl')
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    roi2color = {'pFus-face': 'limegreen', 'mFus-face': 'cornflowerblue'}
    meas2ylabel = {'thickness': 'thickness',
                   'myelin': 'myelination',
                   'activ': 'face selectivity',
                   'va': 'region size'}
    meas2ylim = {'thickness': 2.7,
                 'myelin': 1.3,
                 'activ': 2,
                 'va': 200}

    hemi2meas = {
        'lh': pkl.load(open(src_file.format('lh', meas), 'rb')),
        'rh': pkl.load(open(src_file.format('rh', meas), 'rb'))}
    n_roi = len(rois)
    n_hemi = len(hemis)
    x = np.arange(n_hemi)
    width = auto_bar_width(x, n_roi)
    offset = -(n_roi - 1) / 2
    _, ax = plt.subplots()
    for roi in rois:
        y = np.zeros(n_hemi)
        y_err = np.zeros(n_hemi)
        for hemi_idx, hemi in enumerate(hemis):
            roi_idx = hemi2meas[hemi]['roi'].index(roi)
            meas_map = hemi2meas[hemi]['meas'][roi_idx]
            meas_map = meas_map[~np.isnan(meas_map)]
            print(f'{hemi}_{roi}:', len(meas_map))
            y[hemi_idx] = np.mean(meas_map)
            y_err[hemi_idx] = sem(meas_map)
        ax.bar(x+width*offset, y, width, yerr=y_err,
               label=roi.split('-')[0], color=roi2color[roi])
        offset += 1
    ax.set_xticks(x)
    ax.set_xticklabels(hemis)
    ax.set_ylabel(meas2ylabel[meas])
    ax.set_ylim(meas2ylim[meas])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # calc_meas_individual(hemi='lh', meas='thickness')
    # calc_meas_individual(hemi='lh', meas='myelin')
    # calc_meas_individual(hemi='rh', meas='thickness')
    # calc_meas_individual(hemi='rh', meas='myelin')
    # calc_meas_individual(hemi='lh', meas='va')
    # calc_meas_individual(hemi='rh', meas='va')
    # pre_ANOVA_individual(meas='thickness')
    # pre_ANOVA_individual(meas='myelin')
    # pre_ANOVA_rm_individual(meas='thickness')
    # pre_ANOVA_rm_individual(meas='myelin')
    # pre_ANOVA_rm_individual(meas='va')
    # plot_bar(meas='va')
    calc_meas_group(hemi='lh', meas='thickness')
    calc_meas_group(hemi='lh', meas='myelin')
    calc_meas_group(hemi='rh', meas='thickness')
    calc_meas_group(hemi='rh', meas='myelin')
