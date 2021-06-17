import os
from os.path import join as pjoin

from matplotlib.pyplot import xticks

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                           'retest/tfMRI')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_meas_test(roi_type):
    """
    用45个被试在test或retest中定出的个体FFA去提取各自在test的face selectivity。

    Args:
        roi_type (str): ROI-test or ROI-retest
    """
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from magicbox.io.io import CiftiReader
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    subj_file_45 = pjoin(proj_dir, 'data/HCP/wm/analysis_s2/'
                                   'retest/subject_id')
    subj_file_1080 = pjoin(proj_dir, 'analysis/s2/subject_id')
    meas_file = pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    roi_type2file = {
        'ROI-test': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                    'rois_v3_{hemi}.nii.gz'),
        'ROI-retest': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                      'retest/rois_{hemi}_v2.nii.gz')
    }
    roi_file = roi_type2file[roi_type]

    # outputs
    out_file = pjoin(work_dir, f'activ-test_{roi_type}.csv')

    # prepare
    subj_ids_45 = open(subj_file_45).read().splitlines()
    subj_ids_1080 = open(subj_file_1080).read().splitlines()
    retest_idx_in_1080 = [subj_ids_1080.index(i) for i in subj_ids_45]
    n_subj = len(subj_ids_45)
    out_df = pd.DataFrame(index=range(n_subj))

    # calculate
    meas_reader = CiftiReader(meas_file)
    for hemi in hemis:
        meas_maps = meas_reader.get_data(
            hemi2stru[hemi], True)[retest_idx_in_1080]
        roi_maps = nib.load(roi_file.format(hemi=hemi)).get_fdata().squeeze().T
        if roi_type == 'ROI-test':
            roi_maps = roi_maps[retest_idx_in_1080]
        for roi in rois:
            col = f"{hemi}_{roi.split('-')[0]}"
            for idx in range(n_subj):
                roi_idx_map = roi_maps[idx] == roi2label[roi]
                if np.any(roi_idx_map):
                    out_df.loc[idx, col] = np.mean(meas_maps[idx][roi_idx_map])

    # save
    out_df.to_csv(out_file, index=False)


def calc_meas_retest(roi_type):
    """
    用45个被试在test或retest中定出的个体FFA去提取各自在retest的face selectivity。

    Args:
        roi_type (str): ROI-test or ROI-retest
    """
    import time
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from magicbox.io.io import CiftiReader
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label

    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    subj_file_45 = pjoin(proj_dir, 'data/HCP/wm/analysis_s2/'
                                   'retest/subject_id')
    subj_file_1080 = pjoin(proj_dir, 'analysis/s2/subject_id')
    meas_file = '/nfs/m1/hcp/retest/{0}/MNINonLinear/Results/tfMRI_WM/'\
                'tfMRI_WM_hp200_s2_level2_MSMAll.feat/GrayordinatesStats/'\
                'cope20.feat/zstat1.dtseries.nii'
    roi_type2file = {
        'ROI-test': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                    'rois_v3_{hemi}.nii.gz'),
        'ROI-retest': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                      'retest/rois_{hemi}_v2.nii.gz')
    }
    roi_file = roi_type2file[roi_type]

    # outputs
    out_file = pjoin(work_dir, f'activ-retest_{roi_type}.csv')

    # prepare
    subj_ids_45 = open(subj_file_45).read().splitlines()
    subj_ids_1080 = open(subj_file_1080).read().splitlines()
    retest_idx_in_1080 = [subj_ids_1080.index(i) for i in subj_ids_45]
    n_subj = len(subj_ids_45)
    out_df = pd.DataFrame(index=range(n_subj))

    # prepare atlas
    hemi2atlas = {}
    for hemi in hemis:
        atlas = nib.load(roi_file.format(hemi=hemi)).get_fdata().squeeze().T
        if roi_type == 'ROI-test':
            hemi2atlas[hemi] = atlas[retest_idx_in_1080]
        else:
            hemi2atlas[hemi] = atlas

    # calculate
    for idx, subj_id in enumerate(subj_ids_45):
        time1 = time.time()
        meas_reader = CiftiReader(meas_file.format(subj_id))
        for hemi in hemis:
            meas_map = meas_reader.get_data(hemi2stru[hemi], True)[0]
            roi_map = hemi2atlas[hemi][idx]
            for roi in rois:
                col = f"{hemi}_{roi.split('-')[0]}"
                roi_idx_map = roi_map == roi2label[roi]
                if np.any(roi_idx_map):
                    out_df.loc[idx, col] = np.mean(meas_map[roi_idx_map])
        print(f'Finished {idx+1}/{n_subj}: cost {time.time()-time1} seconds')

    # save
    out_df.to_csv(out_file, index=False)


def plot_bar():
    import numpy as np
    import pandas as pd
    from scipy.stats.stats import sem
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width

    # inputs
    gids = (1, 2)
    hemis = ('lh', 'rh')
    rois = ('pFus', 'mFus')
    sessions = ('test', 'retest')
    subj_file_45 = pjoin(proj_dir, 'data/HCP/wm/analysis_s2/'
                                   'retest/subject_id')
    subj_file_1080 = pjoin(proj_dir, 'analysis/s2/subject_id')
    ses2gid_file = {
        'test': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                'grouping/group_id_{}.npy'),
        'retest': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                                  'retest/grouping/group_id_{}.npy')
    }
    data_file = pjoin(work_dir, 'activ-{ses1}_ROI-{ses2}.csv')
    gid2name = {1: 'two-C', 2: 'two-S'}

    # prepare
    n_gid = len(gids)
    n_ses = len(sessions)
    subj_ids_45 = open(subj_file_45).read().splitlines()
    subj_ids_1080 = open(subj_file_1080).read().splitlines()
    retest_idx_in_1080 = [subj_ids_1080.index(i) for i in subj_ids_45]
    ses_hemi2gid_vec = {}
    for ses in sessions:
        for hemi in hemis:
            gid_file = ses2gid_file[ses].format(hemi)
            gid_vec = np.load(gid_file)
            if ses == 'test':
                gid_vec = gid_vec[retest_idx_in_1080]
            ses_hemi2gid_vec[f'{ses}_{hemi}'] = gid_vec

    # plot
    x = np.arange(len(hemis) * len(rois))
    width = auto_bar_width(x, n_gid)
    _, axes = plt.subplots(n_ses, n_ses * n_ses)
    col_idx = -1
    for ses1 in sessions:
        for ses2 in sessions:
            col_idx += 1
            fpath = data_file.format(ses1=ses1, ses2=ses2)
            data = pd.read_csv(fpath)
            for ses3_idx, ses3 in enumerate(sessions):
                ax = axes[ses3_idx, col_idx]
                offset = -(n_gid - 1) / 2
                for gid in gids:
                    y = []
                    yerr = []
                    xticklabels = []
                    for hemi in hemis:
                        gid_vec = ses_hemi2gid_vec[f'{ses3}_{hemi}']
                        gid_idx_vec = gid_vec == gid
                        print(f'#{ses3}_{hemi}_{gid}:', np.sum(gid_idx_vec))
                        for roi in rois:
                            column = f'{hemi}_{roi}'
                            meas_vec = np.array(data[column])[gid_idx_vec]
                            print(np.sum(np.isnan(meas_vec)))
                            meas_vec = meas_vec[~np.isnan(meas_vec)]
                            y.append(np.mean(meas_vec))
                            yerr.append(sem(meas_vec))
                            xticklabels.append(column)
                    ax.bar(x+width*offset, y, width, yerr=yerr,
                           label=gid2name[gid])
                    offset += 1
                ax.set_xticks(x)
                ax.set_xticklabels(xticklabels)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if col_idx == 0:
                    ax.set_ylabel('face selectivity')
                    if ses3_idx == 0:
                        ax.legend()
                if ses3_idx == 0:
                    fname = os.path.basename(fpath)
                    ax.set_title(fname.rstrip('.csv'))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # calc_meas_test(roi_type='ROI-test')
    # calc_meas_test(roi_type='ROI-retest')
    # calc_meas_retest(roi_type='ROI-test')
    # calc_meas_retest(roi_type='ROI-retest')
    plot_bar()
