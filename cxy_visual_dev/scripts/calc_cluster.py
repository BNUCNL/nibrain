import os
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.cluster.hierarchy import linkage
from matplotlib import pyplot as plt
from magicbox.algorithm.array import summary_across_col_by_mask
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    get_rois

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'cluster')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def hac():
    """
    hierarchical/agglomerative clustering
    """
    atlas = Atlas('HCP-MMP')
    rois_L = get_rois('MMP-vis2-L')
    rois_R = get_rois('MMP-vis2-R')
    rois = [roi[2:] for roi in rois_L]
    assert rois == [roi[2:] for roi in rois_R]
    labels_L = [atlas.roi2label[roi] for roi in rois_L]
    labels_R = [atlas.roi2label[roi] for roi in rois_R]

    fpath = pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii')
    out_file = pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_hac-C1.pkl')

    C1 = nib.load(fpath).get_fdata()[[0]]
    data_L = summary_across_col_by_mask(C1, atlas.maps[0], labels_L, 'mean')
    data_R = summary_across_col_by_mask(C1, atlas.maps[0], labels_R, 'mean')
    data = np.r_[data_L, data_R]
    data = np.mean(data, 0, keepdims=True).T

    Z = linkage(data, 'ward', 'euclidean')
    pkl.dump({'label': rois, 'Z': Z}, open(out_file, 'wb'))


if __name__ == '__main__':
    hac()
