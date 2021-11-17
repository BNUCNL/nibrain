import os
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.cluster.hierarchy import linkage, fcluster
from magicbox.algorithm.array import summary_across_col_by_mask
from cxy_visual_dev.lib.predefine import LR_count_32k, proj_dir, Atlas,\
    get_rois, mmp_map_file
from magicbox.io.io import CiftiReader, save2cifti

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


def hac_fcluster():
    """
    Form flat clusters from the hierarchical clustering
    defined by the given linkage matrix.
    """
    fpath = pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_hac-C1.pkl')
    out_file = pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_hac-C1.csv')
    data = pkl.load(open(fpath, 'rb'))
    out_dict = {'roi_name': data['label']}
    for i in range(2, 11):
        out_dict[i] = fcluster(data['Z'], i, 'maxclust')
    df = pd.DataFrame(out_dict)
    df.to_csv(out_file, index=False)


def csv2cii():
    """
    把每个聚类数量情况下的ROI聚类标签投到脑图上
    """
    fpath = pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_hac-C1.csv')
    out_file = pjoin(work_dir, 'HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj_hac-C1.dscalar.nii')
    df = pd.read_csv(fpath)

    atlas = Atlas('HCP-MMP')
    reader = CiftiReader(mmp_map_file)

    rois_L = np.array([f'L_{i}' for i in df['roi_name']])
    rois_R = np.array([f'R_{i}' for i in df['roi_name']])
    cols = df.columns.to_list()
    cols.remove('roi_name')
    n_col = len(cols)
    data = np.ones((n_col, LR_count_32k), np.float64) * np.nan
    for col_idx, col in enumerate(cols):
        labels = np.unique(df[col])
        assert len(labels) == int(col)
        for lbl in labels:
            rois_tmp = np.r_[
                rois_L[df[col] == lbl],
                rois_R[df[col] == lbl]
            ]
            mask = atlas.get_mask(rois_tmp)[0]
            data[col_idx, mask] = lbl

    save2cifti(out_file, data, reader.brain_models(), cols)


if __name__ == '__main__':
    # hac()
    # hac_fcluster()
    csv2cii()
