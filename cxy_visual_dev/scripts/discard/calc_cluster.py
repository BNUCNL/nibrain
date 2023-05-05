import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.cluster.hierarchy import linkage, fcluster
from community import best_partition, modularity
from magicbox.io.io import CiftiReader, save2cifti, GiftiReader, save2nifti
from magicbox.algorithm.array import summary_across_col_by_mask
from magicbox.algorithm.graph import array2graph
from magicbox.algorithm.triangular_mesh import get_n_ring_neighbor
from cxy_visual_dev.lib.predefine import LR_count_32k, proj_dir, Atlas,\
    get_rois, mmp_map_file, hemi2stru, mmp_name2label

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


def detect_community1():
    """
    对各map的顶点依据各自的值进行Louvain社团检测，并返回检测结果与modularity分数
    """
    hemi = 'rh'
    data_file = pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')
    rois = get_rois('MMP-vis3-R')
    distance_limit = 1  # None即不设限制(时间与内存都不太允许)，正整数N表示N阶近邻
    geo_file = '/nfs/z1/HCP/HCPYA/HCP_S1200_GroupAvg_v1/'\
        'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
    out_name = 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_LV-limit1'
    out_csv_file = pjoin(work_dir, f'{out_name}.csv')
    out_nii_file = pjoin(work_dir, f'{out_name}.nii.gz')

    # prepare mask
    reader = CiftiReader(mmp_map_file)
    mmp_map = reader.get_data(hemi2stru[hemi], True)[0]
    mask = np.zeros_like(mmp_map, bool)
    for roi in rois:
        mask[mmp_map == mmp_name2label[roi]] = True
    n_vtx = len(mask)

    reader1 = CiftiReader(data_file)
    data = reader1.get_data(hemi2stru[hemi], True)[:, mask]
    map_names = reader1.map_names()
    n_map = data.shape[0]

    # prepare edges
    if distance_limit is None:
        edges = 'upper_right_triangle'
    else:
        vtx2vtx = np.arange(n_vtx)
        idx2vtx = vtx2vtx[mask]
        vtx2idx = {}
        for idx, vtx in enumerate(idx2vtx):
            vtx2idx[vtx] = idx
        neighbors_list = get_n_ring_neighbor(
            GiftiReader(geo_file).faces, distance_limit,
            mask=mask.astype(np.uint8))
        edges1 = []
        edges2 = []
        for idx, vtx in enumerate(idx2vtx):
            neighbors = neighbors_list[vtx]
            edges1.extend([idx] * len(neighbors))
            edges2.extend([vtx2idx[i] for i in neighbors])
        edges = list(zip(edges1, edges2))

    # detect community and calculate modularity
    out_nii = np.zeros((n_vtx, 1, 1, n_map), int)
    out_csv = {'name': map_names, 'modularity': np.zeros(n_map, np.float64)}
    for map_idx in range(n_map):
        time1 = time.time()
        arr = data[[map_idx]].T
        graph = array2graph(arr, ('dissimilar', 'euclidean'), True, edges)
        partition = best_partition(graph)
        out_csv['modularity'][map_idx] = \
            modularity(partition, graph, weight='weight')
        for idx, lbl in partition.items():
            out_nii[idx2vtx[idx], 0, 0, map_idx] = lbl + 1
        print(f'Finished {map_idx + 1}/{n_map}: '
              f'cost {time.time() - time1} seconds.')

    # save out
    pd.DataFrame(out_csv).to_csv(out_csv_file, index=False)
    save2nifti(out_nii_file, out_nii)


if __name__ == '__main__':
    # hac()
    # hac_fcluster()
    # csv2cii()
    detect_community1()
