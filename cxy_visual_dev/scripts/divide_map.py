import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from magicbox.io.io import CiftiReader, GiftiReader, save2cifti
from magicbox.graph.triangular_mesh import get_n_ring_neighbor
from magicbox.graph.segmentation import connectivity_grow
from cxy_visual_dev.lib.predefine import proj_dir, get_rois,\
    mmp_label2name, mmp_name2label, mmp_map_file, s1200_midthickness_R,\
    hemi2stru, R_count_32k, R_offset_32k, LR_count_32k,\
    L_count_32k, L_offset_32k

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'divide_map')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def find_roi_neighbors():
    """
    找出MMP-vis3-R所有脑区的近邻脑区
    """
    hemi = 'rh'
    mask_name = 'MMP-vis3-R'

    # get ROIs, label map, and mask
    rois = get_rois(mask_name)
    reader = CiftiReader(mmp_map_file)
    mmp_map = reader.get_data(hemi2stru[hemi], True)[0]
    mask = np.zeros_like(mmp_map, dtype=np.uint8)
    roi2vertices = {}  # prepare vertices
    roi2neighbors = {}  # initial outputs
    for roi in rois:
        vertices = np.where(mmp_map == mmp_name2label[roi])[0]
        mask[vertices] = 1
        roi2vertices[roi] = vertices
        roi2neighbors[roi] = []

    # get full neighbors
    faces = GiftiReader(s1200_midthickness_R).faces
    vtx2neighbors = get_n_ring_neighbor(faces, mask=mask)

    # calculating
    for roi in rois:
        roi_neighbor_labels = set()
        for vtx in roi2vertices[roi]:
            roi_neighbor_labels.update(
                mmp_map[list(vtx2neighbors[vtx])])
        for roi_neighbor_lbl in roi_neighbor_labels:
            if roi_neighbor_lbl == mmp_name2label[roi]:
                continue
            roi2neighbors[roi].append(mmp_label2name[roi_neighbor_lbl])

    return roi2neighbors


def cluster_roi(connectivity='undirected', linkage='ward'):
    """
    （最终结果不行，相关调用代码和结果已删）
    根据stru-C2对MMP-vis3-R所有脑区进行聚类

    Args:
        connectivity (str, optional): Defaults to 'undirected'.
            unidirection: 当近邻脑区的值大于自己时，才视作有连接
            undirected: 只要是近邻脑区就都算是与自己有连接

    References:
        1. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
        2. https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html
    """
    n_clusters = list(range(2, 10))
    roi2neighbors = find_roi_neighbors()
    data_file = pjoin(anal_dir, 'ROI_scalar/ROI_scalar1_MMP-vis3-R.csv')
    out_file = pjoin(work_dir, 'cluster-roi_data-PC2_'
                     f'{connectivity}_{linkage}.dlabel.nii')

    rois = list(roi2neighbors.keys())
    n_roi = len(rois)
    data = pd.read_csv(data_file, index_col=0).loc['mean_stru-C2']
    X = np.expand_dims(np.array([data[roi] for roi in rois]), axis=1)
    connect_mat = np.zeros((n_roi, n_roi))
    for roi_idx1, roi1 in enumerate(rois):
        for roi2 in roi2neighbors[roi1]:
            if connectivity == 'unidirection' and data[roi1] > data[roi2]:
                continue
            roi_idx2 = rois.index(roi2)
            connect_mat[roi_idx1, roi_idx2] = 1

    reader = CiftiReader(mmp_map_file)
    mmp_map = reader.get_data()[0]
    roi2mask = {}
    for roi in rois:
        roi2mask[roi] = mmp_map == mmp_name2label[roi]

    map_names = [f'n_cluster={i}' for i in n_clusters]
    n_map = len(map_names)
    lbl_tabs = []
    out_maps = np.zeros((n_map, mmp_map.shape[0]), dtype=np.uint8)
    cmap = plt.cm.jet
    for c_idx, n_cluster in enumerate(n_clusters):
        time1 = time.time()
        color_indices = np.linspace(0, 1, n_cluster)
        model = AgglomerativeClustering(
            n_clusters=n_cluster, connectivity=connect_mat, linkage=linkage
        ).fit(X)
        lbl_tab = nib.cifti2.Cifti2LabelTable()
        lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
        for lbl in range(1, n_cluster + 1):
            lbl_tab[lbl] = nib.cifti2.Cifti2Label(
                lbl, str(lbl), *cmap(color_indices[lbl - 1]))
        lbl_tabs.append(lbl_tab)
        for roi, lbl in zip(rois, model.labels_ + 1):
            out_maps[c_idx, roi2mask[roi]] = lbl
        print(f'Finished {connectivity}-{linkage}-{c_idx + 1}/{n_map}, '
              f'cost {time.time() - time1} seconds.')

    save2cifti(out_file, out_maps, reader.brain_models(), map_names,
               label_tables=lbl_tabs)


def get_lowest_vertices():
    """
    找出那些比自己的一环近邻都小的顶点。
    目前只用于stru-C2
    """
    hemi = 'rh'
    mask_name = 'MMP-vis3-R'
    hemi2offset_count = {
        'lh': (L_offset_32k, L_count_32k),
        'rh': (R_offset_32k, R_count_32k)}
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'{mask_name}_zscore1_PCA-subj.dscalar.nii')
    out_file = pjoin(work_dir, f'lowest-vtx_{mask_name}.dlabel.nii')

    # prepare atlas information
    reader = CiftiReader(mmp_map_file)
    LR_shape = reader.full_data.shape
    assert (1, LR_count_32k) == LR_shape
    bms = reader.brain_models()
    mmp_map = reader.get_data(hemi2stru[hemi], True)[0]
    _, hemi_shape, idx2vtx = reader.get_data(hemi2stru[hemi], False)
    mask = np.zeros(hemi_shape, dtype=np.uint8)
    for roi in get_rois(mask_name):
        mask[mmp_map == mmp_name2label[roi]] = 1
    mask_vertices = np.nonzero(mask)[0]

    # get vertex neighbors
    faces = GiftiReader(s1200_midthickness_R).faces
    vtx2neighbors = get_n_ring_neighbor(faces, mask=mask)

    # get source data
    src_map = CiftiReader(src_file).get_data(
        hemi2stru[hemi], True)[1]

    # calculating
    out_map = np.zeros(LR_shape, dtype=np.uint8)
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    lbl_tab[1] = nib.cifti2.Cifti2Label(1, 'lowest-vtx', 0, 0, 1, 1)
    hemi_map = np.zeros(hemi_shape, dtype=np.uint8)
    for mask_vtx in mask_vertices:
        neighbor_vertices = list(vtx2neighbors[mask_vtx])
        assert len(neighbor_vertices) != 0
        if np.all(src_map[neighbor_vertices] > src_map[mask_vtx]):
            hemi_map[mask_vtx] = 1
    offset, count = hemi2offset_count[hemi]
    out_map[0, offset:(offset+count)] = hemi_map[idx2vtx]

    # save out
    save2cifti(out_file, out_map, bms, label_tables=[lbl_tab])


def get_lowest_seeds():
    """
    参考get_lowest_vertices得到的局部最小值点，
    选定几个区域作为后续扩张的种子区域。
    目前只用于stru-C2和MMP-vis3-R
    """
    hemi = 'rh'
    mask_name = 'MMP-vis3-R'
    hemi2offset_count = {
        'lh': (L_offset_32k, L_count_32k),
        'rh': (R_offset_32k, R_count_32k)}
    seed_vertices = [23175, 24938, 25131,
                     25402, 1474, 12586,
                     12394, 21501, 22485]
    seed_names = ['early-1', 'early-2', 'early-3', 'early-4',
                  'dorsal-1', 'dorsal-2', 'dorsal-3',
                  'ventral-1', 'ventral-2']
    ex_vertices = [12466, 12423, 12333, 12378, 12286, 12238, 12189]
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'{mask_name}_zscore1_PCA-subj.dscalar.nii')
    out_file = pjoin(work_dir, f'lowest-seed_{mask_name}.dlabel.nii')

    # prepare map information
    reader = CiftiReader(src_file)
    LR_shape = (1, LR_count_32k)
    assert reader.full_data.shape[1] == LR_count_32k
    bms = reader.brain_models()
    src_map = reader.get_data(hemi2stru[hemi], True)[1]
    _, hemi_shape, idx2vtx = reader.get_data(hemi2stru[hemi], False)
    mask = (src_map < -13.6).astype(np.uint8)
    mask[ex_vertices] = 0

    # get vertex neighbors
    faces = GiftiReader(s1200_midthickness_R).faces
    vtx2neighbors = get_n_ring_neighbor(faces, mask=mask)

    # get seed regions
    n_seed = len(seed_vertices)
    seeds_id = [[i] for i in seed_vertices]
    seed_regions = connectivity_grow(seeds_id, vtx2neighbors)

    # save out
    cmap = plt.cm.jet
    color_indices = np.linspace(0, 1, n_seed)
    out_map = np.zeros(LR_shape, dtype=np.uint8)
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    hemi_map = np.zeros(hemi_shape, dtype=np.uint8)
    for seed_idx, seed_region in enumerate(seed_regions):
        seed_key = seed_idx + 1
        hemi_map[list(seed_region)] = seed_key
        lbl_tab[seed_key] = nib.cifti2.Cifti2Label(
            seed_key, seed_names[seed_idx],
            *cmap(color_indices[seed_idx]))
    offset, count = hemi2offset_count[hemi]
    out_map[0, offset:(offset+count)] = hemi_map[idx2vtx]
    save2cifti(out_file, out_map, bms, label_tables=[lbl_tab])


if __name__ == '__main__':
    # get_lowest_vertices()
    get_lowest_seeds()
