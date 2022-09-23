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

    # thr = -13.6
    # seed_vertices = [23175, 24938, 25131,
    #                  25402, 1474, 12586,
    #                  12394, 21501, 22485]
    # seed_names = ['early-1', 'early-2', 'early-3', 'early-4',
    #               'dorsal-1', 'dorsal-2', 'dorsal-3',
    #               'ventral-1', 'ventral-2']
    # ex_vertices = [12466, 12423, 12333, 12378, 12286, 12238, 12189]
    # out_file = pjoin(work_dir, f'lowest-seed_{mask_name}.dlabel.nii')

    thr = -10
    seed_vertices = [25541, 23175, 24938, 25131, 25666,
                     25841, 1474, 12394, 13797,
                     21501, 22485, 22304, 26554]
    seed_names = ['early-1', 'early-2', 'early-3', 'early-4', 'early-5',
                  'dorsal-1', 'dorsal-2', 'dorsal-3', 'dorsal-4',
                  'ventral-1', 'ventral-2', 'ventral-3', 'ventral-4']
    ex_vertices = []
    out_file = pjoin(work_dir, f'lowest-seed_thr-{thr}_{mask_name}.dlabel.nii')

    hemi2offset_count = {
        'lh': (L_offset_32k, L_count_32k),
        'rh': (R_offset_32k, R_count_32k)}
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'{mask_name}_zscore1_PCA-subj.dscalar.nii')
    # prepare map information
    reader = CiftiReader(src_file)
    LR_shape = (1, LR_count_32k)
    assert reader.full_data.shape[1] == LR_count_32k
    bms = reader.brain_models()
    src_map = reader.get_data(hemi2stru[hemi], True)[1]
    _, hemi_shape, idx2vtx = reader.get_data(hemi2stru[hemi], False)
    mask = (src_map < thr).astype(np.uint8)
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


def expand_seed_combo():
    """
    对每个seed region组合：
    1. 以该组合包含的所有顶点为起点
    2. 遍历各扩张起点，对于每个起点，合并1环近邻中大于它，
        并且不属于该组合的顶点，同时作为下一步扩张的起点。
    3. 重复第2步，直到没有扩张起点为止。
    结果保存在.dlabel.nii文件中，每种组合存为其中一个map，
    属于该组合的顶点标记为1，其它为0

    目前只用于stru-C2
    """
    hemi = 'rh'
    mask_name = 'MMP-vis3-R'
    hemi2offset_count = {
        'lh': (L_offset_32k, L_count_32k),
        'rh': (R_offset_32k, R_count_32k)}
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'{mask_name}_zscore1_PCA-subj.dscalar.nii')

    # seed_key2lbl = {
    #     1: 'early-1', 2: 'early-2', 3: 'early-3', 4: 'early-4',
    #     5: 'dorsal-1', 6: 'dorsal-2', 7: 'dorsal-3',
    #     8: 'ventral-1', 9: 'ventral-2'}
    # seed_combos = [(i,) for i in range(1, 10)] + \
    #     [(2, 3, 4), (1, 2), (1, 2, 3, 4), (5, 6, 7), (6, 7), (8, 9)]
    # seed_file = pjoin(work_dir, f'lowest-seed_{mask_name}.dlabel.nii')
    # out_file = pjoin(work_dir, f'seed-expansion_{mask_name}.dlabel.nii')

    thr = -10
    seed_key2lbl = {
        1: 'early-1', 2: 'early-2', 3: 'early-3', 4: 'early-4', 5: 'early-5',
        6: 'dorsal-1', 7: 'dorsal-2', 8: 'dorsal-3', 9: 'dorsal-4',
        10: 'ventral-1', 11: 'ventral-2', 12: 'ventral-3', 13: 'ventral-4'}
    seed_combos = [(1, 2, 3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13)]
    seed_file = pjoin(work_dir, f'lowest-seed_thr-{thr}_{mask_name}.dlabel.nii')
    out_file = pjoin(work_dir, f'seed-expansion_thr-{thr}_{mask_name}.dlabel.nii')

    # prepare map information
    reader = CiftiReader(seed_file)
    assert reader.full_data.shape == (1, LR_count_32k)
    bms = reader.brain_models()
    lbl_tab = reader.label_tables()[0]
    seed_map = reader.get_data(hemi2stru[hemi], True)[0]
    _, hemi_shape, idx2vtx = reader.get_data(hemi2stru[hemi], False)

    # get vertex neighbors
    mmp_map = CiftiReader(mmp_map_file).get_data(
        hemi2stru[hemi], True)[0]
    mask = np.zeros(hemi_shape, np.uint8)
    for roi in get_rois(mask_name):
        mask[mmp_map == mmp_name2label[roi]] = 1
    faces = GiftiReader(s1200_midthickness_R).faces
    vtx2neighbors = get_n_ring_neighbor(faces, mask=mask)

    # get source data
    src_map = CiftiReader(src_file).get_data(
        hemi2stru[hemi], True)[1]

    # calculating
    n_combo = len(seed_combos)
    out_maps = np.zeros((n_combo, LR_count_32k), np.uint8)
    map_names = []
    lbl_tabs = []
    offset, count = hemi2offset_count[hemi]
    for combo_idx, combo in enumerate(seed_combos):
        hemi_map = np.zeros(hemi_shape, np.uint8)
        base_vertices = []
        for seed_key in combo:
            base_vertices.extend(np.where(seed_map == seed_key)[0])
            assert seed_key2lbl[seed_key] == lbl_tab[seed_key].label
        hemi_map[base_vertices] = 1
        while len(base_vertices) > 0:
            base_vertices_tmp = []
            for base_vtx in base_vertices:
                for neigh_vtx in vtx2neighbors[base_vtx]:
                    if hemi_map[neigh_vtx] == 1:
                        continue
                    if src_map[neigh_vtx] > src_map[base_vtx]:
                        hemi_map[neigh_vtx] = 1
                        base_vertices_tmp.append(neigh_vtx)
            base_vertices = base_vertices_tmp
        out_maps[combo_idx, offset:(offset+count)] = hemi_map[idx2vtx]
        map_names.append(str(combo))
        lbl_tab_new = nib.cifti2.Cifti2LabelTable()
        lbl_tab_new[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
        lbl_tab_new[1] = nib.cifti2.Cifti2Label(1, map_names[-1], 0, 0, 1, 1)
        lbl_tabs.append(lbl_tab_new)

    # save out
    save2cifti(out_file, out_maps, bms, map_names, label_tables=lbl_tabs)


def get_EDLV_seeds(Hemi='R'):
    """
    在EDLV四个通路中寻找stru-C2最小的那部分作为各通路的种子区域
    """
    props = list(range(1, 16))
    edlv_file = pjoin(proj_dir, 'data/HCP/HCP-MMP1_visual-cortex3_EDLV.dlabel.nii')
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    out_file = pjoin(work_dir, f'MMP-vis3-{Hemi}-EDLV-seed_bottom-proportion.dlabel.nii')

    # get EDLV data
    reader = CiftiReader(edlv_file)
    lbl_tab = reader.label_tables()[0]
    bms = reader.brain_models()
    edlv_map = reader.get_data()[0]
    assert len(edlv_map) == LR_count_32k

    # get stru-C2 data
    src_map = CiftiReader(src_file).get_data()[1]
    assert len(src_map) == LR_count_32k

    # calculating
    n_prop = len(props)
    out_maps = np.zeros((n_prop, LR_count_32k), np.uint8)
    lbl_tab_new = nib.cifti2.Cifti2LabelTable()
    lbl_tab_new[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    map_names = [f'{i}%' for i in props]
    for k in lbl_tab.keys():
        lbl = lbl_tab[k].label
        if k == 0 or not lbl.startswith(f'{Hemi}_'):
            continue
        lbl_tab_new[k] = nib.cifti2.Cifti2Label(k, lbl, *lbl_tab[k].rgba)
        mask = edlv_map == k
        sorted_indices = np.argsort(src_map[mask])
        n_vtx = len(sorted_indices)
        for prop_idx, prop in enumerate(props):
            local_seed_map = np.zeros(n_vtx, np.uint8)
            n_bottom_vtx = int(n_vtx * prop / 100)
            indices = sorted_indices[:n_bottom_vtx]
            local_seed_map[indices] = k
            out_maps[prop_idx, mask] = local_seed_map

    # save out
    lbl_tabs = [lbl_tab_new] * n_prop
    save2cifti(out_file, out_maps, bms, map_names, label_tables=lbl_tabs)


if __name__ == '__main__':
    # get_lowest_vertices()
    # get_lowest_seeds()
    # expand_seed_combo()
    get_EDLV_seeds(Hemi='R')
