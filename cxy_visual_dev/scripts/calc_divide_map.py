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
from magicbox.graph.segmentation import connectivity_grow, watershed
from cxy_visual_dev.lib.predefine import proj_dir, get_rois,\
    mmp_label2name, mmp_name2label, mmp_map_file, s1200_midthickness_R,\
    hemi2stru, LR_count_32k, hemi2Hemi, s1200_midthickness_L, Hemi2stru

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


def get_extremum_vertices(hemi='rh'):
    """
    找出那些极值点
    比自己的一环近邻都小的顶点（极小值点）标记为1
    比自己的一环近邻都大的顶点（极大值点）标记为2
    目前只用于stru-C2
    """
    Hemi = hemi2Hemi[hemi]
    mask_name = f'MMP-vis3-{Hemi}'
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'{mask_name}_zscore1_PCA-subj.dscalar.nii')
    out_file = pjoin(work_dir, f'extremum-vtx_{mask_name}.dlabel.nii')

    hemi2geo = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    geo_file = hemi2geo[hemi]

    # prepare atlas information
    reader = CiftiReader(mmp_map_file)
    LR_shape = reader.full_data.shape
    assert (1, LR_count_32k) == LR_shape
    bms = reader.brain_models()
    mmp_map = reader.get_data(hemi2stru[hemi], True)[0]
    offset, count, hemi_shape, idx2vtx = reader.get_stru_pos(hemi2stru[hemi])
    mask = np.zeros(hemi_shape, dtype=np.uint8)
    for roi in get_rois(mask_name):
        mask[mmp_map == mmp_name2label[roi]] = 1
    mask_vertices = np.nonzero(mask)[0]

    # get vertex neighbors
    faces = GiftiReader(geo_file).faces
    vtx2neighbors = get_n_ring_neighbor(faces, mask=mask)

    # get source data
    src_map = CiftiReader(src_file).get_data(
        hemi2stru[hemi], True)[1]

    # calculating
    out_map = np.zeros(LR_shape, dtype=np.uint8)
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    lbl_tab[1] = nib.cifti2.Cifti2Label(1, 'minima', 0, 0, 1, 1)
    lbl_tab[2] = nib.cifti2.Cifti2Label(2, 'maxima', 1, 0, 0, 1)
    hemi_map = np.zeros(hemi_shape, dtype=np.uint8)
    for mask_vtx in mask_vertices:
        neighbor_vertices = list(vtx2neighbors[mask_vtx])
        assert len(neighbor_vertices) != 0
        if np.all(src_map[neighbor_vertices] > src_map[mask_vtx]):
            hemi_map[mask_vtx] = 1
        elif np.all(src_map[neighbor_vertices] < src_map[mask_vtx]):
            hemi_map[mask_vtx] = 2
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

    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'{mask_name}_zscore1_PCA-subj.dscalar.nii')
    # prepare map information
    reader = CiftiReader(src_file)
    LR_shape = (1, LR_count_32k)
    assert reader.full_data.shape[1] == LR_count_32k
    bms = reader.brain_models()
    src_map = reader.get_data(hemi2stru[hemi], True)[1]
    offset, count, hemi_shape, idx2vtx = reader.get_stru_pos(hemi2stru[hemi])
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
    offset, count, hemi_shape, idx2vtx = reader.get_stru_pos(hemi2stru[hemi])

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


def expand_EDLV_seeds(hemi='rh'):
    """
    对每种proportion，每个局部的seed region：
    1. 以其所有顶点为起点
    2. 遍历各扩张起点，对于每个起点，合并1环近邻中大于它，
        并且不属于该region的顶点，同时作为下一步扩张的起点。
    3. 重复第2步，直到没有扩张起点为止。
    4. 结果存为一个map，属于region的顶点标记为1，其它为0
    对每个局部的所有proportion计算概率图，存到.dscalar.nii中
    """
    Hemi = hemi2Hemi[hemi]
    mask_name = f'MMP-vis3-{Hemi}'
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'{mask_name}_zscore1_PCA-subj.dscalar.nii')

    local_names = ('early', 'dorsal', 'lateral', 'ventral')
    local_names = [f'{Hemi}_{i}' for i in local_names]
    seed_file = pjoin(work_dir, f'{mask_name}-EDLV-seed_bottom-proportion.dlabel.nii')
    out_file1 = pjoin(work_dir, f'{mask_name}-EDLV-seed-expansion.dlabel.nii')
    out_file2 = pjoin(work_dir, f'{mask_name}-EDLV-seed-expansion_prob.dscalar.nii')

    # prepare map information
    reader = CiftiReader(seed_file)
    assert reader.full_data.shape[1] == LR_count_32k
    bms = reader.brain_models()
    lbl_tab = reader.label_tables()[0]
    local2key = {}
    for k, v in lbl_tab.items():
        local2key[v.label] = k
    seed_maps = reader.get_data(hemi2stru[hemi], True)
    props = reader.map_names()
    offset, count, hemi_shape, idx2vtx = reader.get_stru_pos(hemi2stru[hemi])

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
    n_local = len(local_names)
    n_prop = len(props)
    out_maps1 = np.zeros((n_prop*n_local, LR_count_32k), np.uint8)
    out_maps2 = np.ones((n_local, LR_count_32k)) * np.nan
    map_idx = 0
    lbl_tabs = []
    map_names = []
    for local_idx, local_name in enumerate(local_names):
        prob_map = np.zeros(hemi_shape)
        for prop_idx, prop in enumerate(props):
            hemi_map = np.zeros(hemi_shape, np.uint8)
            base_vertices = np.where(
                seed_maps[prop_idx] == local2key[local_name])[0]
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
            prob_map = prob_map + hemi_map
            out_maps1[map_idx, offset:(offset+count)] = hemi_map[idx2vtx]
            map_name = f'{local_name}-{prop}'
            lbl_tab_new = nib.cifti2.Cifti2LabelTable()
            lbl_tab_new[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
            lbl_tab_new[1] = nib.cifti2.Cifti2Label(1, map_name, 0, 0, 1, 1)
            lbl_tabs.append(lbl_tab_new)
            map_names.append(map_name)
            map_idx += 1
        prob_map = prob_map / n_prop
        out_maps2[local_idx, offset:(offset+count)] = prob_map[idx2vtx]

    # save out
    save2cifti(out_file1, out_maps1, bms, map_names, label_tables=lbl_tabs)
    save2cifti(out_file2, out_maps2, bms, local_names)


def get_observed_seeds(hemi='rh'):
    """
    根据观察大致有四个区域，手动选择大致的中心位置作为种子区域
    先在wb_view上选定点和线，然后加粗（合并1环近邻）
    """
    seed_name2vtx = {
        'early': [24963, 24860, 24808, 24754, 24525, 24433, 24381,
                  24295, 24233, 24168, 24100, 24028, 24030, 24032,
                  24774, 24884, 24989, 24481, 25089, 25137, 25184,
                  25230, 25275, 25319, 25361, 25402],
        'dorsal': [12772, 12701, 12625, 12586, 12505, 12463, 12376,
                   12333, 12381, 12383, 12384, 12431, 12433, 12434,
                   12436, 12437, 12483, 12528, 12530, 12573],
        'lateral': [15090, 15034, 23209, 23264, 23318, 23371, 23423,
                    23474, 23475, 23526, 23527, 23528, 23529, 23481,
                    23432, 23382, 23331, 23279, 23226, 15015, 15124,
                    15230, 15332, 15430, 15523],
        'ventral': [21494, 21496, 21498, 23062, 21501, 23020, 22988,
                    22951, 22931, 22890, 22868, 22821, 22770, 22743,
                    22655, 22591, 22522, 22485, 22447]}

    Hemi = hemi2Hemi[hemi]
    mask_name = f'MMP-vis3-{Hemi}'
    hemi2geo_file = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    out_file = pjoin(work_dir, f'observed-seed-v4_{mask_name}.dlabel.nii')

    # prepare atlas information
    reader = CiftiReader(mmp_map_file)
    LR_shape = reader.full_data.shape
    assert (1, LR_count_32k) == LR_shape
    bms = reader.brain_models()
    mmp_map = reader.get_data(hemi2stru[hemi], True)[0]
    offset, count, hemi_shape, idx2vtx = reader.get_stru_pos(hemi2stru[hemi])
    mask = np.zeros(hemi_shape, dtype=np.uint8)
    for roi in get_rois(mask_name):
        mask[mmp_map == mmp_name2label[roi]] = 1

    # get vertex neighbors
    faces = GiftiReader(hemi2geo_file[hemi]).faces
    vtx2neighbors = get_n_ring_neighbor(faces, mask=mask)

    # save out
    out_map = np.zeros(LR_shape, dtype=np.uint8)
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    hemi_map = np.zeros(hemi_shape, dtype=np.uint8)
    for seed_key, seed_name in enumerate(seed_name2vtx.keys(), 1):
        seed_vertices = seed_name2vtx[seed_name]
        hemi_map[seed_vertices] = seed_key
        for seed_vtx in seed_vertices:
            hemi_map[list(vtx2neighbors[seed_vtx])] = seed_key
        if seed_name == 'lateral':
            color = (0, 0, 0, 1)
        else:
            color = (1, 0, 0, 1)
        lbl_tab[seed_key] = nib.cifti2.Cifti2Label(
            seed_key, f'{Hemi}_{seed_name}', *color)
    out_map[0, offset:(offset+count)] = hemi_map[idx2vtx]
    save2cifti(out_file, out_map, bms, label_tables=[lbl_tab])


def expand_observed_seeds(hemi='rh'):
    """
    对每个局部的seed region：
    1. 以其所有顶点为起点
    2. 遍历各扩张起点，对于每个起点，合并1环近邻中大于它(如果是lateral则是合并小于它的)，
        并且不属于该region的顶点，同时作为下一步扩张的起点。
    3. 重复第2步，直到没有扩张起点为止。
    4. 结果存为一个map，属于region的顶点标记为1，其它为0
    """
    Hemi = hemi2Hemi[hemi]
    mask_name = f'MMP-vis3-{Hemi}'
    hemi2geo_file = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'{mask_name}_zscore1_PCA-subj.dscalar.nii')

    local_names = ('early', 'dorsal', 'lateral', 'ventral')
    local_names = [f'{Hemi}_{i}' for i in local_names]
    seed_file = pjoin(work_dir, f'observed-seed-v2_{mask_name}.dlabel.nii')
    out_file = pjoin(work_dir, f'observed-seed-v2-expansion_{mask_name}.dlabel.nii')

    # prepare map information
    reader = CiftiReader(seed_file)
    assert reader.full_data.shape == (1, LR_count_32k)
    bms = reader.brain_models()
    lbl_tab = reader.label_tables()[0]
    local2key = {}
    for k, v in lbl_tab.items():
        local2key[v.label] = k
    seed_map = reader.get_data(hemi2stru[hemi], True)[0]
    offset, count, hemi_shape, idx2vtx = reader.get_stru_pos(hemi2stru[hemi])

    # get vertex neighbors
    mmp_map = CiftiReader(mmp_map_file).get_data(
        hemi2stru[hemi], True)[0]
    mask = np.zeros(hemi_shape, np.uint8)
    for roi in get_rois(mask_name):
        mask[mmp_map == mmp_name2label[roi]] = 1
    faces = GiftiReader(hemi2geo_file[hemi]).faces
    vtx2neighbors = get_n_ring_neighbor(faces, mask=mask)

    # get source data
    src_map = CiftiReader(src_file).get_data(
        hemi2stru[hemi], True)[1]

    # calculating
    n_local = len(local_names)
    out_maps = np.zeros((n_local, LR_count_32k), np.uint8)
    lbl_tabs = []
    map_names = []
    for local_idx, local_name in enumerate(local_names):
        if 'lateral' in local_name:
            def compare_func(x, y):
                return x < y
        else:
            def compare_func(x, y):
                return x > y
        hemi_map = np.zeros(hemi_shape, np.uint8)
        base_vertices = np.where(
            seed_map == local2key[local_name])[0]
        hemi_map[base_vertices] = 1
        while len(base_vertices) > 0:
            base_vertices_tmp = []
            for base_vtx in base_vertices:
                for neigh_vtx in vtx2neighbors[base_vtx]:
                    if hemi_map[neigh_vtx] == 1:
                        continue
                    if compare_func(src_map[neigh_vtx], src_map[base_vtx]):
                        hemi_map[neigh_vtx] = 1
                        base_vertices_tmp.append(neigh_vtx)
            base_vertices = base_vertices_tmp
        out_maps[local_idx, offset:(offset+count)] = hemi_map[idx2vtx]
        lbl_tab_new = nib.cifti2.Cifti2LabelTable()
        lbl_tab_new[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
        lbl_tab_new[1] = nib.cifti2.Cifti2Label(1, local_name, 0, 0, 1, 1)
        lbl_tabs.append(lbl_tab_new)
        map_names.append(local_name)

    # save out
    save2cifti(out_file, out_maps, bms, map_names, label_tables=lbl_tabs)


def expand_observed_seeds1(hemi='rh'):
    """
    对每个局部的seed region：
    1. 以其所有顶点为起点
    2. 遍历各扩张起点，对于每个起点，合并1环近邻中大于它，
        并且不属于该region的顶点，同时作为下一步扩张的起点。
    3. 重复第2步，直到没有扩张起点为止。
    4. 结果存为一个map，属于region的顶点标记为1，其它为0
    """
    Hemi = hemi2Hemi[hemi]
    mask_name = f'MMP-vis3-{Hemi}'
    hemi2geo_file = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     f'{mask_name}_zscore1_PCA-subj.dscalar.nii')

    local_names = ('early', 'dorsal', 'lateral', 'ventral')
    local_names = [f'{Hemi}_{i}' for i in local_names]
    seed_file = pjoin(work_dir, f'observed-seed-v4_{mask_name}.dlabel.nii')
    out_file = pjoin(work_dir, f'observed-seed-v4-expansion_{mask_name}.dlabel.nii')

    # prepare map information
    reader = CiftiReader(seed_file)
    assert reader.full_data.shape == (1, LR_count_32k)
    bms = reader.brain_models()
    lbl_tab = reader.label_tables()[0]
    local2key = {}
    for k, v in lbl_tab.items():
        local2key[v.label] = k
    seed_map = reader.get_data(hemi2stru[hemi], True)[0]
    offset, count, hemi_shape, idx2vtx = reader.get_stru_pos(hemi2stru[hemi])

    # get vertex neighbors
    mmp_map = CiftiReader(mmp_map_file).get_data(
        hemi2stru[hemi], True)[0]
    mask = np.zeros(hemi_shape, np.uint8)
    for roi in get_rois(mask_name):
        mask[mmp_map == mmp_name2label[roi]] = 1
    faces = GiftiReader(hemi2geo_file[hemi]).faces
    vtx2neighbors = get_n_ring_neighbor(faces, mask=mask)

    # get source data
    src_map = CiftiReader(src_file).get_data(
        hemi2stru[hemi], True)[1]

    # calculating
    n_local = len(local_names)
    out_maps = np.zeros((n_local, LR_count_32k), np.uint8)
    lbl_tabs = []
    map_names = []
    for local_idx, local_name in enumerate(local_names):
        hemi_map = np.zeros(hemi_shape, np.uint8)
        base_vertices = np.where(
            seed_map == local2key[local_name])[0]
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
        out_maps[local_idx, offset:(offset+count)] = hemi_map[idx2vtx]
        lbl_tab_new = nib.cifti2.Cifti2LabelTable()
        lbl_tab_new[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
        lbl_tab_new[1] = nib.cifti2.Cifti2Label(1, local_name, 0, 0, 1, 1)
        lbl_tabs.append(lbl_tab_new)
        map_names.append(local_name)

    # save out
    save2cifti(out_file, out_maps, bms, map_names, label_tables=lbl_tabs)


def watershed_PC2():
    Hemis = ('L', 'R')
    Hemi2geo_file = {
        'L': s1200_midthickness_L,
        'R': s1200_midthickness_R}
    pc_files = pjoin(anal_dir, 'decomposition/HCPY-M+corrT_'
                     '{vis_name}_zscore1_PCA-subj.dscalar.nii')
    seed_file = pjoin(anal_dir, 'mask_map/EDLV-seed-v1.dlabel.nii')
    out_file = pjoin(work_dir, 'watershed-PC2_EDLV-seed-v1.dlabel.nii')

    # prepare map information
    reader = CiftiReader(seed_file)
    assert reader.full_data.shape == (1, LR_count_32k)
    bms = reader.brain_models()
    lbl_tab1 = reader.label_tables()[0]
    reader_mmp = CiftiReader(mmp_map_file)

    boundary_key = np.max(list(lbl_tab1.keys())) + 1
    out_maps = np.zeros((2, LR_count_32k), np.uint8)
    mns = ['basins and boundary', 'boundary']
    lbl_tab2 = nib.cifti2.Cifti2LabelTable()
    lbl_tab2[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    for Hemi in Hemis:
        vis_name = f'MMP-vis3-{Hemi}'
        pc_file = pc_files.format(vis_name=vis_name)
        pc_map = CiftiReader(pc_file).get_data(Hemi2stru[Hemi], True)[1]
        vtx2label = reader.get_data(Hemi2stru[Hemi], True)[0]
        offset, count, hemi_shape, idx2vtx = \
            reader.get_stru_pos(Hemi2stru[Hemi])

        # get vertex neighbors
        mmp_map = reader_mmp.get_data(Hemi2stru[Hemi], True)[0]
        mask = np.zeros(hemi_shape, np.uint8)
        for roi in get_rois(vis_name):
            mask[mmp_map == mmp_name2label[roi]] = 1
        faces = GiftiReader(Hemi2geo_file[Hemi]).faces
        vtx2neighbors = get_n_ring_neighbor(faces, mask=mask)

        vtx2label = watershed(pc_map, vtx2label, vtx2neighbors)
        hemi_map1 = vtx2label[idx2vtx]
        hemi_map2 = np.zeros(count, np.uint8)
        boundary_idx_map = hemi_map1 == -1
        hemi_map1[boundary_idx_map] = boundary_key
        hemi_map2[boundary_idx_map] = boundary_key
        lbl = nib.cifti2.Cifti2Label(
            boundary_key, f'{Hemi}_boundary', 1, 1, 1, 1)
        lbl_tab1[boundary_key] = lbl
        lbl_tab2[boundary_key] = lbl

        out_maps[0, offset:(offset+count)] = hemi_map1
        out_maps[1, offset:(offset+count)] = hemi_map2
        boundary_key += 1

    save2cifti(out_file, out_maps, bms, mns,
               label_tables=[lbl_tab1, lbl_tab2])


if __name__ == '__main__':
    # get_extremum_vertices(hemi='lh')
    # get_extremum_vertices(hemi='rh')
    # get_lowest_seeds()
    # expand_seed_combo()

    # get_EDLV_seeds(Hemi='R')
    # expand_EDLV_seeds(hemi='rh')

    # get_observed_seeds(hemi='rh')
    # expand_observed_seeds(hemi='rh')
    # expand_observed_seeds1(hemi='rh')
    watershed_PC2()
