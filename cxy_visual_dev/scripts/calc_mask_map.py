import os
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from matplotlib import pyplot as plt
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir,\
    Atlas, get_rois, All_count_32k, LR_count_32k,\
    mmp_map_file, s1200_avg_RFsize, s1200_avg_R2,\
    s1200_avg_curv, s1200_avg_myelin,\
    s1200_avg_thickness, s1200_avg_corrThickness,\
    s1200_avg_angle, s1200_avg_eccentricity,\
    hemi2Hemi, hemi2stru, s1200_midthickness_L,\
    s1200_midthickness_R, Hemi2stru

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'mask_map')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def mask_cii(src_file, mask, out_file):
    """
    把data map在指定mask以外的部分全赋值为nan

    Args:
        src_file (str): end with .dscalar.nii/.dlabel.nii
            shape=(n_map, n_vtx)
        mask (1D index array)
        out_file (str): end with .dscalar.nii/.dlabel.nii
    """
    # prepare
    reader = CiftiReader(src_file)
    data = reader.get_data()

    # calculate
    if src_file.endswith('.dlabel.nii'):
        data[:, ~mask] = 0
    else:
        data[:, ~mask] = np.nan

    # save
    save2cifti(out_file, data, reader.brain_models(),
               reader.map_names(), reader.volume, reader.label_tables())


def make_mask1(N):
    """
    将HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj的PC1和PC2分段
    以值排序，然后切割成N段顶点数量基本相同的片段
    """
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')
    map_names = ['C1', 'C2']
    mask = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-R'))[0]
    out_file = pjoin(work_dir, f'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_N{N}.dlabel.nii')

    n_vtx = np.sum(mask)
    step = int(np.ceil(n_vtx / N))
    bounds = np.arange(0, n_vtx, step)
    bounds = np.r_[bounds, n_vtx]
    print(bounds)

    n_map = len(map_names)
    reader = CiftiReader(src_file)
    src_maps = reader.get_data()[:n_map]
    assert map_names == reader.map_names()[:n_map]

    lbl_tabs = []
    cmap = plt.cm.jet
    color_indices = np.linspace(0, 1, N)
    out_maps = np.zeros_like(src_maps, np.uint8)
    for map_idx in range(n_map):
        data = src_maps[map_idx, mask]
        vtx_indices = np.argsort(data)
        lbl_tab = nib.cifti2.Cifti2LabelTable()
        lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
        for s_idx, s_bound in enumerate(bounds[:-1]):
            e_idx = s_idx + 1
            e_bound = bounds[e_idx]
            batch = vtx_indices[s_bound:e_bound]
            data[batch] = e_idx
            lbl = nib.cifti2.Cifti2Label(e_idx, f'{s_bound}~{e_bound}',
                                         *cmap(color_indices[s_idx]))
            lbl_tab[e_idx] = lbl
        out_maps[map_idx, mask] = data
        lbl_tabs.append(lbl_tab)

    save2cifti(out_file, out_maps, reader.brain_models(), map_names,
               reader.volume, lbl_tabs)


def make_mask2():
    """
    将make_mask1得到的PC1的N段和PC2的N段相交得到N*N个region。
    PC1第i段中依照PC2的N段分的区域编号为(i-1)N+1~iN
    """
    N = 3
    src_file = pjoin(work_dir, f'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_N{N}.dlabel.nii')
    out_file = pjoin(work_dir, f'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_mask2-N{N}.dlabel.nii')

    reader = CiftiReader(src_file)
    src_maps = reader.get_data()
    names = reader.map_names()

    out_map = np.zeros((1, src_maps.shape[1]), np.uint8)
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    cmap = plt.cm.jet
    color_indices = np.linspace(0, 1, N*N)
    for i in range(1, N+1):
        idx_map1 = src_maps[0] == i
        for j in range(1, N+1):
            idx_map2 = src_maps[1] == j
            idx_map = np.logical_and(idx_map1, idx_map2)
            k = (i-1)*N + j
            lbl = f'{names[0]}-{i} & {names[1]}-{j}'
            out_map[0, idx_map] = k
            lbl_tab[k] = nib.cifti2.Cifti2Label(k, lbl, *cmap(color_indices[k-1]))

    save2cifti(out_file, out_map, reader.brain_models(), volume=reader.volume,
               label_tables=[lbl_tab])


def make_mask3():
    """
    依据HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj的PC1的大小排序分成N等分
    然后在PC1的各层级内，按照PC2的大小排序分成N等分
    """
    N = 3
    src_file = pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')
    map_names = ['C1', 'C2']
    mask = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-R'))[0]
    out_file = pjoin(work_dir, f'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_{N}x{N}.dlabel.nii')

    n_map = len(map_names)
    reader = CiftiReader(src_file)
    src_maps = reader.get_data()[:n_map]
    assert map_names == reader.map_names()[:n_map]
    pc_data1 = src_maps[0, mask]
    pc_indices1 = np.argsort(pc_data1)
    pc_data2 = src_maps[1, mask]

    n_vtx1 = np.sum(mask)
    step1 = int(np.ceil(n_vtx1 / N))
    bounds1 = np.arange(0, n_vtx1, step1)
    bounds1 = np.r_[bounds1, n_vtx1]
    print(bounds1)

    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    cmap = plt.cm.jet
    color_indices = np.linspace(0, 1, N*N)
    out_map = np.zeros((1, src_maps.shape[1]), np.uint8)
    k = 1
    for s_idx1, s_bound1 in enumerate(bounds1[:-1]):
        e_idx1 = s_idx1 + 1
        e_bound1 = bounds1[e_idx1]
        batch1 = pc_indices1[s_bound1:e_bound1]
        pc_data2_tmp = pc_data2[batch1]
        pc_indices2 = np.argsort(pc_data2_tmp)
        n_vtx2 = len(batch1)
        step2 = int(np.ceil(n_vtx2 / N))
        bounds2 = np.arange(0, n_vtx2, step2)
        bounds2 = np.r_[bounds2, n_vtx2]
        print(bounds2)
        for s_idx2, s_bound2 in enumerate(bounds2[:-1]):
            e_idx2 = s_idx2 + 1
            e_bound2 = bounds2[e_idx2]
            batch2 = pc_indices2[s_bound2:e_bound2]
            pc_data2_tmp[batch2] = k
            lbl = nib.cifti2.Cifti2Label(k, f'PC1-{e_idx1}+PC2-{e_idx2}',
                                         *cmap(color_indices[k-1]))
            lbl_tab[k] = lbl
            k += 1
        pc_data1[batch1] = pc_data2_tmp
    out_map[0, mask] = pc_data1

    save2cifti(out_file, out_map, reader.brain_models(),
               volume=reader.volume, label_tables=[lbl_tab])


def make_mask6():
    """
    依据各ROI在HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj中的PC1的值排序分成4级
    """
    n_level = 4
    fpath = pjoin(anal_dir, 'ROI_scalar/ROI_scalar1_MMP-vis3-R.csv')
    atlas = Atlas('HCP-MMP')
    out_file = pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_4level-ROI.dlabel.nii')

    df = pd.read_csv(fpath, index_col=0)
    idx = 'mean_stru-C1'
    rois_list = [[] for _ in range(n_level)]
    rois_list[0].append('R_V1')

    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    cmap = plt.cm.jet
    color_indices = np.linspace(0, 1, n_level)
    out_map = np.zeros((1, LR_count_32k), np.uint8)
    for roi in df.columns:
        x = df.loc[idx, roi]
        if df.loc[idx, 'R_V2'] <= x <= df.loc[idx, 'R_MT']:
            rois_list[1].append(roi)
        elif df.loc[idx, 'R_MT'] < x < df.loc[idx, 'R_PIT']:
            rois_list[2].append(roi)
        elif df.loc[idx, 'R_PIT'] <= x < df.loc[idx, 'R_TF']:
            rois_list[3].append(roi)

    for key, rois in enumerate(rois_list, 1):
        out_map[atlas.get_mask(rois)] = key
        lbl = nib.cifti2.Cifti2Label(key, f'level{key}',
                                     *cmap(color_indices[key-1]))
        lbl_tab[key] = lbl

    reader = CiftiReader(mmp_map_file)
    save2cifti(out_file, out_map, reader.brain_models(), label_tables=[lbl_tab])


def make_mask7():
    """
    直接指定某些顶点为某个ROI
    """
    # Hemis = ('L', 'R')
    # seed_name2vtx_L = {
    #     'early': [24939, 24324],
    #     'dorsal': [12485, 12545],
    #     'lateral': [23474, 15570],
    #     'ventral': [21501, 22655]}
    # seed_name2vtx_R = {
    #     'early': [24938, 24433],
    #     'dorsal': [12442, 12586],
    #     'lateral': [23526, 15523],
    #     'ventral': [21501, 22485]}
    # out_file = pjoin(work_dir, 'EDLV-seed-v1.dlabel.nii')

    Hemis = ('L', 'R')
    seed_name2vtx_L = {
        'early': [24939, 24324],
        'dorsal': [12485, 12545],
        'lateral': [15294],
        'ventral': [21501, 22655]}
    seed_name2vtx_R = {
        'early': [24938, 24433],
        'dorsal': [12442, 12586],
        'lateral': [15241],
        'ventral': [21501, 22485]}
    out_file = pjoin(work_dir, 'EDLV-seed.dlabel.nii')

    Hemi2seed_dict = {'L': seed_name2vtx_L, 'R': seed_name2vtx_R}
    n_seed = len(Hemi2seed_dict['L']) + len(Hemi2seed_dict['R'])
    cmap = plt.cm.jet
    color_indices = np.linspace(0, 1, n_seed)

    # prepare atlas information
    reader = CiftiReader(mmp_map_file)
    LR_shape = reader.full_data.shape
    assert (1, LR_count_32k) == LR_shape
    bms = reader.brain_models()

    out_map = np.zeros(LR_shape, dtype=np.uint8)
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    seed_key = 0
    for Hemi in Hemis:
        offset, count, hemi_shape, idx2vtx = \
            reader.get_stru_pos(Hemi2stru[Hemi])
        hemi_map = np.zeros(hemi_shape, dtype=np.uint8)
        for seed_name, seed_vertices in Hemi2seed_dict[Hemi].items():
            seed_color = cmap(color_indices[seed_key])
            seed_key += 1
            hemi_map[seed_vertices] = seed_key
            lbl_tab[seed_key] = nib.cifti2.Cifti2Label(
                seed_key, f'{Hemi}_{seed_name}', *seed_color)
        out_map[0, offset:(offset+count)] = hemi_map[idx2vtx]
    save2cifti(out_file, out_map, bms, label_tables=[lbl_tab])


if __name__ == '__main__':
    atlas = Atlas('HCP-MMP')
    mask = atlas.get_mask(get_rois('MMP-vis3-L') + get_rois('MMP-vis3-R'))[0]
    # mask_maps(
    #     data_file=s1200_avg_RFsize,
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'S1200-avg-RFsize_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus-split.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-CalcarineSulcus-split_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'gdist/gdist_src-OpMt.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-OpMt_MMP-vis3.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(anal_dir, 'variation/MMP-vis3_ring1-CS1_R_width5.dlabel.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'MMP-vis3_ring1-CS1_R_width5_mask-MMP-vis3.dlabel.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'summary_map/HCPY-corrThickness_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-corrThickness_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'summary_map/HCPY-myelin_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-myelin_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'summary_map/HCPY-thickness_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-thickness_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'summary_map/HCPY-curv_mean.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-curv_mean_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'gdist/gdist_src-Calc+MT.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-Calc+MT_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'gdist/gdist_src-Calc+MT=V4.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-Calc+MT=V4_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'gdist/gdist_src-OP+MT.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-OP+MT_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'gdist/gdist_src-OP+MT=V4.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-OP+MT=V4_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'gdist/gdist_src-OP.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist_src-OP_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'gdist/gdist4_src-EDLV-seed-v1.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist4_src-EDLV-seed-v1_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'AHEAD/AHEAD-YA_t1map.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'AHEAD-YA_t1map_MMP-vis3.dscalar.nii')
    # )
    mask_cii(
        src_file=pjoin(anal_dir, 'AHEAD/AHEAD-YA_t2starmap.dscalar.nii'),
        mask=mask,
        out_file=pjoin(work_dir, 'AHEAD-YA_t2starmap_MMP-vis3.dscalar.nii')
    )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'AHEAD/AHEAD-YA_qsm.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'AHEAD-YA_qsm_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'AHEAD/AHEAD-YA_r1map.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'AHEAD-YA_r1map_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'AHEAD/AHEAD-YA_r2starmap.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'AHEAD-YA_r2starmap_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'AHEAD/AHEAD-YA_t1w.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'AHEAD-YA_t1w_MMP-vis3.dscalar.nii')
    # )

    # atlas = Atlas('HCP-MMP')
    # mask = atlas.get_mask(get_rois('MMP-vis3-L') + get_rois('MMP-vis3-R'), 'grayordinate')[0]
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'tfMRI/tfMRI-WM-cope.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'tfMRI-WM-cope_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=s1200_avg_eccentricity,
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'S1200-avg-ECC_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=s1200_avg_angle,
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'S1200-avg-angle_MMP-vis3.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'AFF/HCPY-faff.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-faff_MMP-vis3.dscalar.nii')
    # )

    # atlas = Atlas('HCP-MMP')
    # mask = atlas.get_mask(get_rois('MMP-vis3-R'))[0]
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'gdist/gdist4_src-observed-seed-v4_R.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist4_src-observed-seed-v4_MMP-vis3-R.dscalar.nii')
    # )
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'gdist/gdist4_src-EDLV-seed_R.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist4_src-EDLV-seed_MMP-vis3-R.dscalar.nii')
    # )

    # atlas = Atlas('HCP-MMP')
    # mask = atlas.get_mask(get_rois('MMP-vis3-L'))[0]
    # mask_cii(
    #     src_file=pjoin(anal_dir, 'gdist/gdist4_src-EDLV-seed_L.dscalar.nii'),
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'gdist4_src-EDLV-seed_MMP-vis3-L.dscalar.nii')
    # )

    # make_mask1(N=2)
    # make_mask1(N=3)
    # make_mask1(N=5)
    # make_mask2()
    # make_mask3()
    # make_mask6()
    # make_mask7()

    # atlas = Atlas('HCP-MMP')
    # R2_mask = nib.load(s1200_avg_R2).get_fdata()[0, :LR_count_32k] > 9.8
    # mask = atlas.get_mask(get_rois('MMP-vis3-R'))[0]
    # mask_maps(
    #     data_file=pjoin(anal_dir,
    #                     'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     mask=np.logical_and(R2_mask, mask),
    #     out_file=pjoin(work_dir, 'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj_R2.dscalar.nii')
    # )
