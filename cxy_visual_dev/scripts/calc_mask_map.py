import os
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from matplotlib import pyplot as plt
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir,\
    Atlas, get_rois, All_count_32k, LR_count_32k,\
    mmp_map_file

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'mask_map')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def mask_maps(data_file, mask, out_file):
    """
    把data map在指定mask以外的部分全赋值为nan

    Args:
        data_file (str): end with .dscalar.nii
            shape=(n_map, n_vtx)
        mask (1D index array)
        out_file (str):
    """
    # prepare
    reader1 = CiftiReader(mmp_map_file)
    reader2 = CiftiReader(data_file)
    data = reader2.get_data()
    if data.shape[1] == All_count_32k:
        data = data[:, :LR_count_32k]
    elif data.shape[1] == LR_count_32k:
        pass
    else:
        raise ValueError

    # calculate
    data[:, ~mask] = np.nan

    # save
    save2cifti(out_file, data, reader1.brain_models(), reader2.map_names())


def make_mask1():
    """
    将HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj的PC1和PC2分段
    以值排序，然后切割成N段顶点数量基本相同的片段
    """
    N = 10
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


if __name__ == '__main__':
    # atlas = Atlas('HCP-MMP')
    # mask = atlas.get_mask(get_rois('MMP-vis3-L') + get_rois('MMP-vis3-R'))[0]
    # mask_maps(
    #     data_file='/nfs/m1/hcp/ACF-decay.dscalar.nii',
    #     mask=mask,
    #     out_file=pjoin(work_dir, 'HCPY-ACF-decay_MMP-vis3.dscalar.nii')
    # )

    make_mask1()
