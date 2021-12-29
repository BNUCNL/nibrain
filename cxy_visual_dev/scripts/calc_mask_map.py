import os
import numpy as np
from os.path import join as pjoin
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


if __name__ == '__main__':
    atlas = Atlas('HCP-MMP')
    mask = atlas.get_mask(get_rois('MMP-vis3-L') + get_rois('MMP-vis3-R'))[0]
    mask_maps(
        data_file='/nfs/m1/hcp/ACF-decay.dscalar.nii',
        mask=mask,
        out_file=pjoin(work_dir, 'HCPY-ACF-decay_MMP-vis3.dscalar.nii')
    )
