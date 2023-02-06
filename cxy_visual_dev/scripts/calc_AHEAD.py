import os
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir, mmp_map_file,\
    Hemi2stru, LR_count_32k

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'AHEAD')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def get_mean_metric_YA(modality):
    """
    计算并保存18~40岁被试的平均map
    """
    Hemis = ('L', 'R')
    ages = ('18-30', '31-40')
    info_file = '/nfs/h1/AHEAD/participants.csv'
    src_files = '/nfs/h1/AHEAD/derivatives/vol2surf/native_to_32kfsLR/{sid}/'\
        '{sid}_ses-1_acq-wb_mod-{mod}_orient-std.{Hemi}.32k_fs_LR.func.gii'
    out_file = pjoin(work_dir, f'AHEAD-YA_{modality}.dscalar.nii')

    df = pd.read_csv(info_file)
    reader = CiftiReader(mmp_map_file)
    out_map = np.zeros((1, LR_count_32k))
    for Hemi in Hemis:
        offset, count, map_shape, idx2vtx = reader.get_stru_pos(
            Hemi2stru[Hemi])
        src_maps = []
        for idx in df.index:
            age = df.loc[idx, 'Group']
            if age not in ages:
                continue
            sid = df.loc[idx, 'ScanName']
            src_file = src_files.format(sid=sid, mod=modality, Hemi=Hemi)
            if not os.path.exists(src_file):
                continue
            src_maps.append(nib.load(src_file).darrays[0].data)
        print(f'{Hemi}H: n_sid={len(src_maps)}')
        avg_map = np.mean(src_maps, 0)
        out_map[0, offset:(offset+count)] = avg_map[idx2vtx]

    save2cifti(out_file, out_map, reader.brain_models())


if __name__ == '__main__':
    # get_mean_metric_YA(modality='t1map')
    get_mean_metric_YA(modality='t2starmap')
    get_mean_metric_YA(modality='qsm')
    get_mean_metric_YA(modality='r1map')
    get_mean_metric_YA(modality='r2starmap')
    get_mean_metric_YA(modality='t1w')
