import os
import numpy as np
import nibabel as nib

from os.path import join as pjoin
from scipy.io import savemat
from scipy.spatial.distance import euclidean
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    get_rois

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'paper_fig')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def gradient_distance(Hemi):
    """
    Calculate gradient distance for each pair of visual cortex vertices
    PC1: absolute difference between primary gradient values of two vertices
    PC2: absolute difference between secondary gradient values of two vertices
    2D-PC: euclidean distance in the 2D gradient space constructed by
        the primary and secondary gradients.

    Args:
        Hemi (str): L or R.
            L: left visual cortex
            R: right visual cortex
    """
    vis_name = f'MMP-vis3-{Hemi}'
    pc_file = pjoin(
        anal_dir,
        f'decomposition/HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    out_file = pjoin(work_dir, f'gradient_distance_{Hemi}.mat')

    vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    vtx_indices = np.where(vis_mask)[0]
    n_vtx = len(vtx_indices)
    n_pair = int((n_vtx * n_vtx - n_vtx) / 2)
    pc_maps = nib.load(pc_file).get_fdata()[:2]

    data = {'PC1': np.zeros(n_pair), 'PC2': np.zeros(n_pair),
            '2D-PC': np.zeros(n_pair)}
    pair_idx = 0
    for idx, vtx_idx1 in enumerate(vtx_indices[:-1], 1):
        vtx1_pc = pc_maps[:, vtx_idx1]
        for vtx_idx2 in vtx_indices[idx:]:
            vtx2_pc = pc_maps[:, vtx_idx2]
            data['PC1'][pair_idx] = np.abs(vtx1_pc[0] - vtx2_pc[0])
            data['PC2'][pair_idx] = np.abs(vtx1_pc[1] - vtx2_pc[1])
            data['2D-PC'][pair_idx] = euclidean(vtx1_pc, vtx2_pc)
            pair_idx += 1

    savemat(out_file, data)


if __name__ == '__main__':
    gradient_distance(Hemi='R')
    gradient_distance(Hemi='L')
