import os
import numpy as np
import pickle as pkl
import nibabel as nib

from os.path import join as pjoin
from scipy.io import savemat
from scipy.stats import zscore
from scipy.spatial.distance import euclidean
from magicbox.io.io import CiftiReader
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    get_rois, mmp_map_file, Hemi2stru

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'paper_fig')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def gradient_distance(Hemi):
    """
    Calculate gradient distance between each pair of visual cortex vertices
    PC1: absolute difference between primary gradient values of two vertices
    PC2: absolute difference between secondary gradient values of two vertices
    2D-PC: euclidean distance in the 2D gradient space constructed by
        the primary and secondary gradients.
    2D-PC-zscore: euclidean distance in the 2D gradient space constructed by
        the primary and secondary gradients after zscore.

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

    pc_maps_vis = zscore(pc_maps[:, vtx_indices], 1)
    pc_maps_zscore = np.ones_like(pc_maps) * np.nan
    pc_maps_zscore[0, vtx_indices] = pc_maps_vis[0]
    pc_maps_zscore[1, vtx_indices] = pc_maps_vis[1]
    data = {'PC1': np.zeros(n_pair), 'PC2': np.zeros(n_pair),
            '2D-PC': np.zeros(n_pair), '2D-PC-zscore': np.zeros(n_pair)}
    pair_idx = 0
    for idx, vtx_idx1 in enumerate(vtx_indices[:-1], 1):
        vtx1_pc = pc_maps[:, vtx_idx1]
        for vtx_idx2 in vtx_indices[idx:]:
            vtx2_pc = pc_maps[:, vtx_idx2]
            data['PC1'][pair_idx] = np.abs(vtx1_pc[0] - vtx2_pc[0])
            data['PC2'][pair_idx] = np.abs(vtx1_pc[1] - vtx2_pc[1])
            data['2D-PC'][pair_idx] = euclidean(vtx1_pc, vtx2_pc)
            data['2D-PC-zscore'][pair_idx] = euclidean(
                pc_maps_zscore[:, vtx_idx1], pc_maps_zscore[:, vtx_idx2])
            pair_idx += 1

    savemat(out_file, data)


def geodesic_distance(Hemi):
    """
    Get geodesic distance between each pair of visual cortex vertices

    Args:
        Hemi (str): L or R.
            L: left visual cortex
            R: right visual cortex
    """
    vis_name = f'MMP-vis3-{Hemi}'
    gdist_file = pjoin(
        anal_dir, f'gdist/gdist-between-all-pair-vtx_{vis_name}.pkl')
    out_file = pjoin(work_dir, f'geodesic_distance_{Hemi}.mat')

    # prepare visual vertex indices
    vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    vtx_indices = np.where(vis_mask)[0]
    n_vtx = len(vtx_indices)
    n_pair = int((n_vtx * n_vtx - n_vtx) / 2)

    # prepare geodesic data
    gdist = pkl.load(open(gdist_file, 'rb'))
    vertices = gdist[f'vtx_number_in_32k_fs_{Hemi}'].tolist()

    # prepare mapping from index to vertex number
    reader = CiftiReader(mmp_map_file)
    offset, _, _, idx2vtx = reader.get_stru_pos(Hemi2stru[Hemi])

    # get data
    data = {'data': np.zeros(n_pair)}
    pair_idx = 0
    vtx_indices = vtx_indices - offset
    for idx, vtx_idx1 in enumerate(vtx_indices[:-1], 1):
        row_idx = vertices.index(idx2vtx[vtx_idx1])
        for vtx_idx2 in vtx_indices[idx:]:
            col_idx = vertices.index(idx2vtx[vtx_idx2])
            data['data'][pair_idx] = gdist['gdist'][row_idx, col_idx]
            pair_idx += 1
    print(pair_idx)

    # save out
    savemat(out_file, data)


def RSFC_pair_vertices(Hemi):
    """
    Get RSFC between each pair of visual cortex vertices

    Args:
        Hemi (str): L or R.
            L: left visual cortex
            R: right visual cortex
    """
    vis_name = f'MMP-vis3-{Hemi}'
    rsfc_file = pjoin(
        proj_dir, f'data/HCP/HCPY-avg_RSFC-{vis_name}.pkl')
    out_file = pjoin(work_dir, f'HCPY-avg_RSFC-{vis_name}.mat')

    # prepare visual vertex indices
    vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    vtx_indices = np.where(vis_mask)[0]
    n_vtx = len(vtx_indices)
    idx_mat = np.tri(n_vtx, k=-1, dtype=bool).T

    # get RSFC data
    rsfc_dict = pkl.load(open(rsfc_file, 'rb'))
    assert np.all(vtx_indices == rsfc_dict['row-idx_to_32k-fs-LR-idx'])
    data = {'data': rsfc_dict['matrix'][idx_mat]}

    # save out
    savemat(out_file, data)


if __name__ == '__main__':
    gradient_distance(Hemi='R')
    gradient_distance(Hemi='L')
    # geodesic_distance(Hemi='R')
    # geodesic_distance(Hemi='L')
    # RSFC_pair_vertices(Hemi='R')
    # RSFC_pair_vertices(Hemi='L')
