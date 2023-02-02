import os
import time
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir, get_rois

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'stru_conn')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def apply_affine(affine, coords):
    """
    Change coordinates according to affine

    Args:
        affine (array): 4 x 4
        coords (array): 3 x N

    References:
        1. https://nipy.org/nibabel/coordinate_systems.html#applying-the-affine
    """
    M = affine[:3, :3]
    abc = affine[:3, [3]]
    return M.dot(coords) + abc


def xyz2ijk(affine, xyz):
    """
    Return i, j, k indices for x, y, z coordinates

    Args:
        affine (array): 4 x 4
            体素索引到物理坐标的映射矩阵
        xyz (array): 3 x N
    """
    affine = np.linalg.inv(affine)
    ijk = apply_affine(affine, xyz) + 0.5
    return ijk.astype(int)


def load_bundles(data_dir):
    """
    导入所有纤维束

    Args:
        data_dir (str): 纤维束数据目录
            例如HCP1065_PopulationAveraged_TractographyAtlas/tracks_trk_origin-AC

    Returns:
        (dict): 纤维束数据
            键：纤维束的名字
            值：nibabel.streamlines.array_sequence.ArraySequence
    """
    bundles = {}
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            bundle_name = fname.split('.')[0]
            bundle_obj = nib.streamlines.load(pjoin(root, fname))
            bundles[bundle_name] = bundle_obj.streamlines

    return bundles


def intersect_bundle(bundle, affine, roi_masks):
    """
    从纤维束中寻找穿过roi_masks中所有感兴趣区域(ROI)的纤维

    Args:
        bundle (ArraySequence): 纤维束
            序列中的每个数组的形状为Nx3
            代表一条纤维所经过N个点的xyz坐标
        affine (array): 4 x 4
            体素索引到物理坐标的映射矩阵
            来自存储ROI mask的NIFTI文件
        roi_masks (list): 一个及以上的ROI mask
            一个ROI mask就是一个volume，ROI由其中所有非零值的体素构成。
            要保证ROI mask被配准到ICBM 2009a Nonlinear Asymmetric space.

    Returns:
        (ArraySequence): 穿过所有ROI的纤维
            序列中的每个数组的形状为Nx3
            代表一条纤维所经过N个点的xyz坐标
        (array): 穿过所有ROI的纤维mask
            形状和ROI mask一样的bool数组，属于这些纤维的体素值为True
    """
    # select streamlines
    for roi_mask in roi_masks:
        roi_mask = np.asarray(roi_mask, dtype=bool)
        streamlines = nib.streamlines.ArraySequence()
        for sl in bundle:
            i, j, k = xyz2ijk(affine, sl.T)
            if roi_mask[i, j, k].any():
                streamlines.append(sl)
        bundle = streamlines

    # make bundle mask
    bundle_mask = np.zeros_like(roi_masks[0], bool)
    for sl in bundle:
        i, j, k = xyz2ijk(affine, sl.T)
        bundle_mask[i, j, k] = True

    return bundle, bundle_mask


def get_intersect_bundles(Hemi='R'):
    """
    为视觉皮层内各对脑区寻找相连的纤维丝
    """
    mask_name = f'MMP-vis3-{Hemi}'
    bundle_dir = '/nfs/h1/HCP1065_PopulationAveraged_TractographyAtlas/'\
        'tracks_trk_origin-AC'
    mmp_file = pjoin(proj_dir, 'data/HCP/HCP-MMP_LPS.nii.gz')
    mmp_label_file = pjoin(proj_dir, 'data/HCP/HCP-MMP.txt')
    out_file = pjoin(work_dir, f'intersect_bundles_{mask_name}.pkl')

    # prepare ROI info
    rois = get_rois(mask_name)
    mmp_nii = nib.load(mmp_file)
    mmp_vol = mmp_nii.get_fdata()
    mmp_roi2num = {}
    for line in open(mmp_label_file).read().splitlines():
        num, roi = line.split(' ')
        mmp_roi2num[roi] = int(num)
    roi2mask = {}
    for roi in rois:
        roi2mask[roi] = mmp_vol == mmp_roi2num[roi]

    # load all bundles
    bundles = load_bundles(bundle_dir)

    # calculating
    out_dict = {}
    affine = mmp_nii.affine
    count = 1
    for idx, roi1 in enumerate(rois[:-1], 1):
        out_dict[roi1] = {}
        for roi2 in rois[idx:]:
            time1 = time.time()
            roi_masks = [roi2mask[roi1], roi2mask[roi2]]
            bundle_dict = {}
            for bundle_name, bundle in bundles.items():
                bundle, _ = intersect_bundle(bundle, affine, roi_masks)
                if len(bundle) == 0:
                    continue
                bundle_dict[bundle_name] = bundle
            out_dict[roi1][roi2] = bundle_dict
            print(f'{count}-{roi1}-{roi2} cost {time.time() - time1} seconds.')
            count += 1

    # save out
    pkl.dump(out_dict, open(out_file, 'wb'))


def get_intersect_bundles1(Hemi='R'):
    """
    为视觉皮层内各脑区寻找与HCP-MMP的360个脑区相连的纤维丝
    """
    mask_name = f'MMP-vis3-{Hemi}'
    bundle_dir = '/nfs/h1/HCP1065_PopulationAveraged_TractographyAtlas/'\
        'tracks_trk_origin-AC'
    mmp_file = pjoin(proj_dir, 'data/HCP/HCP-MMP_LPS.nii.gz')
    mmp_label_file = pjoin(proj_dir, 'data/HCP/HCP-MMP.txt')
    out_file = pjoin(work_dir, f'intersect-bundles1_{Hemi}.pkl')

    # prepare ROI info
    vis_rois = get_rois(mask_name)
    mmp_nii = nib.load(mmp_file)
    mmp_vol = mmp_nii.get_fdata()
    mmp_roi2mask = {}
    for line in open(mmp_label_file).read().splitlines():
        num, roi = line.split(' ')
        mmp_roi2mask[roi] = mmp_vol == int(num)

    # load all bundles
    bundles = load_bundles(bundle_dir)

    # calculating
    out_dict = {'vis_roi': vis_rois,
                'mmp_roi': list(mmp_roi2mask.keys())}
    affine = mmp_nii.affine
    total = len(vis_rois)
    for idx, vis_roi in enumerate(vis_rois, 1):
        time1 = time.time()
        out_dict[vis_roi] = {}
        vis_roi_mask = mmp_roi2mask[vis_roi]
        vis_roi_bundles = {}
        for bundle_name, bundle in bundles.items():
            bundle, _ = intersect_bundle(bundle, affine, [vis_roi_mask])
            if len(bundle) == 0:
                continue
            vis_roi_bundles[bundle_name] = bundle
        for roi, roi_mask in mmp_roi2mask.items():
            if roi == vis_roi:
                out_dict[vis_roi][roi] = vis_roi_bundles
                continue
            bundle_dict = {}
            for bundle_name, bundle in vis_roi_bundles.items():
                bundle, _ = intersect_bundle(bundle, affine, [roi_mask])
                if len(bundle) == 0:
                    continue
                bundle_dict[bundle_name] = bundle
            out_dict[vis_roi][roi] = bundle_dict
        print(f'{idx}/{total}-{vis_roi} cost: '
              f'{time.time() - time1} seconds.')

    # save out
    pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    # get_intersect_bundles(Hemi='R')
    get_intersect_bundles1(Hemi='R')
