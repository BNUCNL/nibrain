import os
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr
from nibabel.gifti import GiftiDataArray, GiftiImage
from cxy_visual_dev.lib.predefine import proj_dir, hemi2Hemi

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'bigbrain')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def get_msp_from_FFA_proj():
    """
    把之前在搞FFA那个项目的时候计算的胞体密度数据
    换成更标准一些的格式存到这边，并计算每个layer的
    平均map（之前对每个layer采样了10个surface）
    """
    hemi = 'rh'
    n_layer = 6
    n_interbedded = 10
    msp_file = '/nfs/t3/workingshop/chenxiayu/study/'\
        'FFA_pattern/analysis/s2/1080_fROI/refined_with_Kevin/bigbrain/'\
        'Msp_BB_layer{0}-{1}_{2}.gii'
    out_file1 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}_{2}.func.gii')
    out_file2 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}.func.gii')

    for layer_idx in range(n_layer):
        msp = nib.load(msp_file.format(
            layer_idx, layer_idx+1, hemi)).darrays[0].data
        assert msp.shape[0] == n_interbedded
        darrays = []
        for i in range(n_interbedded):
            darrays.append(GiftiDataArray(msp[i]))
        gii1 = GiftiImage(darrays=darrays)
        gii2 = GiftiImage(darrays=[GiftiDataArray(np.mean(msp, 0))])
        nib.save(gii1, out_file1.format(layer_idx, layer_idx+1, hemi))
        nib.save(gii2, out_file2.format(layer_idx, layer_idx+1, hemi))


def mask_msp():
    """
    把各层平均map在PC1中属于nan的部分的值设置为nan
    """
    hemi = 'rh'
    Hemi = hemi2Hemi[hemi]
    n_layer = 6
    resample_way = '164fsLR2bigbrain'
    msp_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}.func.gii')
    pc_file = pjoin(anal_dir,
                    'decomposition/HCPY-M+T_MMP-vis3-'
                    f'{Hemi}_zscore1_PCA-subj_{resample_way}.func.gii')
    out_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_mask.func.gii')

    pc_map = nib.load(pc_file).darrays[0].data
    nan_vec = np.isnan(pc_map)
    for layer_idx in range(n_layer):
        msp_gii = nib.load(msp_file.format(layer_idx, layer_idx+1, hemi))
        msp_gii.darrays[0].data[nan_vec] = np.nan
        nib.save(msp_gii, out_file.format(layer_idx, layer_idx+1, hemi))


def PC12_corr_msp():
    """
    C1_corr_layer: PC1和各层10个细分map的相关
    C2_corr_layer: PC2和各层10个细分map的相关
    C1_corr_layer-mean: PC1和各层平均map的相关
    C2_corr_layer-mean: PC2和各层平均map的相关
    """
    hemi = 'rh'
    Hemi = hemi2Hemi[hemi]
    n_layer = 6
    n_interbedded = 10
    resample_way = '164fsLR2bigbrain'
    # resample_way = 'fsavg2bigbrain'
    msp_file1 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}_{2}.func.gii')
    msp_file2 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}.func.gii')
    pc_file = pjoin(anal_dir,
                    'decomposition/HCPY-M+T_MMP-vis3-'
                    f'{Hemi}_zscore1_PCA-subj_{resample_way}.func.gii')
    pc_names = ('C1', 'C2')
    out_file = pjoin(work_dir, f'{resample_way}_{Hemi}.pkl')

    pc_gii = nib.load(pc_file)
    non_nan_vec = None
    out_dict = {'layer_name': tuple(range(1, n_layer+1))}
    for pc_idx, pc_name in enumerate(pc_names):
        pc_map = pc_gii.darrays[pc_idx].data
        if non_nan_vec is None:
            non_nan_vec = ~np.isnan(pc_map)
        else:
            assert np.all(non_nan_vec == ~np.isnan(pc_map))
        pc_map = pc_map[non_nan_vec]
        data1 = np.zeros((n_interbedded, n_layer), np.float64)
        data1_p = np.zeros((n_interbedded, n_layer), np.float64)
        data2 = np.zeros(n_layer, np.float64)
        data2_p = np.zeros(n_layer, np.float64)
        for layer_idx in range(n_layer):
            msp_gii1 = nib.load(msp_file1.format(layer_idx, layer_idx+1, hemi))
            for i in range(n_interbedded):
                msp_map = msp_gii1.darrays[i].data[non_nan_vec]
                r, p = pearsonr(pc_map, msp_map)
                data1[i, layer_idx] = r
                data1_p[i, layer_idx] = p

            msp_gii2 = nib.load(msp_file2.format(layer_idx, layer_idx+1, hemi))
            msp_map = msp_gii2.darrays[0].data[non_nan_vec]
            r, p = pearsonr(pc_map, msp_map)
            data2[layer_idx] = r
            data2_p[layer_idx] = p
        out_dict[f'{pc_name}_corr_layer'] = data1
        out_dict[f'{pc_name}_corr_layer-mean'] = data2
        out_dict[f'{pc_name}_corr_layer-p'] = data1_p
        out_dict[f'{pc_name}_corr_layer-mean-p'] = data2_p

    pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    # get_msp_from_FFA_proj()
    mask_msp()
    # PC12_corr_msp()
