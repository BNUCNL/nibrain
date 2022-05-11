import os
import subprocess
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr
from nibabel.gifti import GiftiDataArray, GiftiImage
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


def smooth_msp():
    """
    对各msp map进行平滑
    """
    sigma = '2'
    hemi = 'rh'
    n_layer = 6
    surf_file = '/nfs/s2/userhome/chenxiayu/workingdir/test/bigbrain/'\
        'bigbrain.loris.ca/BigBrainRelease.2015/Surface_Parcellations/'\
        f'BigBrain_space/Surfaces/{hemi}.white.anat.surf.gii'
    msp_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}_{2}.func.gii')
    out_file1 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}_{2}_s{3}.func.gii')
    out_file2 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_s{3}.func.gii')

    for lyr_idx in range(n_layer):
        cmd = ['wb_command', '-metric-smoothing', surf_file]
        cmd.append(msp_file.format(lyr_idx, lyr_idx+1, hemi))
        cmd.append(sigma)
        cmd.append(out_file1.format(lyr_idx, lyr_idx+1, hemi, sigma))
        print('Running: ' + ' '.join(cmd) + '\n')
        subprocess.run(cmd)
        gii1 = nib.load(out_file1.format(lyr_idx, lyr_idx+1, hemi, sigma))
        data = []
        for i in gii1.darrays:
            data.append(i.data)
        print(len(data))
        gii2 = GiftiImage(darrays=[GiftiDataArray(np.mean(data, 0))])
        nib.save(gii2, out_file2.format(lyr_idx, lyr_idx+1, hemi, sigma))


def mask_msp():
    """
    把各层平均map在PC1中属于nan的部分的值设置为nan
    """
    hemi = 'rh'
    Hemi = hemi2Hemi[hemi]
    n_layer = 6
    resample_way = '164fsLR2bigbrain'
    msp_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_s2.func.gii')
    pc_file = pjoin(anal_dir,
                    'decomposition/HCPY-M+T_MMP-vis3-'
                    f'{Hemi}_zscore1_PCA-subj_{resample_way}.func.gii')
    out_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_s2_mask.func.gii')

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
    msp_file1 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}_{2}_s2.func.gii')
    msp_file2 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_s2.func.gii')
    pc_file = pjoin(anal_dir,
                    'decomposition/HCPY-M+T_MMP-vis3-'
                    f'{Hemi}_zscore1_PCA-subj_{resample_way}.func.gii')
    pc_names = ('C1', 'C2')
    out_file = pjoin(work_dir, f'{resample_way}_{Hemi}_s2.pkl')

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


def msp_fit_PC12(hemi='rh', method='ordinary'):
    """
    用6层胞体密度map作为特征分别对结构梯度的PC1/2做线性拟合
    比较各特征系数的大小
    """
    Hemi = hemi2Hemi[hemi]
    n_layer = 6
    resample_way = '164fsLR2bigbrain'
    pc_file = pjoin(anal_dir,
                    'decomposition/HCPY-M+T_MMP-vis3-'
                    f'{Hemi}_zscore1_PCA-subj_{resample_way}.func.gii')
    msp_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_s2.func.gii')
    pc_names = ('C1', 'C2')
    n_pc = len(pc_names)
    out_file = pjoin(work_dir, f'Msp1~6-s2_fit-{method}_PC12-{resample_way}_{Hemi}.pkl')

    # prepare Y
    pc_gii = nib.load(pc_file)
    non_nan_vec = None
    Y = []
    for pc_idx in range(n_pc):
        pc_map = pc_gii.darrays[pc_idx].data
        if non_nan_vec is None:
            non_nan_vec = ~np.isnan(pc_map)
        else:
            assert np.all(non_nan_vec == ~np.isnan(pc_map))
        Y.append(pc_map[non_nan_vec])
    Y = np.array(Y).T

    # prepare X
    X = []
    for lyr_idx in range(n_layer):
        msp_gii = nib.load(msp_file.format(lyr_idx, lyr_idx+1, hemi))
        X.append(msp_gii.darrays[0].data[non_nan_vec])
    X = np.array(X).T
    
    # prepare model
    if method == 'ordinary':
        model = Pipeline([('preprocesser', StandardScaler()),
                          ('regressor', LinearRegression())])
    elif method == 'lasso':
        model = Pipeline([('preprocesser', StandardScaler()),
                          ('regressor', Lasso())])
    else:
        raise ValueError('not supported method:', method)
    model.fit(X, Y)
    Y_pred = model.predict(X)
    scores = [r2_score(Y[:, i], Y_pred[:, i]) for i in range(n_pc)]

    # save out
    out_dict = {
        'PC name': pc_names, 'layer ID': list(range(1, n_layer+1)),
        'Y': Y, 'Y_pred': Y_pred, 'model': model, 'R2': scores,
        'coef': model.named_steps['regressor'].coef_}
    pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    # get_msp_from_FFA_proj()
    # smooth_msp()
    # mask_msp()
    # PC12_corr_msp()
    msp_fit_PC12(hemi='rh', method='ordinary')
    msp_fit_PC12(hemi='rh', method='lasso')
