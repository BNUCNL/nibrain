import os
import subprocess
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr, permutation_test
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


def get_msp_from_FFA_proj(hemi):
    """
    把之前在搞FFA那个项目的时候计算的胞体密度数据
    换成更标准一些的格式存到这边，并计算每个layer的
    平均map（之前对每个layer采样了10个surface）
    """
    n_layer = 6
    n_interbedded = 10
    msp_file = '/nfs/h1/userhome/ChenXiaYu/workingdir/study/FFA_pattern/'\
        'analysis/s2/1080_fROI/refined_with_Kevin/bigbrain/'\
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


def smooth_msp(hemi):
    """
    对各msp map进行平滑
    """
    sigma = '2'
    n_layer = 6
    surf_file = '/nfs/z1/zhenlab/stanford/ABA/brainmap/bigbrain/'\
        'BigBrainRelease.2015/Surface_Parcellations/'\
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


def mask_msp(hemi):
    """
    把各层平均map在PC1中属于nan的部分的值设置为nan
    """
    Hemi = hemi2Hemi[hemi]
    n_layer = 6
    resample_way = '164fsLR2bigbrain'
    msp_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_s2.func.gii')
    pc_file = pjoin(anal_dir,
                    'decomposition/HCPY-M+corrT_MMP-vis3-'
                    f'{Hemi}_zscore1_PCA-subj_{resample_way}.func.gii')
    out_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_s2_mask.func.gii')

    pc_map = nib.load(pc_file).darrays[0].data
    nan_vec = np.isnan(pc_map)
    for layer_idx in range(n_layer):
        msp_gii = nib.load(msp_file.format(layer_idx, layer_idx+1, hemi))
        msp_gii.darrays[0].data[nan_vec] = np.nan
        nib.save(msp_gii, out_file.format(layer_idx, layer_idx+1, hemi))


def PC12_corr_msp(hemi, resample_way):
    """
    C1_corr_layer: PC1和各层10个细分map的相关
    C2_corr_layer: PC2和各层10个细分map的相关
    C1_corr_layer-mean: PC1和各层平均map的相关
    C2_corr_layer-mean: PC2和各层平均map的相关
    """
    Hemi = hemi2Hemi[hemi]
    n_layer = 6
    n_interbedded = 10
    msp_file1 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}_{2}_s2.func.gii')
    msp_file2 = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_s2.func.gii')
    pc_file = pjoin(anal_dir,
                    'decomposition/HCPY-M+corrT_MMP-vis3-'
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


def msp_lasso_PC12(hemi='rh'):
    """
    用6层胞体密度map作为特征分别对结构梯度的PC1/2做lasso线性拟合
    对比不同alpha的结果 (比较各特征系数的大小
    """
    Hemi = hemi2Hemi[hemi]
    msp_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}-mean_{2}_s2.func.gii')
    n_layer = 6
    resample_way = '164fsLR2bigbrain'
    pc_file = pjoin(anal_dir,
                    'decomposition/HCPY-M+T_MMP-vis3-'
                    f'{Hemi}_zscore1_PCA-subj_{resample_way}.func.gii')
    pc_names = ('C1', 'C2')
    n_pc = len(pc_names)
    # alphas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    # out_file = pjoin(work_dir, f'Msp1~6-s2_lasso_PC12-{resample_way}_{Hemi}.pkl')
    alphas = [0.625, 0.6875, 0.75, 0.78125, 0.8125, 0.875]
    out_file = pjoin(work_dir, f'Msp1~6-s2_lasso_PC12-{resample_way}_{Hemi}_fine1.pkl')

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

    # fit
    out_dict = {
        'PC name': pc_names, 'layer ID': list(range(1, n_layer+1)),
        'Y': Y, 'alpha': alphas}
    for alpha in alphas:
        model = Pipeline([
            ('preprocesser', StandardScaler()),
            ('regressor', Lasso(alpha=alpha, selection='cyclic'))
        ])
        model.fit(X, Y)
        Y_pred = model.predict(X)
        scores = [r2_score(Y[:, i], Y_pred[:, i]) for i in range(n_pc)]
        out_dict[f'alpha{alpha}'] = {
            'Y_pred': Y_pred, 'model': model, 'R2': scores,
            'coef': model.named_steps['regressor'].coef_
        }

    # save out
    pkl.dump(out_dict, open(out_file, 'wb'))


# ===to 32k_fs_LR===
def resample_msp_bigbrain_to_164fsLR(hemi):
    n_layer = 6
    Hemi = hemi2Hemi[hemi]
    msp_file = pjoin(work_dir, 'Msp_BB_layer{0}-{1}_{2}.func.gii')
    surf_dir = '/nfs/z1/zhenlab/stanford/ABA/brainmap/bigbrain/'\
        'BigBrainRelease.2015/Surface_Parcellations'
    out_file = pjoin(work_dir, 'to_32fsLR/'
                     'Msp_BB_layer{0}-{1}_{2}_164fsLR.func.gii')
    for lyr_idx in range(n_layer):
        cmd = ['wb_command', '-metric-resample']
        cmd.append(msp_file.format(lyr_idx, lyr_idx+1, hemi))
        cmd.append(pjoin(
            surf_dir, f'BigBrain_space/Surfaces/{hemi}.sphere.anat.surf.gii'))
        cmd.append(pjoin(
            surf_dir, f'fs_LR/Surfaces/{hemi}.sphere.surf.gii'))
        cmd.append('BARYCENTRIC')
        cmd.append(out_file.format(lyr_idx, lyr_idx+1, Hemi))
        print('Running: ' + ' '.join(cmd) + '\n')
        subprocess.run(cmd)


def resample_msp_164_to_32fsLR(Hemi):
    n_layer = 6
    msp_file = pjoin(work_dir, 'to_32fsLR/'
                     'Msp_BB_layer{0}-{1}_{2}_164fsLR.func.gii')
    surf_dir = '/usr/local/neurosoft/HCPpipelines/global/'\
        'templates/standard_mesh_atlases'
    out_file = pjoin(work_dir, 'to_32fsLR/'
                     'Msp_BB_layer{0}-{1}_{2}_32fsLR.func.gii')
    for lyr_idx in range(n_layer):
        cmd = ['wb_command', '-metric-resample']
        cmd.append(msp_file.format(lyr_idx, lyr_idx+1, Hemi))
        cmd.append(pjoin(
            surf_dir, f'fsaverage.{Hemi}_LR.spherical_std.164k_fs_LR.surf.gii'))
        cmd.append(pjoin(
            surf_dir, f'{Hemi}.sphere.32k_fs_LR.surf.gii'))
        cmd.append('ADAP_BARY_AREA')
        cmd.append(out_file.format(lyr_idx, lyr_idx+1, Hemi))
        cmd.append('-area-metrics')
        cmd.append(pjoin(
            surf_dir, 'resample_fsaverage/'
            f'fs_LR.{Hemi}.midthickness_va_avg.164k_fs_LR.shape.gii'))
        cmd.append(pjoin(
            surf_dir, 'resample_fsaverage/'
            f'fs_LR.{Hemi}.midthickness_va_avg.32k_fs_LR.shape.gii'))
        print('Running: ' + ' '.join(cmd) + '\n')
        subprocess.run(cmd)


def average_msp(Hemi):
    """
    并计算每个layer的平均map（之前对每个layer采样了10个surface）
    """
    n_layer = 6
    msp_file = pjoin(work_dir, 'to_32fsLR/'
                     'Msp_BB_layer{0}-{1}_{2}_32fsLR.func.gii')
    out_file = pjoin(work_dir, 'to_32fsLR/'
                     'Msp_BB_layer{0}-{1}-mean_{2}_32fsLR.func.gii')

    for lyr_idx in range(n_layer):
        gii1 = nib.load(msp_file.format(lyr_idx, lyr_idx+1, Hemi))
        data = []
        for i in gii1.darrays:
            data.append(i.data)
        print(len(data))
        gii2 = GiftiImage(darrays=[GiftiDataArray(np.mean(data, 0))])
        nib.save(gii2, out_file.format(lyr_idx, lyr_idx+1, Hemi))


def PC12_corr_msp_32fsLR(Hemi):
    """
    C1_corr_layer: PC1和各层10个细分map的相关
    C2_corr_layer: PC2和各层10个细分map的相关
    C1_corr_layer-mean: PC1和各层平均map的相关
    C2_corr_layer-mean: PC2和各层平均map的相关
    """
    n_layer = 6
    n_interbedded = 10
    msp_file1 = pjoin(work_dir, 'to_32fsLR/'
                      'Msp_BB_layer{0}-{1}_{2}_32fsLR.func.gii')
    msp_file2 = pjoin(work_dir, 'to_32fsLR/'
                      'Msp_BB_layer{0}-{1}-mean_{2}_32fsLR.func.gii')
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        f'HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj_nan.func.gii')
    pc_names = ('C1', 'C2')
    out_file = pjoin(work_dir, f'to_32fsLR/PC12-corr-msp_{Hemi}.pkl')

    def statistic(x, y):
        return pearsonr(x, y)[0]

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
        data2_pmt_p = np.zeros(n_layer, np.float64)
        for layer_idx in range(n_layer):
            msp_gii1 = nib.load(msp_file1.format(layer_idx, layer_idx+1, Hemi))
            for i in range(n_interbedded):
                msp_map = msp_gii1.darrays[i].data[non_nan_vec]
                r, p = pearsonr(pc_map, msp_map)
                data1[i, layer_idx] = r
                data1_p[i, layer_idx] = p

            msp_gii2 = nib.load(msp_file2.format(layer_idx, layer_idx+1, Hemi))
            msp_map = msp_gii2.darrays[0].data[non_nan_vec]
            r, p = pearsonr(pc_map, msp_map)
            pmt_test = permutation_test(
                (pc_map, msp_map), statistic, permutation_type='pairings',
                vectorized=False, n_resamples=10000, alternative='two-sided',
                random_state=7)
            data2[layer_idx] = r
            data2_p[layer_idx] = p
            data2_pmt_p[layer_idx] = pmt_test.pvalue
        out_dict[f'{pc_name}_corr_layer'] = data1
        out_dict[f'{pc_name}_corr_layer-mean'] = data2
        out_dict[f'{pc_name}_corr_layer-p'] = data1_p
        out_dict[f'{pc_name}_corr_layer-mean-p'] = data2_p
        out_dict[f'{pc_name}_corr_layer-mean-pmt-p'] = data2_pmt_p

    pkl.dump(out_dict, open(out_file, 'wb'))


def msp_fit_PC12_32fsLR(Hemi='R', method='ordinary'):
    """
    用6层胞体密度map作为特征分别对结构梯度的PC1/2做线性拟合
    比较各特征系数的大小
    """
    n_layer = 6
    msp_file = pjoin(work_dir, 'to_32fsLR/'
                     'Msp_BB_layer{0}-{1}-mean_{2}_32fsLR.func.gii')
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        f'HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj_nan.func.gii')
    pc_names = ('C1', 'C2')
    n_pc = len(pc_names)
    out_file = pjoin(work_dir, 'to_32fsLR/'
                     f'Msp1~6_fit-{method}_PC12_{Hemi}.pkl')

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
        msp_gii = nib.load(msp_file.format(lyr_idx, lyr_idx+1, Hemi))
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
    # get_msp_from_FFA_proj(hemi='lh')
    # get_msp_from_FFA_proj(hemi='rh')
    # smooth_msp(hemi='lh')
    # smooth_msp(hemi='rh')
    # mask_msp(hemi='lh')
    # mask_msp(hemi='rh')
    # PC12_corr_msp(hemi='lh', resample_way='164fsLR2bigbrain')
    # PC12_corr_msp(hemi='rh', resample_way='164fsLR2bigbrain')
    # PC12_corr_msp(hemi='lh', resample_way='fsavg2bigbrain')
    # PC12_corr_msp(hemi='rh', resample_way='fsavg2bigbrain')
    # msp_fit_PC12(hemi='rh', method='ordinary')
    # msp_fit_PC12(hemi='rh', method='lasso')
    # msp_lasso_PC12(hemi='rh')

    # resample_msp_bigbrain_to_164fsLR(hemi='lh')
    # resample_msp_bigbrain_to_164fsLR(hemi='rh')
    # resample_msp_164_to_32fsLR(Hemi='L')
    # resample_msp_164_to_32fsLR(Hemi='R')
    # average_msp(Hemi='L')
    # average_msp(Hemi='R')
    # PC12_corr_msp_32fsLR(Hemi='L')
    # PC12_corr_msp_32fsLR(Hemi='R')
    msp_fit_PC12_32fsLR(Hemi='R', method='ordinary')
