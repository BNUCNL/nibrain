import os
import time
import math
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from cxy_visual_dev.lib.predefine import All_count_32k, LR_count_32k, proj_dir, Atlas,\
    get_rois, hemi2Hemi, mmp_map_file, s1200_avg_RFsize
from cxy_visual_dev.lib.algo import cat_data_from_cifti,\
    linear_fit1
from magicbox.io.io import CiftiReader, save2cifti

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'fit')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def old_fit():
    """
    分别用枕极和MT做锚点，和距状沟锚点一起描述C1
    这里用这三个map作为三个特征去线性拟合C1（左右脑拼起来之前分别做zscore）

    用HCPD每个被试的myelin和thickness map去拟合C1C2
    这里做拼接左右脑数据的时候会分别做zscore
    """
    mask = Atlas('HCP-MMP').get_mask(
        get_rois('MMP-vis2-L') + get_rois('MMP-vis2-R'))[0]
    mask_L = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis2-L'))[0]
    mask_R = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis2-R'))[0]
    C1C2_maps = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii'
    )).get_fdata()[:2]

    src_files = [
            pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-MT.dscalar.nii')]
    X_list = []
    for src_file in src_files:
        data = cat_data_from_cifti([src_file], (1, 1),
                                   [mask_L, mask_R], zscore1='split')[0]
        X_list.append(data.T)
    Y = C1C2_maps[0, mask][:, None]
    linear_fit1(
        X_list=X_list, feat_names=['CalcarineSulcus', 'OccipitalPole', 'MT'],
        Y=Y, trg_names=['C1'], score_metric='R2',
        out_file=pjoin(work_dir, 'CalcS+OcPole+MT=C1.csv'),
        standard_scale=False
    )

    src_files = [
            pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
            pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')]
    X_list = []
    for src_file in src_files:
        data = cat_data_from_cifti([src_file], (1, 1),
                                   [mask_L, mask_R], zscore1='split')[0]
        X_list.append(data.T)
    Y = C1C2_maps[:, mask].T
    linear_fit1(
        X_list=X_list, feat_names=['Myelination', 'Thickness'],
        Y=Y, trg_names=['C1', 'C2'], score_metric='R2',
        out_file=pjoin(work_dir, 'HCPD-M+T=C1C2.csv'),
        standard_scale=False
    )


def gdist_fit_PC1():
    """
    分别用枕极和MT做锚点，和距状沟锚点一起描述C1
    这里用这三个map作为三个特征去线性拟合C1 (半脑)
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    src_files = [
            # pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-MT.dscalar.nii')]
    feat_names = ['OccipitalPole', 'MT']
    out_file = pjoin(work_dir, 'OcPole+MT=C1.csv')

    C1_map = nib.load(pc_file).get_fdata()[0]
    X_list = []
    for src_file in src_files:
        data = nib.load(src_file).get_fdata()[:, mask]
        X_list.append(data.T)
    Y = np.expand_dims(C1_map[mask], 1)
    linear_fit1(
        X_list=X_list, feat_names=feat_names,
        Y=Y, trg_names=['C1'], score_metric='R2',
        out_file=out_file, standard_scale=True)


def HCPDA_fit_PC12():
    """
    用HCPD/A每个被试的myelin和thickness map去拟合C1C2（半脑）
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    src_files = [
            pjoin(proj_dir, 'data/HCP/HCPA_myelin.dscalar.nii'),
            pjoin(proj_dir, 'data/HCP/HCPA_thickness.dscalar.nii')]
    feat_names = ['Myelination', 'Thickness']
    out_file = pjoin(work_dir, 'HCPA-M+T=C1C2.csv')

    C1C2_maps = nib.load(pc_file).get_fdata()[:2]
    X_list = []
    for src_file in src_files:
        data = nib.load(src_file).get_fdata()[:, mask]
        X_list.append(data.T)
    Y = C1C2_maps[:, mask].T
    linear_fit1(
        X_list=X_list, feat_names=feat_names,
        Y=Y, trg_names=['C1', 'C2'], score_metric='R2',
        out_file=out_file, standard_scale=True)


def mean_tau_diff_fit_PC12():
    """
    用某岁的平均map+tau+diff预测成人PC map
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    src_files = [
        pjoin(anal_dir, 'summary_map/HCPD-myelin_age-map-mean.dscalar.nii'),
        pjoin(anal_dir, 'summary_map/HCPD-thickness_age-map-mean.dscalar.nii'),
        # pjoin(anal_dir, 'dev_trend/HCPD-myelin_MMP-vis3_kendall.dscalar.nii'),
        # pjoin(anal_dir, 'dev_trend/HCPD-thickness_MMP-vis3_kendall.dscalar.nii'),
        # pjoin(anal_dir, 'dev_trend/HCPD-myelin_age-map-mean_21-8.dscalar.nii'),
        # pjoin(anal_dir, 'dev_trend/HCPD-thickness_age-map-mean_21-8.dscalar.nii')
    ]
    # feat_names = ['myelin-8', 'thickness-8', 'myelin-tau', 'thickness-tau',
    #               'myelin-diff', 'thickness-diff']
    # map_indices = [3, 3, 0, 0, 0, 0]
    # out_file = pjoin(work_dir, 'Mean8+Tau+Diff=C1C2.csv')
    feat_names = ['myelin-8', 'thickness-8']
    map_indices = [3, 3]
    out_file = pjoin(work_dir, 'Mean8=C1C2.csv')

    C1C2_maps = nib.load(pc_file).get_fdata()[:2]
    X_list = []
    for i, src_file in enumerate(src_files):
        data = nib.load(src_file).get_fdata()[map_indices[i], mask]
        X_list.append(np.expand_dims(data, 1))
    Y = C1C2_maps[:, mask].T
    linear_fit1(
        X_list=X_list, feat_names=feat_names,
        Y=Y, trg_names=['C1', 'C2'], score_metric='R2',
        out_file=out_file, standard_scale=True)


def HCPDA_fit_PC12_local():
    """
    用HCPD/A每个被试的myelin和thickness map去局部拟合C1C2（半脑）
    """
    data_name = 'HCPD'
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    mask_local_file = pjoin(anal_dir, f'mask_map/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj_N3.dlabel.nii')
    mask_local_lbls = np.arange(1, 4)
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')

    src_files = [
            pjoin(proj_dir, f'data/HCP/{data_name}_myelin.dscalar.nii'),
            pjoin(proj_dir, f'data/HCP/{data_name}_thickness.dscalar.nii')]
    feat_names = ['Myelination', 'Thickness']
    out_file = pjoin(work_dir, f'{data_name}-M+T=C1C2_local-mask1-N3-{Hemi}.csv')

    mask_locals = nib.load(mask_local_file).get_fdata()[:, mask]
    pc_names = ['C1', 'C2']
    n_pc = len(pc_names)
    reader = CiftiReader(pc_file)
    pc_maps = reader.get_data()[:n_pc, mask].T
    assert pc_names == reader.map_names()[:n_pc]

    feat_maps_list = []
    for src_file in src_files:
        data = nib.load(src_file).get_fdata()[:, mask]
        feat_maps_list.append(data.T)

    dfs = []
    for pc_idx in range(n_pc):
        for mask_local_lbl in mask_local_lbls:
            mask_local = mask_locals[pc_idx] == mask_local_lbl
            X_list = [i[mask_local] for i in feat_maps_list]
            Y = np.expand_dims(pc_maps[mask_local, pc_idx], 1)
            df = linear_fit1(
                X_list=X_list, feat_names=feat_names,
                Y=Y, trg_names=[f'{pc_names[pc_idx]}-{mask_local_lbl}'], score_metric='R2',
                out_file='df', standard_scale=True)
            dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df.to_csv(out_file, index=False)


def HCPDA_fit_PC12_local1(data_name='HCPD', Hemi='R'):
    """
    用HCPD/A每个被试的myelin和thickness map去局部拟合C1C2（半脑）
    """
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')

    src_files = [
            pjoin(proj_dir, f'data/HCP/{data_name}_myelin.dscalar.nii'),
            pjoin(proj_dir, f'data/HCP/{data_name}_thickness.dscalar.nii')]
    feat_names = ['Myelination', 'Thickness']

    mask_local_file = pjoin(anal_dir, 'tmp/MMP-vis3-EDMV.dlabel.nii')
    mask_local_lbls = np.arange(1, 5)
    out_file = pjoin(work_dir, f'{data_name}-M+T=C1C2_MMP-vis3-{Hemi}-EDMV.csv')

    mask_local_map = nib.load(mask_local_file).get_fdata()[0, mask]
    pc_names = ['C1', 'C2']
    n_pc = len(pc_names)
    reader = CiftiReader(pc_file)
    pc_maps = reader.get_data()[:n_pc, mask].T
    assert pc_names == reader.map_names()[:n_pc]

    feat_maps_list = []
    for src_file in src_files:
        data = nib.load(src_file).get_fdata()[:, mask]
        feat_maps_list.append(data.T)

    dfs = []
    for mask_local_lbl in mask_local_lbls:
        mask_local = mask_local_map == mask_local_lbl
        for pc_idx in range(n_pc):
            X_list = [i[mask_local] for i in feat_maps_list]
            Y = np.expand_dims(pc_maps[mask_local, pc_idx], 1)
            df = linear_fit1(
                X_list=X_list, feat_names=feat_names,
                Y=Y, trg_names=[f'{pc_names[pc_idx]}-{mask_local_lbl}'], score_metric='R2',
                out_file='df', standard_scale=True)
            dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df.to_csv(out_file, index=False)


def age_linearFit_col():
    """
    用年龄线性拟合csv文件中的各列
    """
    # data_name = 'HCPD'
    # age_name = 'age in years'
    # src_file = pjoin(work_dir, f'{data_name}-M+T=C1C2_local-mask1-N3-R.csv')
    # info_file = pjoin(proj_dir, f'data/HCP/{data_name}_SubjInfo.csv')
    # out_file = pjoin(work_dir, f'{data_name}-M+T=C1C2_local-mask1-N3-R_AgeFitCol1.csv')

    data_name = 'HCPD'
    age_name = 'age in years'
    src_file = pjoin(anal_dir, f'ROI_scalar/{data_name}-thickness_N3-C1.csv')
    info_file = pjoin(proj_dir, f'data/HCP/{data_name}_SubjInfo.csv')
    out_file = pjoin(work_dir, f'{data_name}-thickness_N3-C1_AgeFitCol1.csv')

    df = pd.read_csv(src_file)
    info_df = pd.read_csv(info_file)
    ages = np.array(info_df[age_name])
    if data_name == 'HCPD':
        print('remove 5/6/7')
        idx_vec = np.zeros_like(ages, bool)
        for i in (5, 6, 7):
            idx_vec = np.logical_or(ages == i, idx_vec)
        idx_vec = ~idx_vec
        ages = ages[idx_vec]
        df = df.loc[idx_vec]

    X_list = [np.expand_dims(ages, 1)]
    out_df = []
    for col in df.columns:
        Y = np.expand_dims(np.array(df[col]), 1)
        df_tmp = linear_fit1(
            X_list=X_list, feat_names=[''], Y=Y, trg_names=[''],
            score_metric='R2', out_file='df', standard_scale=False)
        out_df.append(df_tmp)

    out_df = pd.concat(out_df, axis=0)
    print(out_df.columns)
    out_df.columns = [i.rstrip('_') for i in out_df.columns]
    out_df.index = df.columns
    out_df.to_csv(out_file, index=True)


def PC12_fit_func():
    """
    用HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj的PC1和PC2
    线性拟合功能map
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    feat_names = ['C1', 'C2']
    func_file = s1200_avg_RFsize
    trg_name = 'RFsize'
    fname = f'PC1+2={trg_name}'
    out_file1 = pjoin(work_dir, f'{fname}.csv')
    out_file2 = pjoin(work_dir, f'{fname}.pkl')
    out_file3 = pjoin(work_dir, f'{fname}.dscalar.nii')

    # Hemi = 'R'
    # lbl = 2
    # mask = nib.load(pjoin(
    #     anal_dir, f'mask_map/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj_N2.dlabel.nii'
    # )).get_fdata()[0] == lbl
    # pc_file = pjoin(
    #     anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    # feat_names = ['C1', 'C2']
    # func_file = pjoin(anal_dir, 'summary_map/HCPY-face_mean.dscalar.nii')
    # trg_name = 'face'
    # fname = f'PC1+2={trg_name}_mask-PC1-N2-{lbl}'
    # out_file1 = pjoin(work_dir, f'{fname}.csv')
    # out_file2 = pjoin(work_dir, f'{fname}.pkl')
    # out_file3 = pjoin(work_dir, f'{fname}.dscalar.nii')

    func_data = nib.load(func_file).get_fdata()[0]
    if func_data.shape[0] == All_count_32k:
        func_data = func_data[:LR_count_32k]
    elif func_data.shape[0] == LR_count_32k:
        pass
    else:
        raise ValueError
    X = nib.load(pc_file).get_fdata()[:2, mask].T
    y = func_data[mask]
    model = Pipeline([('preprocesser', StandardScaler()),
                      ('regressor', LinearRegression())])
    model.fit(X, y)
    y_pred = model.predict(X)
    print('true corr pred:', pearsonr(y, y_pred))
    print(f'PC1 corr {trg_name}:', pearsonr(X[:, 0], y))
    print(f'PC2 corr {trg_name}:', pearsonr(X[:, 1], y))

    # save parameters
    df = pd.DataFrame()
    for feat_idx, feat_name in enumerate(feat_names):
        df[f'coef_{trg_name}_{feat_name}'] = \
            [model.named_steps['regressor'].coef_[feat_idx]]
    df[f'score_{trg_name}'] = [r2_score(y, y_pred)]
    df[f'intercept_{trg_name}'] = \
        [model.named_steps['regressor'].intercept_]
    df.to_csv(out_file1, index=False)

    # save model
    pkl.dump(model, open(out_file2, 'wb'))

    # save fitted map
    reader = CiftiReader(mmp_map_file)
    out_map = np.ones((1, mask.shape[0]), np.float64) * np.nan
    out_map[0, mask] = y_pred
    save2cifti(out_file3, out_map, reader.brain_models())


def PC12_fit_func1():
    """
    用HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj的PC1和PC2
    线性拟合功能map（逐被试）
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    feat_names = ['C1', 'C2']
    func_file = pjoin(proj_dir, 'data/HCP/HCPY-falff.dscalar.nii')
    out_file = pjoin(work_dir, 'PC1+2=fALFF_ind.csv')

    X = nib.load(pc_file).get_fdata()[:2, mask].T
    print(X.shape)
    func_maps = nib.load(func_file).get_fdata()[:, mask]
    nan_arr = np.isnan(func_maps)
    nan_vec = np.any(nan_arr, 1)
    assert np.all(nan_vec == np.all(nan_arr, 1))
    non_nan_vec = ~nan_vec
    Y = func_maps[non_nan_vec].T
    n_trg = Y.shape[1]
    print(Y.shape)

    model = Pipeline([('preprocesser', StandardScaler()),
                      ('regressor', LinearRegression())])
    model.fit(X, Y)
    coefs = model.named_steps['regressor'].coef_
    coef_names = [f'coef_{i}' for i in feat_names]
    intercepts = np.expand_dims(model.named_steps['regressor'].intercept_, 1)
    Y_pred = model.predict(X)
    scores = np.zeros((n_trg, 1), np.float64)
    for trg_idx in range(n_trg):
        scores[trg_idx, 0] = r2_score(Y[:, trg_idx], Y_pred[:, trg_idx])
    out_data = np.c_[coefs, intercepts, scores]

    out_df = pd.DataFrame(out_data, columns=coef_names+['intercept', 'score'])
    out_df.to_csv(out_file, index=False)


def PC12_fit_func2():
    """
    用HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj的PC1和PC2
    的线性组合，和功能map做指数拟合
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    func_file = pjoin(anal_dir, 'summary_map/HCPY-falff_mean.dscalar.nii')
    out_file = pjoin(work_dir, 'PC1+2=log(fALFF).dscalar.nii')
    out_fig = pjoin(work_dir, 'PC1+2=log(fALFF).jpg')

    X = nib.load(pc_file).get_fdata()[:2, mask]
    y = nib.load(func_file).get_fdata()[0, mask]
    y_min, y_max = np.min(y), np.max(y)
    # def f(x, n, a1, a2, b, v1, v2):
    #     return (np.power(n, a1*x[0]+a2*x[1]+b) - v1) / (v2 - v1) * (y_max - y_min) + y_min
    # def f(x, n, a1, a2, b, k, c):
    #     return (k*np.power(n, a1*x[0]+a2*x[1]+b) + c) * (y_max - y_min) + y_min
    # def f(x, n, a1, a2, b, k, c):
    #     tmp = a1*x[0] + a2*x[1] + b
    #     tmp = tmp - np.min(tmp) + 2
    #     tmp = np.array([math.log(i, n) for i in tmp])
    #     return k * tmp + c
    # def f(x, a1, a2, b1, b2, k):
    #     return k * np.power(10, a1*x[0]+a2*x[1]+b1) + b2
    # def f(x, a1, a2, b1, b2, c):
    #     return b1*np.power(10, a1*x[0]) + b2*np.power(10, a2*x[1]) + c

    # def f(x, n, a1, a2, b):
    #     return np.power(n, a1*x[0]+a2*x[1]+b)
    # bounds = (-np.inf, np.inf)
    # bounds = ([1.5] + [-np.inf]*5, np.inf)
    bounds = (
        [1, -0.02, -0.02, -np.inf, -np.inf],
        [np.inf, 0.02, 0.02, np.inf, np.inf])
    def f(x, n, a1, a2, k, c):
        return k * np.power(n, a1*x[0]+a2*x[1]) + c
    time1 = time.time()
    popt, pcov = curve_fit(f, X, y, maxfev=1000000, ftol=1e-16, bounds=bounds)
    print(time.time() - time1, 'seconds')
    print(popt)
    y_pred = f(X, *popt)
    print('true corr pred:', pearsonr(y, y_pred))

    # save fitted map
    reader = CiftiReader(mmp_map_file)
    out_map = np.ones((1, mask.shape[0]), np.float64) * np.nan
    out_map[0, mask] = y_pred
    save2cifti(out_file, out_map, reader.brain_models())

    n, a1, a2, k, c = popt
    x_plot = (y - c) / k
    y_plot = a1*X[0] + a2*X[1]
    x_plot_min, x_plot_max = np.min(x_plot), np.max(x_plot)
    x_plot_tmp = np.linspace(x_plot_min, x_plot_max, 100)
    y_plot_tmp = [math.log(i, n) for i in x_plot_tmp]
    fig, ax = plt.subplots()
    ax.scatter(x_plot, y_plot, 1)
    ax.plot(x_plot_tmp, y_plot_tmp)
    ax.set_xlabel('True fALFF')
    ax.set_ylabel('Pred fALFF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_fig)


if __name__ == '__main__':
    # gdist_fit_PC1()
    # HCPDA_fit_PC12()
    # mean_tau_diff_fit_PC12()
    # HCPDA_fit_PC12_local()
    # HCPDA_fit_PC12_local1(data_name='HCPD', Hemi='R')
    HCPDA_fit_PC12_local1(data_name='HCPA', Hemi='R')
    # age_linearFit_col()
    # PC12_fit_func()
    # PC12_fit_func1()
    # PC12_fit_func2()
