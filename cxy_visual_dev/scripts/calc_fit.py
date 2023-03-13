import os
import time
import math
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr, permutation_test
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import CCA
from matplotlib import pyplot as plt
from cxy_visual_dev.lib.predefine import All_count_32k, LR_count_32k, proj_dir, Atlas,\
    get_rois, hemi2Hemi, mmp_map_file, s1200_avg_RFsize, s1200_1096_myelin, s1200_1096_thickness,\
    hemi2stru
from cxy_visual_dev.lib.algo import cat_data_from_cifti,\
    linear_fit1, AgeSlideWindow
from magicbox.io.io import CiftiReader, save2cifti
from magicbox.stats.outlier import outlier_iqr

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


def gdist_fit_PC12():
    """
    用gdist map拟合PC1, PC2
    """
    Hemi = 'R'
    mask = Atlas('HCP-MMP').get_mask(get_rois(f'MMP-vis3-{Hemi}'))[0]
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+T_MMP-vis3-{Hemi}_zscore1_PCA-subj.dscalar.nii')
    src_files = [
            pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-OpMt.dscalar.nii')]
    feat_names = ['CS', 'OpMt']
    out_file = pjoin(work_dir, 'CS+OpMt=C1C2.csv')

    X_list = []
    for src_file in src_files:
        data = nib.load(src_file).get_fdata()[:, mask]
        X_list.append(data.T)
    Y = nib.load(pc_file).get_fdata()[:2, mask].T
    linear_fit1(
        X_list=X_list, feat_names=feat_names,
        Y=Y, trg_names=['C1', 'C2'], score_metric='R2',
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


def HCPDA_MT_fit_PC12_SW(dataset_name, vis_name, width, step, merge_remainder):
    """
    用每个滑窗(slide window)内的所有被试的myelin和thickness map去拟合stru-PC1/2
    """
    n_pc = 2
    m_file = pjoin(proj_dir, f'data/HCP/{dataset_name}_myelin.dscalar.nii')
    t_file = pjoin(proj_dir, f'data/HCP/{dataset_name}_corrThickness.dscalar.nii')
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    asw = AgeSlideWindow(dataset_name, width, step, merge_remainder)
    out_name = f'{dataset_name}-M+corrT=PC12_{vis_name}_SW-width{width}-step{step}'
    if merge_remainder:
        out_name += '-merge'
    out_file = pjoin(work_dir, f'{out_name}.pkl')

    m_maps = nib.load(m_file).get_fdata()[:, mask]
    t_maps = nib.load(t_file).get_fdata()[:, mask]
    Y = nib.load(pc_file).get_fdata()[:n_pc, mask].T
    out_dict = {
        'score_C1': np.zeros(asw.n_win), 'score_C2': np.zeros(asw.n_win),
        'intercept_C1': np.zeros(asw.n_win), 'intercept_C2': np.zeros(asw.n_win),
        'age in months': np.zeros(asw.n_win)
    }
    for win_idx in range(asw.n_win):
        time1 = time.time()
        win_id = win_idx + 1
        indices = asw.get_subj_indices(win_id)
        n_idx = len(indices)
        X = np.r_[m_maps[indices], t_maps[indices]].T
        model = Pipeline([('preprocesser', StandardScaler()),
                          ('regressor', LinearRegression())])
        model.fit(X, Y)
        Y_pred = model.predict(X)
        for pc_idx in range(n_pc):
            out_dict[f'coef_C{pc_idx+1}_Myelination_win{win_id}'] = \
                model.named_steps['regressor'].coef_[pc_idx, :n_idx]
            out_dict[f'coef_C{pc_idx+1}_Thickness_win{win_id}'] = \
                model.named_steps['regressor'].coef_[pc_idx, n_idx:]
            out_dict[f'intercept_C{pc_idx+1}'][win_idx] = \
                model.named_steps['regressor'].intercept_[pc_idx]
            out_dict[f'score_C{pc_idx+1}'][win_idx] = \
                r2_score(Y[:, pc_idx], Y_pred[:, pc_idx])
        out_dict['age in months'][win_idx] = asw.get_ages(win_id, 'month').mean()
        print(f'Finished Win-{win_id}/{asw.n_win}, cost: {time.time() - time1} seconds.')

    pkl.dump(out_dict, open(out_file, 'wb'))


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


def PC12_fit_func3(Hemi):
    """
    用HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj的PC1和PC2
    在整个视觉皮层以及EDLV四个部分线性拟合WM任务激活+fALFF
    """
    vis_name = f'MMP-vis3-{Hemi}'
    mask_names = (vis_name, f'{vis_name}-early', f'{vis_name}-dorsal',
                  f'{vis_name}-lateral', f'{vis_name}-ventral')
    atlas = Atlas('MMP-vis3-EDLV')
    pc_file = pjoin(
        anal_dir, f'decomposition/HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    pc_names = ['PC1', 'PC2']
    func_file = pjoin(anal_dir, 'tfMRI/tfMRI-WM-cope.dscalar.nii')
    faff_file = pjoin(anal_dir, 'AFF/HCPY-faff.dscalar.nii')

    n_pc = len(pc_names)
    pc_maps = nib.load(pc_file).get_fdata()[:n_pc]
    reader = CiftiReader(func_file)
    bms = reader.brain_models([hemi2stru['lh'], hemi2stru['rh']])
    func_data = reader.get_data()[:, :LR_count_32k]
    trg_names = reader.map_names()
    falff_map = nib.load(faff_file).get_fdata()[[0], :LR_count_32k]
    func_data = np.r_[func_data, falff_map]
    trg_names.append('fALFF')
    n_trg = len(trg_names)
    n_mask = len(mask_names)

    fname1 = 'PC1+2=func3'
    out_file1 = pjoin(work_dir, f'{fname1}_{Hemi}.pkl')
    rs = np.zeros((n_mask, n_trg))
    ps = np.zeros((n_mask, n_trg))
    for mask_idx, mask_name in enumerate(mask_names):
        fname2 = f'{fname1}_{mask_name}'
        out_file2 = pjoin(work_dir, f'{fname2}.dscalar.nii')
        if mask_name == vis_name:
            mask = atlas.get_mask(Hemi)[0]
        else:
            edlv_name = f"{Hemi}_{mask_name.split('-')[-1]}"
            mask = atlas.get_mask(edlv_name)[0]
        X = pc_maps[:, mask].T
        out_maps = np.ones((n_trg, LR_count_32k)) * np.nan
        for trg_idx, trg_name in enumerate(trg_names):
            y = func_data[trg_idx, mask]
            model = Pipeline([('preprocesser', StandardScaler()),
                              ('regressor', LinearRegression())])
            model.fit(X, y)
            y_pred = model.predict(X)
            out_maps[trg_idx, mask] = y_pred
            r, p = pearsonr(y, y_pred)
            rs[mask_idx, trg_idx] = r
            ps[mask_idx, trg_idx] = p
        save2cifti(out_file2, out_maps, bms, trg_names)
    out_data = {'row_name': mask_names, 'col_name': trg_names,
                'r': rs, 'p': ps}
    pkl.dump(out_data, open(out_file1, 'wb'))


def PC12_fit_func4(Hemi):
    """
    用HCPY-M+corrT_MMP-vis3-{Hemi}_zscore1_PCA-subj的PC1和PC2
    在整个视觉皮层以及用分水岭算法得到的EDLV四个部分线性拟合WM任务激活+fALFF
    """
    vis_name = f'MMP-vis3-{Hemi}'
    mask_names = (vis_name, f'{vis_name}-early', f'{vis_name}-dorsal',
                  f'{vis_name}-lateral', f'{vis_name}-ventral')
    pc_file = pjoin(
        anal_dir, 'decomposition/'
        f'HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    pc_names = ['PC1', 'PC2']
    func_file = pjoin(anal_dir, 'tfMRI/tfMRI-WM-cope.dscalar.nii')
    faff_file = pjoin(anal_dir, 'AFF/HCPY-faff.dscalar.nii')
    edlv_file = pjoin(anal_dir, 'divide_map/'
                      'watershed-PC2_EDLV-seed-v1.dlabel.nii')

    def statistic(x, y):
        return r2_score(x, y)

    n_pc = len(pc_names)
    pc_maps = nib.load(pc_file).get_fdata()[:n_pc]
    reader = CiftiReader(func_file)
    bms = reader.brain_models([hemi2stru['lh'], hemi2stru['rh']])
    func_data = reader.get_data()[:, :LR_count_32k]
    trg_names = reader.map_names()
    falff_map = nib.load(faff_file).get_fdata()[[0], :LR_count_32k]
    func_data = np.r_[func_data, falff_map]
    trg_names.append('fALFF')
    n_trg = len(trg_names)
    n_mask = len(mask_names)
    reader_edlv = CiftiReader(edlv_file)
    edlv_map = reader_edlv.get_data()[0]
    lbl_tab = reader_edlv.label_tables()[0]
    edlv_label2key = {}
    for k, v in lbl_tab.items():
        edlv_label2key[v.label] = k

    fname1 = 'PC1+2=func4'
    out_file1 = pjoin(work_dir, f'{fname1}_{Hemi}.pkl')
    rs = np.zeros((n_mask, n_trg))
    ps = np.zeros((n_mask, n_trg))
    R2s = np.zeros((n_mask, n_trg))
    R2_ps = np.zeros((n_mask, n_trg))
    weights = np.zeros((n_mask, n_trg, n_pc))
    for mask_idx, mask_name in enumerate(mask_names):
        fname2 = f'{fname1}_{mask_name}'
        out_file2 = pjoin(work_dir, f'{fname2}.dscalar.nii')
        if mask_name == vis_name:
            mask = np.zeros_like(edlv_map, bool)
            for lbl, k in edlv_label2key.items():
                if lbl.startswith(f'{Hemi}_'):
                    mask = np.logical_or(mask, edlv_map == k)
        else:
            edlv_lbl = f"{Hemi}_{mask_name.split('-')[-1]}"
            mask = edlv_map == edlv_label2key[edlv_lbl]
        X = pc_maps[:, mask].T
        out_maps = np.ones((n_trg, LR_count_32k)) * np.nan
        for trg_idx, trg_name in enumerate(trg_names):
            time1 = time.time()
            y = func_data[trg_idx, mask]
            model = Pipeline([('preprocesser', StandardScaler()),
                              ('regressor', LinearRegression())])
            model.fit(X, y)
            y_pred = model.predict(X)
            out_maps[trg_idx, mask] = y_pred
            r, p = pearsonr(y, y_pred)
            rs[mask_idx, trg_idx] = r
            ps[mask_idx, trg_idx] = p
            pmt_test = permutation_test(
                (y, y_pred), statistic, permutation_type='pairings',
                vectorized=False, n_resamples=10000, alternative='greater',
                random_state=7)
            R2s[mask_idx, trg_idx] = pmt_test.statistic
            R2_ps[mask_idx, trg_idx] = pmt_test.pvalue
            assert model.named_steps['regressor'].coef_.shape == (n_pc,)
            weights[mask_idx, trg_idx] = \
                model.named_steps['regressor'].coef_
            print('pmt_test.statistic:', pmt_test.statistic)
            print('model.score:', model.score(X, y))
            print(f'Finished {mask_name}-{mask_idx+1}/{n_mask} '
                  f'{trg_name}-{trg_idx+1}/{n_trg}: '
                  f'cost {time.time()-time1} seconds.')
        save2cifti(out_file2, out_maps, bms, trg_names)
    out_data = {'row_name': mask_names, 'col_name': trg_names,
                'r': rs, 'p': ps, 'R2': R2s, 'R2_p': R2_ps,
                'weight': weights}
    pkl.dump(out_data, open(out_file1, 'wb'))


def HCPY_MT_fit_PC12(Hemi):
    """
    对每个被试用其myelin和thickness map拟合PC1/2
    整体和局部的拟合都做，用以得到在整体或是局部视觉皮层中
    每个被试对PC1/2的贡献
    """
    vis_name = f'MMP-vis3-{Hemi}'
    out_file = pjoin(work_dir, f'HCPY-M+corrT_{vis_name}_fit_PC_subj-wise.pkl')

    # preparation for feature
    m_file = pjoin(proj_dir, 'data/HCP/HCPY_myelin.dscalar.nii')
    t_file = pjoin(proj_dir, 'data/HCP/HCPY_corrThickness.dscalar.nii')
    m_maps = nib.load(m_file).get_fdata()
    t_maps = nib.load(t_file).get_fdata()
    n_subj = m_maps.shape[0]

    # preparation for target
    n_pc = 2
    pc_names = ['C1', 'C2']
    pc_file = pjoin(anal_dir, f'decomposition/HCPY-M+corrT_{vis_name}_zscore1_PCA-subj.dscalar.nii')
    pc_maps = nib.load(pc_file).get_fdata()[:n_pc]

    # preparation for mask
    out_data = {}
    atlas_names = ['HCP-MMP', 'MMP-vis3-EDLV']
    for atlas_name in atlas_names:
        atlas = Atlas(atlas_name)
        if atlas_name == 'HCP-MMP':
            mask_names = [vis_name]
            mask_names.extend(get_rois(vis_name))
        elif atlas_name == 'MMP-vis3-EDLV':
            mask_names = [i for i in atlas.roi2label.keys() if i.startswith(f'{Hemi}_')]
        else:
            raise ValueError('not supported atlas name:', atlas_name)
        n_mask = len(mask_names)

        # fit for each mask
        for mask_idx, mask_name in enumerate(mask_names, 1):
            time1 = time.time()

            if mask_name == vis_name:
                mask = atlas.get_mask(get_rois(mask_name))[0]
            else:
                mask = atlas.get_mask(mask_name)[0]

            for pc_name in pc_names:
                out_data[f'{mask_name}_{pc_name}'] = np.zeros(n_subj)
            for subj_idx in range(n_subj):
                m_map = m_maps[subj_idx][mask]
                t_map = t_maps[subj_idx][mask]
                X = np.array([m_map, t_map]).T
                Y = pc_maps[:, mask].T
                model = Pipeline([('preprocesser', StandardScaler()),
                                  ('regressor', LinearRegression())])
                model.fit(X, Y)
                Y_pred = model.predict(X)
                for pc_idx, pc_name in enumerate(pc_names):
                    out_data[f'{mask_name}_{pc_name}'][subj_idx] = \
                        r2_score(Y[:, pc_idx], Y_pred[:, pc_idx])

            print(f'Finished {atlas_name}-{mask_name}-{mask_idx}/{n_mask}: '
                  f'cost {time.time() - time1} seconds.')

    pkl.dump(out_data, open(out_file, 'wb'))


def weight_CCA_beh(vis_name):

    iqr_coef = 2  # None, 1.5, ...
    # 用上PC2的M和T的权重作为特征（一共2个）
    pc_names = ('C2',)
    meas_names = ('M', 'T')
    n_component = 2
    beh_cols = [
        'PicSeq_Unadj', 'CardSort_Unadj', 'Flanker_Unadj',
        'PMAT24_A_CR', 'ReadEng_Unadj', 'DDisc_AUC_200',
        'DDisc_AUC_40K', 'IWRD_TOT', 'ListSort_Unadj']

    # prepare file
    meas2file = {
        'M': pjoin(
            anal_dir, 'decomposition/'
            f'HCPY-M+corrT_{vis_name}_zscore1_PCA-subj_M.csv'),
        'T': pjoin(
            anal_dir, 'decomposition/'
            f'HCPY-M+corrT_{vis_name}_zscore1_PCA-subj_corrT.csv')}
    beh_file1 = '/nfs/z1/HCP/HCPYA/S1200_behavior.csv'
    beh_file2 = '/nfs/z1/HCP/HCPYA/S1200_behavior_restricted.csv'
    info_file = pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv')
    out_file = pjoin(work_dir, f'weight-CCA-beh_{vis_name}_v22.pkl')

    # load data
    feat_names = []
    weight_arr = []
    for meas_name in meas_names:
        weight_df = pd.read_csv(meas2file[meas_name], usecols=pc_names)
        weight_arr.append(weight_df.values)
        feat_names.extend([f'{i}_{meas_name}_weight' for i in pc_names])
    weight_arr = np.concatenate(weight_arr, axis=1)
    beh_df1 = pd.read_csv(beh_file1, index_col='Subject')
    beh_df2 = pd.read_csv(beh_file2, index_col='Subject')
    assert np.all(beh_df1.index == beh_df2.index)
    beh_df = pd.concat([beh_df1, beh_df2], axis=1)
    info_df = pd.read_csv(info_file, index_col='subID')
    beh_df = beh_df.loc[info_df.index, beh_cols]

    # prepare X and Y
    Y = beh_df.values
    non_nan_vec = ~np.any(np.isnan(Y), 1)
    Y = Y[non_nan_vec]
    X = weight_arr[non_nan_vec]
    if iqr_coef is not None:
        outlier_mask1 = outlier_iqr(X, iqr_coef, 0)
        outlier_mask2 = outlier_iqr(Y, iqr_coef, 0)
        outlier_mask1 = np.any(outlier_mask1, 1)
        outlier_mask2 = np.any(outlier_mask2, 1)
        outlier_mask = np.logical_or(outlier_mask1, outlier_mask2)
        mask = ~outlier_mask
        X = X[mask]
        Y = Y[mask]
    print('feature names:', feat_names)
    print('X.shape:', X.shape)
    print('behavior names:', beh_cols)
    print('Y.shape:', Y.shape)

    # CCA
    cca = CCA(n_components=n_component, scale=True)
    cca.fit(X, Y)
    X_trans, Y_trans = cca.transform(X, Y)

    # save out
    out_dict = {
        'model': cca, 'feature names': feat_names,
        'target names': beh_cols, 'X_trans': X_trans,
        'Y_trans': Y_trans}
    pkl.dump(out_dict, open(out_file, 'wb'))


def weight_CCA_beh1(vis_name):

    # 0. 用上C1和C2的（M和T的权重绝对值之和）作为特征（一共2个）
    # 1. 选择Unadjusted Scale Score，而非Age-Adjusted Scale Score
    # 2. 只保留正确率
    # 3. 对于Delay Discounting，只用两个AUC指标
    # 4. 对于Sustained Attention，只用sensitivity和specificity
    # 5. 去掉分类变量
    # 6. Visual Acuity保留EVA_Denom
    # 7. Contrast Sensitivity保留Mars_Final
    n_component = 2
    meas_names = ('M', 'T')
    pc_names = ('C1', 'C2')
    cognition_cols = [
        'PicSeq_Unadj', 'CardSort_Unadj', 'Flanker_Unadj',
        'PMAT24_A_CR', 'ReadEng_Unadj', 'PicVocab_Unadj',
        'ProcSpeed_Unadj', 'DDisc_AUC_200', 'DDisc_AUC_40K',
        'VSPLOT_TC', 'SCPT_SEN', 'SCPT_SPEC', 'IWRD_TOT',
        'ListSort_Unadj', 'CogFluidComp_Unadj', 'CogEarlyComp_Unadj',
        'CogTotalComp_Unadj', 'CogCrystalComp_Unadj']
    sensory_cols = [
        'Noise_Comp', 'Odor_Unadj', 'PainIntens_RawScore',
        'PainInterf_Tscore', 'Taste_Unadj', 'EVA_Denom', 'Mars_Final']
    beh_cols = cognition_cols + sensory_cols

    # prepare file
    meas2file = {
        'M': pjoin(
            anal_dir, 'decomposition/'
            f'HCPY-M+corrT_{vis_name}_zscore1_PCA-subj_M.csv'),
        'T': pjoin(
            anal_dir, 'decomposition/'
            f'HCPY-M+corrT_{vis_name}_zscore1_PCA-subj_corrT.csv')}
    beh_file1 = '/nfs/z1/HCP/HCPYA/S1200_behavior.csv'
    beh_file2 = '/nfs/z1/HCP/HCPYA/S1200_behavior_restricted.csv'
    info_file = pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv')
    out_file = pjoin(work_dir, f'weight-CCA-beh_{vis_name}_v8.pkl')

    # load data
    feat_names = [f'{i}_abs(w)_M+T' for i in pc_names]
    weight_arr = 0
    for meas_name in meas_names:
        weight_df = pd.read_csv(meas2file[meas_name], usecols=pc_names)
        weight_arr = weight_arr + np.abs(weight_df.values)
    beh_df1 = pd.read_csv(beh_file1, index_col='Subject')
    beh_df2 = pd.read_csv(beh_file2, index_col='Subject')
    assert np.all(beh_df1.index == beh_df2.index)
    beh_df = pd.concat([beh_df1, beh_df2], axis=1)
    info_df = pd.read_csv(info_file, index_col='subID')
    beh_df = beh_df.loc[info_df.index, beh_cols]

    # prepare X and Y
    Y = beh_df.values
    non_nan_vec = ~np.any(np.isnan(Y), 1)
    Y = Y[non_nan_vec]
    X = weight_arr[non_nan_vec]
    print('feature names:', feat_names)
    print('X.shape:', X.shape)
    print('behavior names:', beh_cols)
    print('Y.shape:', Y.shape)

    # CCA
    cca = CCA(n_components=n_component, scale=True)
    cca.fit(X, Y)
    X_trans, Y_trans = cca.transform(X, Y)

    # save out
    out_dict = {
        'model': cca, 'feature names': feat_names,
        'target names': beh_cols, 'X_trans': X_trans,
        'Y_trans': Y_trans}
    pkl.dump(out_dict, open(out_file, 'wb'))


if __name__ == '__main__':
    # gdist_fit_PC1()
    # gdist_fit_PC12()
    # HCPDA_fit_PC12()
    # HCPDA_MT_fit_PC12_SW(dataset_name='HCPD', vis_name='MMP-vis3-L',
    #                      width=50, step=10, merge_remainder=True)
    # HCPDA_MT_fit_PC12_SW(dataset_name='HCPA', vis_name='MMP-vis3-L',
    #                      width=50, step=10, merge_remainder=True)
    # HCPDA_MT_fit_PC12_SW(dataset_name='HCPD', vis_name='MMP-vis3-R',
    #                      width=50, step=10, merge_remainder=True)
    # HCPDA_MT_fit_PC12_SW(dataset_name='HCPA', vis_name='MMP-vis3-R',
    #                      width=50, step=10, merge_remainder=True)
    # mean_tau_diff_fit_PC12()
    # HCPDA_fit_PC12_local()
    # HCPDA_fit_PC12_local1(data_name='HCPD', Hemi='R')
    # HCPDA_fit_PC12_local1(data_name='HCPA', Hemi='R')
    # age_linearFit_col()
    # PC12_fit_func()
    # PC12_fit_func1()
    # PC12_fit_func2()
    # PC12_fit_func3(Hemi='L')
    # PC12_fit_func3(Hemi='R')
    # PC12_fit_func4(Hemi='L')
    # PC12_fit_func4(Hemi='R')
    HCPY_MT_fit_PC12(Hemi='L')
    HCPY_MT_fit_PC12(Hemi='R')
    # weight_CCA_beh(vis_name='MMP-vis3-R')
    # weight_CCA_beh(vis_name='MMP-vis3-L')
    # weight_CCA_beh1(vis_name='MMP-vis3-R')
    # weight_CCA_beh1(vis_name='MMP-vis3-L')
