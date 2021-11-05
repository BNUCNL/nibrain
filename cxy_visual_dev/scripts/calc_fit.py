import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    get_rois

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'fit')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def linear_fit1(src_files, src_names, trg_maps, trg_names, mask, score_metric,
                out_file):
    """
    每个src_file中的map数量相同，有多少map就迭代多少次。
    每次迭代用所有src_file中对应的map作为features，去拟合各target map。
    得到每次迭代对每个target的拟合分数，系数，截距

    Args:
        src_files (str | strings): .dscalar.nii files
        src_names (strings): feature names
        trg_maps (2D array): target maps
        trg_names (strings): target names
        mask (1D array): 指定区域
        score_metric (str): 目前只支持R2
        out_file (str): .csv file

    Raises:
        ValueError: [description]
    """
    if isinstance(src_files, str):
        src_files = [src_files]
    n_src = len(src_files)
    assert n_src == len(src_names)

    trg_maps = trg_maps[:, mask]
    n_trg = trg_maps.shape[0]
    assert n_trg == len(trg_names)
    n_vtx = trg_maps.shape[1]

    # load source maps
    n_iter = None
    src_maps_list = []
    for src_file in src_files:
        src_maps = nib.load(src_file).get_fdata()[:, mask]
        if n_iter is None:
            n_iter = src_maps.shape[0]
        else:
            assert n_iter == src_maps.shape[0]
        src_maps_list.append(src_maps)

    # fitting
    coefs = np.zeros((n_iter, n_trg, n_src), np.float64)
    intercepts = np.zeros((n_iter, n_trg), np.float64)
    scores = np.zeros((n_iter, n_trg), np.float64)
    for iter_idx in range(n_iter):
        time1 = time.time()
        X = np.zeros((n_vtx, n_src), np.float64)
        for src_idx in range(n_src):
            X[:, src_idx] = src_maps_list[src_idx][iter_idx]
        pipe = Pipeline([('preprocesser', StandardScaler()),
                         ('regressor', LinearRegression())])
        pipe.fit(X, trg_maps.T)
        coefs[iter_idx] = pipe.named_steps['regressor'].coef_
        intercepts[iter_idx] = pipe.named_steps['regressor'].intercept_
        Y = pipe.predict(X).T
        if score_metric == 'R2':
            scores[iter_idx] = [
                r2_score(trg_maps[i], Y[i]) for i in range(n_trg)]
        else:
            raise ValueError('not supported score metric')
        print(f'Finished {iter_idx + 1}/{n_iter}, '
              f'cost {time.time() - time1} seconds.')

    # save
    df = pd.DataFrame()
    for trg_idx, trg_name in enumerate(trg_names):
        for src_idx, src_name in enumerate(src_names):
            df[f'coef_{trg_name}_{src_name}'] = coefs[:, trg_idx, src_idx]
        df[f'score_{trg_name}'] = scores[:, trg_idx]
    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    mask = Atlas('HCP-MMP').get_mask(
        get_rois('MMP-vis2-L') + get_rois('MMP-vis2-R'))[0]
    C1C2_maps = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii'
    )).get_fdata()[:2]

    linear_fit1(
        src_files=[
            pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'),
            pjoin(anal_dir, 'gdist/gdist_src-MT.dscalar.nii')
        ], src_names=['CalcarineSulcus', 'OccipitalPole', 'MT'],
        trg_maps=C1C2_maps[[0]], trg_names=['C1'], mask=mask, score_metric='R2',
        out_file=pjoin(work_dir, 'CalcS+OcPole+MT=C1.csv')
    )

    linear_fit1(
        src_files=[
            pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
            pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')
        ], src_names=['Myelination', 'Thickness'],
        trg_maps=C1C2_maps, trg_names=['C1', 'C2'], mask=mask, score_metric='R2',
        out_file=pjoin(work_dir, 'HCPD-M+T=C1C2.csv')
    )
