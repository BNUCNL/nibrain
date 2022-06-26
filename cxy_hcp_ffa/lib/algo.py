import numpy as np
import pandas as pd
import pickle as pkl


def meas_pkl2csv(lh_file, rh_file, out_file, rois):
    """
    把原来左右脑用两个pickle文件分开存的数据格式，转换到一个csv文件中
    目前实测对个体ROI结构和激活数据可用，其它的可以继续探索，应该差不多

    Args:
        lh_file (str): 左脑pickle数据
        rh_file (str): 右脑pickle数据
        out_file (str): 整理后的CSV文件
        rois (sequence): 指定整理哪些ROI
            IOG-face, pFus-face, mFus-face
    """
    hemis = ('lh', 'rh')
    hemi2data = {
        'lh': pkl.load(open(lh_file, 'rb')),
        'rh': pkl.load(open(rh_file, 'rb'))
    }

    df = pd.DataFrame()
    for hemi in hemis:
        data = hemi2data[hemi]
        for roi in rois:
            col = f"{hemi}_{roi.split('-')[0]}"
            roi_idx = data['roi'].index(roi)
            df[col] = data['meas'][roi_idx]

    df.to_csv(out_file, index=False)


def pre_ANOVA_3factors(meas_file, gid_file, out_file, gids, rois):
    """
    准备好3因素被试间设计方差分析需要的数据。
    2 hemispheres x groups x ROIs

    Args:
        meas_file (str): CSV data
        gid_file (str): CSV group ID
        out_file (str): CSV preANOVA
        gids (sequence): group IDs
        rois (sequence): IOG | pFus | mFus
    """
    hemis = ('lh', 'rh')
    meas_df = pd.read_csv(meas_file)
    gid_df = pd.read_csv(gid_file)

    out_dict = {'hemi': [], 'gid': [], 'roi': [], 'meas': []}
    for hemi in hemis:
        for gid in gids:
            gid_vec_idx = gid_df[hemi] == gid
            for roi in rois:
                meas_vec = meas_df[f'{hemi}_{roi}'][gid_vec_idx]
                meas_vec.dropna(inplace=True)
                n_valid = len(meas_vec)
                out_dict['hemi'].extend([hemi] * n_valid)
                out_dict['gid'].extend([gid] * n_valid)
                out_dict['roi'].extend([roi] * n_valid)
                out_dict['meas'].extend(meas_vec)
                print(f'#{hemi}_{gid}_{roi}:', n_valid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(out_file, index=False)
