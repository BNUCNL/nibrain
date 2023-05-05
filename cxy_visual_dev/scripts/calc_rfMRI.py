import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats.stats import zscore
from scipy.spatial.distance import cdist
from cxy_visual_dev.lib.predefine import LR_count_32k, proj_dir,\
    Atlas, get_rois

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'rfMRI')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def pc_corr_rftp(Hemi):
    """
    Calculate correlation between structural PCs and
    rest-state functional time point map for runs which
    have 1200 time points in 1096 subjects.

    遍历1096个被试，只要有状态为'ok=(1200, 91282)'的静息run都拿来计算其时间点map和PC的相关。
    HCPY_pc-corr-rftp_R: 时间点map和PC直接做相关
    HCPY_pc-corr-rftp_remove-mean_R: 每个顶点的时间序列减去各自的均值之后的时间点map和PC做相关
    HCPY_pc-corr-rftp_zscore_R: 每个顶点的时间序列各自做zscore之后的时间点map和PC做相关
    """
    vis_name = f'MMP-vis3-{Hemi}'
    info_file = pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv')
    check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    pc_names = ['stru-C1', 'stru-C2']
    pc_file = pjoin(anal_dir, 'decomposition/'
                    'HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    run_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/{1}/'\
        '{1}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    out_file1 = pjoin(work_dir, f'HCPY_pc-corr-rftp_{Hemi}.pkl')
    out_file2 = pjoin(work_dir, f'HCPY_pc-corr-rftp_remove-mean_{Hemi}.pkl')
    out_file3 = pjoin(work_dir, f'HCPY_pc-corr-rftp_zscore_{Hemi}.pkl')

    # loading
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    n_pc = len(pc_names)
    pc_maps = nib.load(pc_file).get_fdata()[:n_pc, mask]
    df = pd.read_csv(info_file)
    df_ck = pd.read_csv(check_file, sep='\t', index_col='subID')

    n_subj = df.shape[0]
    out_dict1 = {'pc_name': pc_names}
    out_dict2 = {'pc_name': pc_names}
    out_dict3 = {'pc_name': pc_names}
    for sidx, sid in enumerate(df['subID'], 1):
        time1 = time.time()
        out_dict1[sid] = {}
        out_dict2[sid] = {}
        out_dict3[sid] = {}
        for run in runs:
            if df_ck.loc[sid, run] != 'ok=(1200, 91282)':
                continue
            run_file = run_files.format(sid, run)
            t_series = nib.load(run_file).get_fdata()[:, :LR_count_32k]
            t_series = t_series[:, mask]
            out_dict1[sid][run] = 1 - cdist(pc_maps, t_series, 'correlation')

            t_mean_map = np.mean(t_series, 0, keepdims=True)
            t_series_remove_mean = t_series - t_mean_map
            out_dict2[sid][run] = 1 - cdist(pc_maps, t_series_remove_mean, 'correlation')

            t_series_zscore = zscore(t_series, 0)
            out_dict3[sid][run] = 1 - cdist(pc_maps, t_series_zscore, 'correlation')
        print(f'Finished {sidx}/{n_subj}-{sid} cost {time.time() - time1} seconds.')

    # save out
    pkl.dump(out_dict1, open(out_file1, 'wb'))
    pkl.dump(out_dict2, open(out_file2, 'wb'))
    pkl.dump(out_dict3, open(out_file3, 'wb'))


if __name__ == '__main__':
    pc_corr_rftp(Hemi='R')
