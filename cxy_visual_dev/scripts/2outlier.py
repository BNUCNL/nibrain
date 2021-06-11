import os
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from matplotlib import pyplot as plt
from cxy_visual_dev.lib.predefine import Atlas, LR_count_32k

proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'analysis/outlier')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def select_subject():
    """
    从每个年龄随机选一个被试
    """
    info_file = '/nfs/e1/HCPD/HCPD_SubjInfo.csv'
    out_file = pjoin(work_dir, 'age_subj.csv')

    df = pd.read_csv(info_file)
    age_name = 'age in years'
    age_uniq = np.unique(df[age_name])

    subj_ids = []
    for age in age_uniq:
        row = df.loc[df[age_name] == age].sample(n=1, axis=0)
        subj_ids.append(row['subID'].item())

    out_data = {'age': age_uniq, 'subID': subj_ids}
    out_df = pd.DataFrame(out_data)
    out_df.to_csv(out_file, index=False)


def plot_box_violin(meas_name, roi_name):
    """
    观察上一步选中的被试，左/右视觉系统所有顶点的测量值分布。

    Args:
        meas_name (str): thickness | myelin
        roi_name (str): L_cole_visual | R_cole_visual
    """
    subj_file = pjoin(work_dir, 'age_subj.csv')
    meas2file = {
        'myelin': '/nfs/e1/HCPD/fmriresults01/{sid}_V1_MR/'
                  'MNINonLinear/fsaverage_LR32k/'
                  '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
        'thickness': '/nfs/e1/HCPD/fmriresults01/{sid}_V1_MR/'
                     'MNINonLinear/fsaverage_LR32k/'
                     '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
    }

    df = pd.read_csv(subj_file)
    atlas = Atlas('Cole_visual_LR')
    assert atlas.maps.shape == (1, LR_count_32k)
    roi_idx_map = atlas.maps[0] == atlas.roi2label[roi_name]

    points_list = []
    for subj_id in df['subID']:
        meas_file = meas2file[meas_name].format(sid=subj_id)
        meas_map = nib.load(meas_file).get_fdata()[0]
        meas_vec = meas_map[roi_idx_map]
        points_list.append(meas_vec)
    plt.boxplot(points_list, positions=df['age'])
    plt.violinplot(points_list, positions=df['age'], showextrema=False)
    plt.title(roi_name)
    plt.ylabel(meas_name)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_outlier_distribution(meas_name, roi_name):
    """
    统计在所有N个被试中，同时在至少x个被试中成为outliner的顶点数量
    x属于range(1, N+1)

    Args:
        meas_name (str): thickness | myelin
        roi_name (str): L_cole_visual | R_cole_visual
    """
    info_file = '/nfs/e1/HCPD/HCPD_SubjInfo.csv'


if __name__ == '__main__':
    # select_subject()
    plot_box_violin(meas_name='thickness', roi_name='R_cole_visual')
    plot_box_violin(meas_name='myelin', roi_name='R_cole_visual')
