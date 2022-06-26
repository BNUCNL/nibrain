import os
import numpy as np
import pandas as pd
import pickle as pkl
from os.path import join as pjoin
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from magicbox.io.io import CiftiReader, save2cifti

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'grouping')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def merge_group():
    """
    把pFus，mFus和two-C组合并为contiguous组
    """
    hemis = ('lh', 'rh')
    gid_files = pjoin(work_dir, 'group_id_{hemi}_v2.npy')
    out_files = pjoin(work_dir, 'group_id_{hemi}_v2_merged.npy')

    for hemi in hemis:
        gid_vec = np.load(gid_files.format(hemi=hemi))
        idx_vec = np.logical_or(gid_vec == -1, gid_vec == 0)
        gid_vec[idx_vec] = 1
        np.save(out_files.format(hemi=hemi), gid_vec)


def npy2csv(lh_file, rh_file, out_file):
    """
    把左右的分组编号合并到一个CSV文件中
    """
    df = pd.DataFrame()
    df['lh'] = np.load(lh_file)
    df['rh'] = np.load(rh_file)

    df.to_csv(out_file, index=False)


def count_subject(subj_mask=None):
    """
    统计每组的人数

    Args:
        subj_mask (bool vector, optional): Defaults to None.
            在统计人数时，只用这个mask指定的被试
    """
    # gids = (-1, 0, 1, 2)
    # gid_file = pjoin(work_dir, 'group_id_v2.csv')
    # gid2name = {-1: 'pFus', 0: 'mFus', 1: 'two-C', 2: 'two-S'}

    # gids = (1, 2)
    # gid_file = pjoin(work_dir, 'group_id_v2_merged.csv')
    # gid2name = {1: 'continuous', 2: 'separate'}

    gids = (0, 1, 2)
    gid_file = pjoin(work_dir, 'group_id_v2_012.csv')
    gid2name = {0: 'single', 1: 'continuous', 2: 'separate'}

    df = pd.read_csv(gid_file)
    if subj_mask is not None:
        df = df.loc[subj_mask]
    n_subj = df.shape[0]

    hemis = ('lh', 'rh')
    hemi2gid_vec = {}
    for hemi in hemis:
        hemi2gid_vec[hemi] = np.array(df[hemi])

    print('the number of subjects of each group:')
    for gid in gids:
        n_subjs = []
        for hemi in hemis:
            n_subj_g = np.sum(hemi2gid_vec[hemi] == gid)
            n_subjs.append(f'{n_subj_g}({n_subj_g / n_subj})')
        print(f"{gid2name[gid]} ({'/'.join(hemis)}): {'/'.join(n_subjs)}")


def roi_stats(gid=1, hemi='lh'):
    import nibabel as nib
    import pickle as pkl
    from cxy_hcp_ffa.lib.predefine import roi2label
    from magicbox.io.io import save2nifti

    # inputs
    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    gid_file = pjoin(work_dir, 'group_id_v2_012.csv')
    roi_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               f'rois_v3_{hemi}.nii.gz')

    # outputs
    rois_info_file = pjoin(work_dir, f'rois_info_{gid}_{hemi}.pkl')
    prob_maps_file = pjoin(work_dir, f'prob_maps_{gid}_{hemi}.nii.gz')

    # load
    df = pd.read_csv(gid_file)
    gid_idx_vec = np.array(df[hemi]) == gid
    roi_maps = nib.load(roi_file).get_data().squeeze().T[gid_idx_vec]

    # prepare
    rois_info = dict()
    prob_maps = np.zeros((roi_maps.shape[1], 1, 1, len(rois)),
                         dtype=np.float64)

    # calculate
    for roi_idx, roi in enumerate(rois):
        label = roi2label[roi]
        rois_info[roi] = dict()

        # get indices of subjects which contain the roi
        indices = roi_maps == label
        subj_indices = np.any(indices, 1)

        # calculate the number of the valid subjects
        n_subject = np.sum(subj_indices)
        rois_info[roi]['n_subject'] = n_subject

        # calculate roi sizes for each valid subject
        sizes = np.sum(indices[subj_indices], 1)
        rois_info[roi]['sizes'] = sizes

        # calculate roi probability map among valid subjects
        prob_map = np.mean(indices[subj_indices], 0)
        prob_maps[:, 0, 0, roi_idx] = prob_map

    # save
    pkl.dump(rois_info, open(rois_info_file, 'wb'))
    save2nifti(prob_maps_file, prob_maps)


def plot_roi_info(gid=1, hemi='lh'):
    import numpy as np
    import pickle as pkl
    from matplotlib import pyplot as plt
    from magicbox.algorithm.plot import show_bar_value, auto_bar_width

    roi_info_file = pjoin(work_dir, f'rois_info_{gid}_{hemi}.pkl')
    roi_infos = pkl.load(open(roi_info_file, 'rb'))

    # -plot n_subject-
    x_labels = list(roi_infos.keys())
    n_roi = len(x_labels)
    x = np.arange(n_roi)
    width = auto_bar_width(x)

    plt.figure()
    y_n_subj = [info['n_subject'] for info in roi_infos.values()]
    rects_subj = plt.bar(x, y_n_subj, width, facecolor='white', edgecolor='black')
    show_bar_value(rects_subj)
    plt.xticks(x, x_labels)
    plt.ylabel('#subject')

    # -plot sizes-
    for roi, info in roi_infos.items():
        plt.figure()
        sizes = info['sizes']
        bins = np.linspace(min(sizes), max(sizes), 40)
        _, _, patches = plt.hist(sizes, bins, color='white', edgecolor='black')
        plt.xlabel('#vertex')
        plt.title(f'distribution of {roi} sizes')
        show_bar_value(patches, '.0f')

    plt.tight_layout()
    plt.show()


def create_FFA_prob(src_file, gid_file, out_file):
    """
    基于CIFTI文件中的个体FFA，为各FFA计算在各组内的概率图
    每个顶点的值代表在出现该FFA的被试中，该顶点属于对应FFA的概率
    和roi_stats算出来的是一样的
    """
    Hemis = ('L', 'R')
    Hemi2hemi = {'L': 'lh', 'R': 'rh'}
    gids = (0, 1, 2)
    rois = ('pFus-faces', 'mFus-faces')
    key2name = {0: '???', 1: 'R_pFus-faces', 2: 'R_mFus-faces',
                3: 'L_pFus-faces', 4: 'L_mFus-faces'}
    name2key = {}
    for k, n in key2name.items():
        name2key[n] = k

    reader = CiftiReader(src_file)
    bms = reader.brain_models()
    data = reader.get_data()
    df = pd.read_csv(gid_file)

    out_dict = {}
    for Hemi in Hemis:
        gid_vec = np.array(df[Hemi2hemi[Hemi]])
        for gid in gids:
            gid_idx_vec = gid_vec == gid
            data_g = data[gid_idx_vec]
            for roi in rois:
                name = f'{Hemi}_{roi}'
                idx_arr = data_g == name2key[name]
                idx_vec = np.any(idx_arr, 1)
                idx_arr = idx_arr[idx_vec]
                out_dict[f'{Hemi}_G{gid}_{roi}'] = np.mean(idx_arr, 0)

    out_data = []
    map_names = []
    for gid in gids:
        for roi in rois:
            map_name = f'G{gid}_{roi}'
            out_data.append(out_dict[f'L_{map_name}'] + out_dict[f'R_{map_name}'])
            map_names.append(map_name)
    out_data = np.asarray(out_data)

    save2cifti(out_file, out_data, bms, map_names)


def calc_prob_map_similarity(src_file, out_file):
    """
    为每个ROI概率图计算组间dice系数和pearson相关
    """
    hemis = ('lh', 'rh')
    hemi2stru = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'}
    gids = (0, 1, 2)
    rois = ('pFus', 'mFus')

    reader = CiftiReader(src_file)
    prob_maps = reader.get_data()
    map_names = reader.map_names()

    # calculate
    n_gid = len(gids)
    data = {'shape': 'n_gid x n_gid',
            'gid': gids}
    for hemi in hemis:
        prob_maps = reader.get_data(hemi2stru[hemi], True)
        for roi in rois:
            roi_dice = f"{hemi}_{roi}_dice"
            roi_corr = f"{hemi}_{roi}_corr"
            roi_corr_p = f"{hemi}_{roi}_corr_p"
            data[roi_dice] = np.ones((n_gid, n_gid)) * np.nan
            data[roi_corr] = np.ones((n_gid, n_gid)) * np.nan
            data[roi_corr_p] = np.ones((n_gid, n_gid)) * np.nan
            for idx1, gid1 in enumerate(gids[:-1]):
                for idx2, gid2 in enumerate(gids[idx1+1:], idx1+1):
                    prob_idx1 = map_names.index(f'G{gid1}_{roi}-faces')
                    prob_idx2 = map_names.index(f'G{gid2}_{roi}-faces')
                    prob_map1 = prob_maps[prob_idx1]
                    prob_map2 = prob_maps[prob_idx2]
                    idx_map1 = prob_map1 != 0
                    idx_map2 = prob_map2 != 0
                    idx_map = np.logical_and(idx_map1, idx_map2)
                    dice = 2 * np.sum(idx_map) / \
                        (np.sum(idx_map1) + np.sum(idx_map2))
                    r, p = pearsonr(prob_map1[idx_map], prob_map2[idx_map])
                    data[roi_dice][idx1, idx2] = dice
                    data[roi_dice][idx2, idx1] = dice
                    data[roi_corr][idx1, idx2] = r
                    data[roi_corr][idx2, idx1] = r
                    data[roi_corr_p][idx1, idx2] = p
                    data[roi_corr_p][idx2, idx1] = p

    pkl.dump(data, open(out_file, 'wb'))


def calc_prob_map_similarity1():
    """
    为每个ROI概率图计算组间pearson相关
    限制在MPM内
    """
    import numpy as np
    import nibabel as nib
    import pickle as pkl
    from scipy.stats import pearsonr
    from cxy_hcp_ffa.lib.predefine import roi2label

    # inputs
    hemis = ('lh', 'rh')
    gids = (0, 1, 2)
    roi2idx = {'pFus-face': 1, 'mFus-face': 2}
    prob_file = pjoin(work_dir, 'prob_maps_{}_{}.nii.gz')
    mpm_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/MPM_v3_{}_0.25.nii.gz')

    # outputs
    out_file = pjoin(work_dir, 'prob_map_similarity1.pkl')

    # calculate
    n_gid = len(gids)
    data = {'shape': 'n_gid x n_gid',
            'gid': gids}
    for hemi in hemis:
        gid2prob_map = {}
        for gid in gids:
            gid2prob_map[gid] = nib.load(
                prob_file.format(gid, hemi)).get_fdata().squeeze().T
        mpm_map = nib.load(mpm_file.format(hemi)).get_fdata().squeeze()
        for roi, prob_idx in roi2idx.items():
            roi_r = f"{hemi}_{roi.split('-')[0]}_corr"
            roi_p = f"{hemi}_{roi.split('-')[0]}_corr_p"
            data[roi_r] = np.ones((n_gid, n_gid)) * np.nan
            data[roi_p] = np.ones((n_gid, n_gid)) * np.nan
            mpm_mask = mpm_map == roi2label[roi]
            for idx1, gid1 in enumerate(gids[:-1]):
                for idx2, gid2 in enumerate(gids[idx1+1:], idx1+1):
                    prob_map1 = gid2prob_map[gid1][prob_idx]
                    prob_map2 = gid2prob_map[gid2][prob_idx]
                    sample1 = prob_map1[mpm_mask]
                    print(f'{hemi}_{roi}_G{gid1}:', np.sum(sample1 == 0))
                    sample2 = prob_map2[mpm_mask]
                    print(f'{hemi}_{roi}_G{gid2}:', np.sum(sample2 == 0))
                    r, p = pearsonr(sample1, sample2)
                    data[roi_r][idx1, idx2] = r
                    data[roi_r][idx2, idx1] = r
                    data[roi_p][idx1, idx2] = p
                    data[roi_p][idx2, idx1] = p

    pkl.dump(data, open(out_file, 'wb'))


def plot_prob_map_similarity(src_file, meas):
    """
    References:
        1. https://blog.csdn.net/qq_27825451/article/details/105652244
        2. https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
        3. https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    """
    # inputs
    rois = ('pFus', 'mFus')
    hemis = ('lh', 'rh')
    fname = os.path.basename(src_file).split('.')[0]
    out_dir = os.path.dirname(src_file)
    out_file = pjoin(out_dir, f'{fname}_{meas}.jpg')
    gid2name = {0: 'single', 1: 'continuous', 2: 'separate'}

    # prepare
    data = pkl.load(open(src_file, 'rb'))
    n_gid = len(data['gid'])
    ticks = np.arange(n_gid)
    gid_names = [gid2name[gid] for gid in data['gid']]

    _, axes = plt.subplots(len(rois), len(hemis), figsize=(6.4, 6.4))
    for roi_idx, roi in enumerate(rois):
        for hemi_idx, hemi in enumerate(hemis):
            ax = axes[roi_idx, hemi_idx]
            arr = data[f'{hemi}_{roi}_{meas}']
            img = ax.imshow(arr, 'autumn')

            if roi_idx == 1:
                ax.set_xticks(ticks)
                ax.set_xticklabels(gid_names)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")
                if hemi_idx == 0:
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(gid_names)
                else:
                    ax.tick_params(left=False, labelleft=False)
            else:
                ax.tick_params(bottom=False, labelbottom=False)
                if hemi_idx == 0:
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(gid_names)
                else:
                    ax.tick_params(left=False, labelleft=False)

            for i in range(n_gid):
                for j in range(n_gid):
                    if i == j:
                        continue
                    ax.text(j, i, '{:.2f}'.format(arr[i, j]),
                            ha="center", va="center", color="k")
            ax.set_title(f'{hemi}_{roi}')

    plt.savefig(out_file)


if __name__ == '__main__':
    # merge_group()
    # count_subject()
    # count_subject(
    #     subj_mask=np.load(pjoin(anal_dir, 'subj_info/subject_id1.npy'))
    # )
    # npy2csv(
    #     lh_file=pjoin(work_dir, 'old_group_id_lh.npy'),
    #     rh_file=pjoin(work_dir, 'old_group_id_rh.npy'),
    #     out_file=pjoin(work_dir, 'old_group_id.csv')
    # )
    # npy2csv(
    #     lh_file=pjoin(work_dir, 'group_id_lh_v2.npy'),
    #     rh_file=pjoin(work_dir, 'group_id_rh_v2.npy'),
    #     out_file=pjoin(work_dir, 'group_id_v2.csv')
    # )
    # npy2csv(
    #     lh_file=pjoin(work_dir, 'group_id_lh_v2_merged.npy'),
    #     rh_file=pjoin(work_dir, 'group_id_rh_v2_merged.npy'),
    #     out_file=pjoin(work_dir, 'group_id_v2_merged.csv')
    # )

    # roi_stats(gid=0, hemi='lh')
    # roi_stats(gid=0, hemi='rh')
    # roi_stats(gid=1, hemi='lh')
    # roi_stats(gid=1, hemi='rh')
    # roi_stats(gid=2, hemi='lh')
    # roi_stats(gid=2, hemi='rh')
    # plot_roi_info(gid=0, hemi='lh')
    # plot_roi_info(gid=0, hemi='rh')
    # plot_roi_info(gid=1, hemi='lh')
    # plot_roi_info(gid=1, hemi='rh')
    # plot_roi_info(gid=2, hemi='lh')
    # plot_roi_info(gid=2, hemi='rh')

    # create_FFA_prob(
    #     src_file=pjoin(anal_dir, 'HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii'),
    #     gid_file=pjoin(work_dir, 'group_id_v2_012.csv'),
    #     out_file=pjoin(work_dir, 'HCP-YA_FFA-prob_grouping.32k_fs_LR.dscalar.nii'))
    # calc_prob_map_similarity(
    #     src_file=pjoin(work_dir, 'HCP-YA_FFA-prob_grouping.32k_fs_LR.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'prob_map_similarity.pkl'))
    create_FFA_prob(
        src_file=pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii'),
        gid_file=pjoin(anal_dir, 'NI_R1/data_1053/group_id_v2_012.csv'),
        out_file=pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-prob_grouping.32k_fs_LR.dscalar.nii'))
    calc_prob_map_similarity(
        src_file=pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-prob_grouping.32k_fs_LR.dscalar.nii'),
        out_file=pjoin(anal_dir, 'NI_R1/data_1053/prob_map_similarity.pkl'))

    # calc_prob_map_similarity1()

    # plot_prob_map_similarity(
    #     src_file=pjoin(work_dir, 'prob_map_similarity.pkl'),
    #     meas='corr')
    # plot_prob_map_similarity(
    #     src_file=pjoin(work_dir, 'prob_map_similarity.pkl'),
    #     meas='dice')
    plot_prob_map_similarity(
        src_file=pjoin(anal_dir, 'NI_R1/data_1053/prob_map_similarity.pkl'),
        meas='corr')
