from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
split_dir = pjoin(proj_dir,
                  'analysis/s2/1080_fROI/refined_with_Kevin/grouping/split')
work_dir = pjoin(split_dir, 'tfMRI')


def split_half12():
    """
    分别将左右脑，组1组2的被试随机分成两半
    组1的half1和half2分别编号为11, 12
    组2的half1和half2分别编号为21, 22
    """
    import numpy as np

    gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     'grouping/group_id_{}.npy')
    trg_file = pjoin(split_dir, 'half12_id_{}.npy')

    for hemi in ('lh', 'rh'):
        gids = np.load(gid_file.format(hemi))
        for gid in (1, 2):
            gh1_id = gid * 10 + 1
            gh2_id = gid * 10 + 2
            gid_idx_vec = gids == gid
            indices = np.where(gid_idx_vec)[0]
            n_gid = len(indices)
            gids[gid_idx_vec] = gh1_id
            half2_indices = np.random.choice(indices, int(n_gid/2),
                                             replace=False)
            gids[half2_indices] = gh2_id

        print(f'The size of group1-half1-{hemi}:', np.sum(gids == 11))
        print(f'The size of group1-half2-{hemi}:', np.sum(gids == 12))
        print(f'The size of group2-half1-{hemi}:', np.sum(gids == 21))
        print(f'The size of group2-half2-{hemi}:', np.sum(gids == 22))
        np.save(trg_file.format(hemi), gids)


def split_half0():
    """
    将组0的半脑随机分成两半
    同时保证pFus或mFus也是对半分的
    half1和half2分别编号为01, 02
    """
    import numpy as np
    import pandas as pd

    gh1_id = 1
    gh2_id = 2
    gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     'grouping/group_id_{}.npy')
    roi_idx_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                         'rois_v3_idx_vec.csv')
    trg_file = pjoin(split_dir, 'half0_id_{}.npy')

    df = pd.read_csv(roi_idx_file)

    for hemi in ('lh', 'rh'):
        gids = np.load(gid_file.format(hemi))
        gids_new = np.ones_like(gids) * np.nan
        p_idx_vec = np.array(df[f'{hemi}_pFus-face'])
        m_idx_vec = np.array(df[f'{hemi}_mFus-face'])
        gid_idx_vec = gids == 0

        gid_p_idx_vec = np.logical_and(p_idx_vec, gid_idx_vec)
        gid_p_indices = np.where(gid_p_idx_vec)[0]
        n_gid_p = len(gid_p_indices)
        gid_m_idx_vec = np.logical_and(m_idx_vec, gid_idx_vec)
        gid_m_indices = np.where(gid_m_idx_vec)[0]
        n_gid_m = len(gid_m_indices)

        gids_new[gid_idx_vec] = gh1_id
        half2_indices_p = np.random.choice(
            gid_p_indices, int(n_gid_p/2), replace=False
        )
        half2_indices_m = np.random.choice(
            gid_m_indices, int(n_gid_m/2), replace=False
        )
        gids_new[half2_indices_p] = gh2_id
        gids_new[half2_indices_m] = gh2_id

        print(f'The size of G0_pFus_{hemi}:', n_gid_p)
        print(f'The size of G0_pFus-half2_{hemi}:', len(half2_indices_p))
        print(f'The size of G0_mFus_{hemi}:', n_gid_m)
        print(f'The size of G0_mFus-half2_{hemi}:', len(half2_indices_m))
        print(f'The size of G0-half1-{hemi}:', np.sum(gids_new == 1))
        print(f'The size of G0-half2-{hemi}:', np.sum(gids_new == 2))
        np.save(trg_file.format(hemi), gids_new)


def merge_gh_id():
    import numpy as np

    hemis = ('lh', 'rh')
    g0h_id_file = pjoin(split_dir, 'half0_id_{}.npy')
    g12h_id_file = pjoin(split_dir, 'half12_id_{}.npy')
    out_file = pjoin(split_dir, 'half_id_{}.npy')

    for hemi in hemis:
        g0h_ids = np.load(g0h_id_file.format(hemi))
        g12h_ids = np.load(g12h_id_file.format(hemi))
        gh_ids = np.zeros_like(g0h_ids, int)

        non_nan_vec = ~np.isnan(g0h_ids)
        non_0_vec = g12h_ids != 0
        assert np.all(non_0_vec == ~non_nan_vec)
        gh_ids[non_nan_vec] = g0h_ids[non_nan_vec]
        gh_ids[non_0_vec] = g12h_ids[non_0_vec]

        np.save(out_file.format(hemi), gh_ids)


def count_roi():
    """
    可视化左右脑各组分半后的ROI数量
    """
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from magicbox.algorithm.plot import show_bar_value
    from nibrain.util.plotfig import auto_bar_width

    hemis = ('lh', 'rh')
    n_hemi = len(hemis)
    gids = (0, 1, 2)
    n_gid = len(gids)
    rois = ('pFus-face', 'mFus-face')
    xticklabels = [roi.split('-')[0] for roi in rois]
    gh_id_file = pjoin(split_dir, 'half_id_{}.npy')
    roi_idx_vec_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                             'rois_v3_idx_vec.csv')

    df = pd.read_csv(roi_idx_vec_file)

    x = np.arange(len(rois))
    width = auto_bar_width(x, 2)
    _, axes = plt.subplots(n_gid, n_hemi)
    for hemi_idx, hemi in enumerate(hemis):
        gh_ids = np.load(gh_id_file.format(hemi))
        for gid_idx, gid in enumerate(gids):
            ax = axes[gid_idx, hemi_idx]
            gh1_id = gid * 10 + 1
            gh2_id = gid * 10 + 2
            y1 = []
            y2 = []
            for roi in rois:
                col = f'{hemi}_{roi}'
                y1.append(np.sum(df[col][gh_ids == gh1_id]))
                y2.append(np.sum(df[col][gh_ids == gh2_id]))
            rects1 = ax.bar(x-width/2, y1, width, label='half1')
            rects2 = ax.bar(x+width/2, y2, width, label='half2')
            show_bar_value(rects1, ax=ax)
            show_bar_value(rects2, ax=ax)

            # if hemi_idx == 0 and gid_idx == 0:
            #     ax.legend()
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels)
            if hemi_idx == 0:
                ax.set_ylabel('#subject')
            if gid_idx == 0:
                ax.set_title(hemi)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def calc_prob_map():
    import numpy as np
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import roi2label
    from magicbox.io.io import save2nifti

    hemis = ('lh', 'rh')
    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    n_roi = len(rois)
    gh_ids = (1, 2, 11, 12, 21, 22)
    gh_id_file = pjoin(split_dir, 'half_id_{}.npy')
    roi_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     'rois_v3_{}.nii.gz')
    out_file = pjoin(split_dir, 'prob_maps_GH{}_{}.nii.gz')

    for hemi in hemis:
        gh_id_vec = np.load(gh_id_file.format(hemi))
        roi_maps = nib.load(roi_file.format(hemi)).get_data().squeeze().T
        for gh_id in gh_ids:
            gh_id_idx_vec = gh_id_vec == gh_id
            roi_maps_tmp = roi_maps[gh_id_idx_vec]

            prob_maps = np.zeros((n_roi, roi_maps.shape[1]))
            for idx, roi in enumerate(rois):
                label = roi2label[roi]

                # get indices of subjects which contain the roi
                indices = roi_maps_tmp == label
                subj_indices = np.any(indices, 1)

                # calculate roi probability map among valid subjects
                prob_map = np.mean(indices[subj_indices], 0)
                prob_maps[idx] = prob_map

            # save out
            save2nifti(out_file.format(gh_id, hemi),
                       prob_maps.T[:, None, None, :])


def get_mpm(gh_id=11, hemi='lh'):
    """maximal probability map"""
    import numpy as np
    import nibabel as nib
    from commontool.io.io import save2nifti

    thr = 0.25
    prob_map_file = pjoin(split_dir, f'prob_maps_GH{gh_id}_{hemi}.nii.gz')
    out_file = pjoin(split_dir, f'MPM_GH{gh_id}_{hemi}.nii.gz')

    prob_maps = nib.load(prob_map_file).get_data()
    supra_thr_idx_arr = prob_maps > thr
    prob_maps[~supra_thr_idx_arr] = 0
    mpm = np.argmax(prob_maps, 3)
    mpm[np.any(prob_maps, 3)] += 1

    # save
    save2nifti(out_file, mpm)


def calc_meas(gid=1, hemi='lh'):
    """
    用一半被试的MPM去提取另一半被试的激活值
    如果某个被试没有某个ROI，就不提取该被试该ROI的信号

    Args:
        gid (int, optional): group ID. Defaults to '1'.
        hemi (str, optional): hemisphere. Defaults to 'lh'.
    """
    import numpy as np
    import pickle as pkl
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import hemi2stru, roi2label
    from commontool.io.io import CiftiReader

    gh_ids = (gid*10+1, gid*10+2)
    gh_id_file = pjoin(split_dir, f'half_id_{hemi}.npy')
    mpm_file = pjoin(split_dir, 'MPM_GH{gh_id}_{hemi}.nii.gz')
    roi_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                               f'rois_v3_{hemi}.nii.gz')
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    src_file = pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    trg_file = pjoin(work_dir, f'G{gid}_activ_{hemi}.pkl')

    gh_id_vec = np.load(gh_id_file)
    gid_idx_vec = np.logical_or(gh_id_vec == gh_ids[0], gh_id_vec == gh_ids[1])
    hid_vec = gh_id_vec[gid_idx_vec]
    subj_ids = open(subj_file).read().splitlines()
    subj_ids = [subj_ids[i] for i in np.where(gid_idx_vec)[0]]
    n_subj = len(subj_ids)
    roi_maps = nib.load(roi_file).get_data().squeeze().T[gid_idx_vec]
    meas_reader = CiftiReader(src_file)
    meas = meas_reader.get_data(hemi2stru[hemi], True)[gid_idx_vec]

    out_dict = {'shape': 'n_roi x n_subj',
                'roi': list(roi2label.keys()),
                'subject': subj_ids,
                'meas': np.ones((len(roi2label), n_subj)) * np.nan}
    for gh_id in gh_ids:
        hid_idx_vec = hid_vec == gh_id
        mpm = nib.load(mpm_file.format(gh_id=gh_id, hemi=hemi)
                       ).get_data().squeeze()
        for roi_idx, roi in enumerate(out_dict['roi']):
            roi_idx_vec = np.any(roi_maps == roi2label[roi], 1)
            valid_idx_vec = np.logical_and(~hid_idx_vec, roi_idx_vec)
            mpm_idx_vec = mpm == roi2label[roi]
            meas_masked = meas[valid_idx_vec][:, mpm_idx_vec]
            out_dict['meas'][roi_idx][valid_idx_vec] = np.mean(meas_masked, 1)
    pkl.dump(out_dict, open(trg_file, 'wb'))


def pre_ANOVA(gid=1):
    """
    准备好二因素被试间设计方差分析需要的数据。
    半球x脑区
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'G{}_activ_{}.pkl')
    trg_file = pjoin(work_dir, f'G{gid}_activ_preANOVA.csv')

    out_dict = {'hemi': [], 'roi': [], 'meas': []}
    for hemi in hemis:
        data = pkl.load(open(src_file.format(gid, hemi), 'rb'))
        for roi in rois:
            roi_idx = data['roi'].index(roi)
            meas_vec = data['meas'][roi_idx]
            meas_vec = meas_vec[~np.isnan(meas_vec)]
            n_valid = len(meas_vec)
            out_dict['hemi'].extend([hemi] * n_valid)
            out_dict['roi'].extend([roi.split('-')[0]] * n_valid)
            out_dict['meas'].extend(meas_vec)
            print(f'{hemi}_{roi}:', n_valid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def pre_ANOVA_rm(gid=1):
    """
    Preparation for two-way repeated-measures ANOVA
    半球x脑区
    """
    import pandas as pd
    import pickle as pkl

    # inputs
    rois = ('pFus-face', 'mFus-face')
    src_lh_file = pjoin(work_dir, f'G{gid}_activ_lh.pkl')
    src_rh_file = pjoin(work_dir, f'G{gid}_activ_rh.pkl')

    # outputs
    trg_file = pjoin(work_dir, f'G{gid}_activ_preANOVA-rm.csv')

    # load data
    data_lh = pkl.load(open(src_lh_file, 'rb'))
    data_rh = pkl.load(open(src_rh_file, 'rb'))
    valid_indices_lh = [i for i, j in enumerate(data_lh['subject'])
                        if j in data_rh['subject']]
    valid_indices_rh = [i for i, j in enumerate(data_rh['subject'])
                        if j in data_lh['subject']]
    assert [data_lh['subject'][i] for i in valid_indices_lh] == \
           [data_rh['subject'][i] for i in valid_indices_rh]
    print(f'#valid subjects in G{gid}:', len(valid_indices_lh))

    # start
    out_dict = {}
    for roi in rois:
        roi_idx = data_lh['roi'].index(roi)
        meas_lh = data_lh['meas'][roi_idx][valid_indices_lh]
        out_dict[f"lh_{roi.split('-')[0]}"] = meas_lh
    for roi in rois:
        roi_idx = data_rh['roi'].index(roi)
        meas_rh = data_rh['meas'][roi_idx][valid_indices_rh]
        out_dict[f"rh_{roi.split('-')[0]}"] = meas_rh
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def pre_ANOVA_3factors():
    """
    准备好3因素被试间设计方差分析需要的数据。
    2 groups x 2 hemispheres x 2 ROIs
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    gids = (1, 2)
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'G{}_activ_{}.pkl')
    trg_file = pjoin(work_dir, f'activ_preANOVA-3factor.csv')

    out_dict = {'gid': [], 'hemi': [], 'roi': [], 'meas': []}
    for gid in gids:
        for hemi in hemis:
            data = pkl.load(open(src_file.format(gid, hemi), 'rb'))
            for roi in rois:
                roi_idx = data['roi'].index(roi)
                meas_vec = data['meas'][roi_idx]
                meas_vec = meas_vec[~np.isnan(meas_vec)]
                n_valid = len(meas_vec)
                out_dict['gid'].extend([gid] * n_valid)
                out_dict['hemi'].extend([hemi] * n_valid)
                out_dict['roi'].extend([roi.split('-')[0]] * n_valid)
                out_dict['meas'].extend(meas_vec)
                print(f'{gid}_{hemi}_{roi}:', n_valid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


if __name__ == '__main__':
    # split_half12()
    # split_half0()
    # merge_gh_id()
    # count_roi()
    calc_prob_map()
    # get_mpm(gh_id=11, hemi='lh')
    # get_mpm(gh_id=11, hemi='rh')
    # get_mpm(gh_id=21, hemi='lh')
    # get_mpm(gh_id=21, hemi='rh')
    # get_mpm(gh_id=12, hemi='lh')
    # get_mpm(gh_id=12, hemi='rh')
    # get_mpm(gh_id=22, hemi='lh')
    # get_mpm(gh_id=22, hemi='rh')
    # calc_meas(gid=1, hemi='lh')
    # calc_meas(gid=1, hemi='rh')
    # calc_meas(gid=2, hemi='lh')
    # calc_meas(gid=2, hemi='rh')
    # pre_ANOVA(gid=1)
    # pre_ANOVA(gid=2)
    # pre_ANOVA_rm(gid=1)
    # pre_ANOVA_rm(gid=2)
    # pre_ANOVA_3factors()
