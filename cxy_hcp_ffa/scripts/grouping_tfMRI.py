from os.path import join as pjoin

from numpy import nan
from cxy_hcp_ffa.lib.predefine import proj_dir

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'grouping/tfMRI')


def pre_ANOVA_3factors():
    """
    准备好3因素被试间设计方差分析需要的数据。
    2 hemispheres x 2 groups x 2 ROIs
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    gids = (1, 2)
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(anal_dir, 'tfMRI/individual_activ_{hemi}_emo.pkl')
    gid_file = pjoin(anal_dir, 'grouping/group_id_{hemi}_v2_merged.npy')
    trg_file = pjoin(work_dir, 'individual_activ-emo_preANOVA-3factor.csv')

    out_dict = {'hemi': [], 'gid': [], 'roi': [], 'meas': []}
    for hemi in hemis:
        data = pkl.load(open(src_file.format(hemi=hemi), 'rb'))
        gid_vec = np.load(gid_file.format(hemi=hemi))
        for gid in gids:
            gid_vec_idx = gid_vec == gid
            for roi in rois:
                roi_idx = data['roi'].index(roi)
                meas_vec = data['meas'][roi_idx][gid_vec_idx]
                meas_vec = meas_vec[~np.isnan(meas_vec)]
                n_valid = len(meas_vec)
                out_dict['hemi'].extend([hemi] * n_valid)
                out_dict['gid'].extend([gid] * n_valid)
                out_dict['roi'].extend([roi.split('-')[0]] * n_valid)
                out_dict['meas'].extend(meas_vec)
                print(f'{hemi}_{gid}_{roi}:', n_valid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def pre_ANOVA_3factors_mix():
    """
    准备好3因素混合设计方差分析需要的数据。
    被试间因子：group
    被试内因子：hemisphere，ROI
    2 groups x 2 hemispheres x 2 ROIs
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    gids = (1, 2)
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(anal_dir, 'tfMRI/individual_activ_{hemi}_emo.pkl')
    gid_file = pjoin(anal_dir, 'grouping/group_id_{hemi}_v2.npy')
    trg_file = pjoin(work_dir, 'individual_activ-emo_preANOVA-3factor-mix.csv')

    hemi2data = {}
    hemi2gids = {}
    for hemi in hemis:
        hemi2data[hemi] = pkl.load(open(src_file.format(hemi=hemi), 'rb'))
        hemi2gids[hemi] = np.load(gid_file.format(hemi=hemi))

    out_dict = {'gid': []}
    for idx, gid in enumerate(gids):
        gid_idx_vec = np.logical_and(hemi2gids['lh'] == gid,
                                     hemi2gids['rh'] == gid)
        nan_vec = None
        for hemi in hemis:
            data = hemi2data[hemi]
            for roi in rois:
                roi_idx = data['roi'].index(roi)
                meas_vec = data['meas'][roi_idx][gid_idx_vec]

                if nan_vec is None:
                    nan_vec = np.isnan(meas_vec)
                    non_nan_vec = ~nan_vec
                    n_valid = np.sum(non_nan_vec)
                    print('#NAN:', np.sum(nan_vec))
                else:
                    assert np.all(nan_vec == np.isnan(meas_vec))

                meas_vec = meas_vec[non_nan_vec]
                if idx == 0:
                    out_dict[f"{hemi}_{roi.split('-')[0]}"] = meas_vec.tolist()
                else:
                    out_dict[f"{hemi}_{roi.split('-')[0]}"].extend(meas_vec)
        print(f'G{gid}:', n_valid)
        out_dict['gid'].extend([gid] * n_valid)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


if __name__ == '__main__':
    # pre_ANOVA_3factors()
    pre_ANOVA_3factors_mix()
