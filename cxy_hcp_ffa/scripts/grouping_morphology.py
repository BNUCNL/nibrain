from os.path import join as pjoin
from cxy_hcp_ffa.lib.predefine import proj_dir
from cxy_hcp_ffa.lib.algo import pre_ANOVA_3factors

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'grouping/structure')


# ---old---
def pre_ANOVA(gid=1, morph='thickness'):
    """
    准备好二因素被试间设计方差分析需要的数据。
    半球x脑区
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    src_file = pjoin(work_dir, 'individual_G{}_{}_{}.pkl')
    trg_file = pjoin(work_dir, f'individual_G{gid}_{morph}_preANOVA.csv')

    out_dict = {'hemi': [], 'roi': [], 'meas': []}
    for hemi in hemis:
        data = pkl.load(open(src_file.format(gid, morph, hemi), 'rb'))
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


def pre_ANOVA_rm_individual(gid=1, morph='thickness'):
    """
    Preparation for two-way repeated-measures ANOVA
    半球x脑区
    """
    import pandas as pd
    import pickle as pkl

    # inputs
    rois = ('pFus-face', 'mFus-face')
    src_lh_file = pjoin(work_dir, f'individual_G{gid}_{morph}_lh.pkl')
    src_rh_file = pjoin(work_dir, f'individual_G{gid}_{morph}_rh.pkl')

    # outputs
    trg_file = pjoin(work_dir, f'individual_G{gid}_{morph}_preANOVA-rm.csv')

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


def plot_bar(gid=1, morph='thickness'):
    import numpy as np
    import pickle as pkl
    from scipy.stats import sem
    from nibrain.util.plotfig import auto_bar_width
    from matplotlib import pyplot as plt

    lh_file = pjoin(work_dir, f'individual_G{gid}_{morph}_lh.pkl')
    rh_file = pjoin(work_dir, f'individual_G{gid}_{morph}_rh.pkl')
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')
    roi2color = {'pFus-face': 'limegreen', 'mFus-face': 'cornflowerblue'}
    morph2ylabel = {'thickness': 'thickness',
                    'myelin': 'myelination',
                    'activ': 'face selectivity',
                    'va': 'region size'}
    morph2ylim = {'thickness': 2.7,
                  'myelin': 1.3,
                  'activ': 2,
                  'va': 200}
    hemi2meas = {
        'lh': pkl.load(open(lh_file, 'rb')),
        'rh': pkl.load(open(rh_file, 'rb'))}
    n_roi = len(rois)
    n_hemi = len(hemis)
    x = np.arange(n_hemi)
    width = auto_bar_width(x, n_roi)
    offset = -(n_roi - 1) / 2
    _, ax = plt.subplots()
    for roi in rois:
        y = np.zeros(n_hemi)
        y_err = np.zeros(n_hemi)
        for hemi_idx, hemi in enumerate(hemis):
            roi_idx = hemi2meas[hemi]['roi'].index(roi)
            meas = hemi2meas[hemi]['meas'][roi_idx]
            meas = meas[~np.isnan(meas)]
            y[hemi_idx] = np.mean(meas)
            y_err[hemi_idx] = sem(meas)
        ax.bar(x+width*offset, y, width, yerr=y_err,
               label=roi.split('-')[0], color=roi2color[roi])
        offset += 1
    ax.set_xticks(x)
    ax.set_xticklabels(hemis)
    ax.set_ylabel(morph2ylabel[morph])
    ax.set_ylim(morph2ylim[morph])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # pre_ANOVA_3factors(
    #     meas_file=pjoin(anal_dir, 'structure/FFA_thickness.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(work_dir, 'FFA_thickness_preANOVA-3factor-gid012.csv'),
    #     gids=(0, 1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors(
    #     meas_file=pjoin(anal_dir, 'structure/FFA_myelin.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(work_dir, 'FFA_myelin_preANOVA-3factor-gid012.csv'),
    #     gids=(0, 1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors(
    #     meas_file=pjoin(anal_dir, 'structure/FFA_va.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(work_dir, 'FFA_va_preANOVA-3factor-gid012.csv'),
    #     gids=(0, 1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors(
    #     meas_file=pjoin(anal_dir, 'structure/FFA_va.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_merged.csv'),
    #     out_file=pjoin(work_dir, 'FFA_va_preANOVA-3factor.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )

    # old
    # pre_ANOVA(gid=1, morph='thickness')
    # pre_ANOVA(gid=1, morph='myelin')
    # pre_ANOVA(gid=2, morph='thickness')
    # pre_ANOVA(gid=2, morph='myelin')
    # pre_ANOVA_rm_individual(gid=1, morph='thickness')
    # pre_ANOVA_rm_individual(gid=1, morph='myelin')
    # pre_ANOVA_rm_individual(gid=2, morph='thickness')
    # pre_ANOVA_rm_individual(gid=2, morph='myelin')
    # plot_bar(gid=1, morph='va')
    # plot_bar(gid=2, morph='va')
