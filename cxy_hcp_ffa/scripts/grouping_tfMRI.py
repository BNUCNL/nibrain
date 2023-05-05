from os.path import join as pjoin
from cxy_hcp_ffa.lib.predefine import proj_dir
from cxy_hcp_ffa.lib.algo import pre_ANOVA_3factors

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'grouping/tfMRI')


if __name__ == '__main__':
    # pre_ANOVA_3factors(
    #     meas_file=pjoin(anal_dir, 'tfMRI/FFA_activ.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_merged.csv'),
    #     out_file=pjoin(work_dir, 'FFA_activ_preANOVA-3factor.csv'),
    #     gids=(1, 2), rois=('pFus', 'mFus')
    # )
    # pre_ANOVA_3factors(
    #     meas_file=pjoin(anal_dir, 'tfMRI/FFA_activ-emo.csv'),
    #     gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv'),
    #     out_file=pjoin(work_dir, 'FFA_activ-emo_preANOVA-3factor-gid012.csv'),
    #     gids=(0, 1, 2), rois=('pFus', 'mFus')
    # )
