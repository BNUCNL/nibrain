import os
import nibabel as nib
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir, Atlas,\
    get_rois
from cxy_visual_dev.lib.algo import cat_data_from_cifti,\
    linear_fit1

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'fit')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


if __name__ == '__main__':
    mask = Atlas('HCP-MMP').get_mask(
        get_rois('MMP-vis2-L') + get_rois('MMP-vis2-R'))[0]
    mask_L = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis2-L'))[0]
    mask_R = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis2-R'))[0]
    C1C2_maps = nib.load(pjoin(
        anal_dir, 'decomposition/HCPY-M+T_MMP-vis2-LR_zscore1-split_PCA-subj.dscalar.nii'
    )).get_fdata()[:2]

    # src_files = [
    #         pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
    #         pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii'),
    #         pjoin(anal_dir, 'gdist/gdist_src-MT.dscalar.nii')]
    # X_list = [cat_data_from_cifti([i], (1, 1), [mask])[0].T for i in src_files]
    # Y = C1C2_maps[0, mask][:, None]
    # linear_fit1(
    #     X_list=X_list, feat_names=['CalcarineSulcus', 'OccipitalPole', 'MT'],
    #     Y=Y, trg_names=['C1'], score_metric='R2',
    #     out_file=pjoin(work_dir, 'CalcS+OcPole+MT=C1_new.csv'),
    #     standard_scale=True
    # )

    # src_files = [
    #         pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #         pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii')]
    # X_list = [cat_data_from_cifti([i], (1, 1), [mask])[0].T for i in src_files]
    # Y = C1C2_maps[:, mask].T
    # linear_fit1(
    #     X_list=X_list, feat_names=['Myelination', 'Thickness'],
    #     Y=Y, trg_names=['C1', 'C2'], score_metric='R2',
    #     out_file=pjoin(work_dir, 'HCPD-M+T=C1C2_new.csv'),
    #     standard_scale=True
    # )

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
