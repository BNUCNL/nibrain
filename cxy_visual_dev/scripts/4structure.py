import os
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir,\
    dataset_name2info, s1200_1096_thickness, s1200_1096_myelin,\
    s1200_avg_thickness, s1200_avg_myelin, rPath1, rPath4
from cxy_visual_dev.lib.algo import ROI_analysis, pca,\
    ROI_analysis_on_PC, make_age_maps, calc_map_corr,\
    mask_maps, merge_by_age, vtx_corr_col, polyfit, row_corr_row,\
    col_operate_col, map_operate_map, zscore_map, concate_map,\
    zscore_map_subj

work_dir = pjoin(proj_dir, 'analysis/structure')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


if __name__ == '__main__':
    # HCP_MMP1 atlas 包含 Cole_visual_ROI

    # zscore_map(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPD-thickness_zscore-R_cole_visual.dscalar.nii'),
    #     atlas_name='Cole_visual_LR', roi_name='R_cole_visual'
    # )

    # concate_map(
    #     data_files=[
    #         pjoin(work_dir, 'HCPD-myelin_zscore-R_cole_visual.dscalar.nii'),
    #         pjoin(work_dir, 'HCPD-thickness_zscore-R_cole_visual.dscalar.nii')
    #     ],
    #     out_file=pjoin(work_dir, 'HCPD-myelin+thickness_zscore-R_cole_visual.dscalar.nii')
    # )

    # zscore_map_subj(
    #     data_file=pjoin(work_dir, 'HCPD-thickness_zscore-R_cole_visual.dscalar.nii'),
    #     out_file=pjoin(work_dir, 'HCPD-thickness_zscore-R_cole_visual-subj.dscalar.nii')
    # )

    # ROI_analysis(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPA_thickness.dscalar.nii'),
    #     atlas_name='HCP_MMP1',
    #     out_file=pjoin(work_dir, 'HCPA_thickness_HCP_MMP1.csv')
    # )
    # ROI_analysis(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii'),
    #     atlas_name='HCP_MMP1', zscore_flag=True,
    #     out_file=pjoin(work_dir, 'HCPD_thickness_HCP_MMP1_zscore.csv')
    # )
    # ROI_analysis(
    #     data_file=s1200_1096_thickness,
    #     atlas_name='Cole_visual_LR',
    #     out_file=pjoin(work_dir, 'HCPY_thickness_Cole_visual_LR.csv')
    # )
    # ROI_analysis(
    #     data_file=s1200_avg_myelin,
    #     atlas_name='FFA',
    #     out_file=pjoin(work_dir, 'HCPY_myelin-avg_FFA.csv')
    # )
    # ROI_analysis(
    #     data_file=s1200_avg_thickness,
    #     atlas_name='HCP_MMP1',
    #     out_file=pjoin(work_dir, 'HCPY_thickness-avg_HCP_MMP1.csv')
    # )

    # merge_by_age(
    #     data_file=pjoin(work_dir, 'HCPD_thickness_HCP_MMP1.csv'),
    #     info_file=dataset_name2info['HCPD'],
    #     out_name=pjoin(work_dir, 'HCPD_thickness_HCP_MMP1_merge-age')
    # )

    # pca(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness_4mm.dscalar.nii'),
    #     atlas_name='Cole_visual_LR', roi_name='R_cole_visual',
    #     n_component=20, axis='vertex',
    #     out_name=pjoin(work_dir, 'HCPD_thickness_4mm_R_cole_visual_PCA-vtx')
    # )
    # pca(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness_4mm.dscalar.nii'),
    #     atlas_name='Cole_visual_LR', roi_name='R_cole_visual',
    #     n_component=20, axis='subject',
    #     out_name=pjoin(work_dir, 'HCPD_thickness_4mm_R_cole_visual_PCA-subj')
    # )

    # ROI_analysis_on_PC(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness_4mm.dscalar.nii'),
    #     pca_file=pjoin(work_dir, 'HCPD_thickness_4mm_R_cole_visual_PCA-vtx.pkl'),
    #     pc_num=1, mask_atlas='Cole_visual_LR', mask_name='R_cole_visual',
    #     roi_atlas='Cole_visual_ROI',
    #     out_file=pjoin(work_dir, 'HCPD_thickness_4mm_R_cole_visual_PCA-vtx-PC1.csv')
    # )

    # make_age_maps(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii'),
    #     info_file=dataset_name2info['HCPD'],
    #     out_name=pjoin(work_dir, 'HCPD_thickness_age-map')
    # )
    # make_age_maps(
    #     data_file=s1200_1096_thickness,
    #     info_file=dataset_name2info['HCPY'],
    #     out_name=pjoin(work_dir, 'HCPY_thickness_age-map')
    # )
    # make_age_maps(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPA_thickness.dscalar.nii'),
    #     info_file=dataset_name2info['HCPA'],
    #     out_name=pjoin(work_dir, 'HCPA_thickness_age-map')
    # )
    # make_age_maps(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness_4mm.dscalar.nii'),
    #     info_file=dataset_name2info['HCPD'],
    #     out_name=pjoin(work_dir, 'HCPD_thickness_4mm_age-map')
    # )
    # make_age_maps(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_myelin_4mm.dscalar.nii'),
    #     info_file=dataset_name2info['HCPD'],
    #     out_name=pjoin(work_dir, 'HCPD_myelin_4mm_age-map')
    # )

    # calc_map_corr(
    #     data_file1=pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii'),
    #     data_file2=s1200_avg_thickness,
    #     atlas_name='Cole_visual_LR', roi_name='R_cole_visual',
    #     out_file=pjoin(work_dir, 'HCPD_thickness_map-corr_s1200-avg_R_cole_visual.csv'),
    #     map_names2=['s1200_avg'], index=False
    # )
    # calc_map_corr(
    #     data_file1=s1200_1096_thickness,
    #     data_file2=s1200_avg_thickness,
    #     atlas_name='Cole_visual_LR', roi_name='R_cole_visual',
    #     out_file=pjoin(work_dir, 'HCPY_thickness_map-corr_s1200-avg_R_cole_visual.csv'),
    #     map_names2=['s1200_avg'], index=False
    # )
    # calc_map_corr(
    #     data_file1=pjoin(work_dir, 'HCPD_thickness_age-map-mean.dscalar.nii'),
    #     data_file2=s1200_avg_thickness,
    #     atlas_name='Cole_visual_LR', roi_name='R_cole_visual',
    #     out_file=pjoin(work_dir, 'HCPD_thickness_age-map_map-corr_s1200-avg_R_cole_visual.csv'),
    #     map_names2=['s1200_avg'], index=True
    # )
    calc_map_corr(
        data_file1=pjoin(proj_dir, 'data/HCP/HCPD_thickness.dscalar.nii'),
        data_file2=s1200_avg_thickness,
        atlas_name='Cole_visual_L1', roi_name='L_cole_visual1',
        out_file=pjoin(work_dir, 'HCPD-thickness_map-corr_s1200-avg_L_cole_visual1.csv'),
        map_names2=['s1200_avg'], index=False
    )
    calc_map_corr(
        data_file1=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
        data_file2=s1200_avg_myelin,
        atlas_name='Cole_visual_L1', roi_name='L_cole_visual1',
        out_file=pjoin(work_dir, 'HCPD-myelin_map-corr_s1200-avg_L_cole_visual1.csv'),
        map_names2=['s1200_avg'], index=False
    )

    # vtx_corr_col(
    #     data_file1=pjoin(work_dir, 'HCPD_thickness_age-map-mean.dscalar.nii'),
    #     atlas_name='Cole_visual_LR', roi_name='R_cole_visual',
    #     data_file2=pjoin(work_dir, 'HCPD_thickness_Cole_visual_LR_merge-age-mean.csv'),
    #     column='R_cole_visual', idx_col=None,
    #     out_file=pjoin(work_dir, 'HCPD_thickness_R_cole_visual_merge-age_vtx-corr-col.dscalar.nii')
    # )

    # polyfit(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness_4mm.dscalar.nii'),
    #     info_file=dataset_name2info['HCPD'], deg=1,
    #     out_file=pjoin(work_dir, 'HCPD_thickness_4mm_linear-fit-age.dscalar.nii')
    # )
    # polyfit(
    #     data_file=pjoin(work_dir, 'HCPD_thickness_4mm_HCP_MMP1.csv'),
    #     info_file=dataset_name2info['HCPD'], deg=1,
    #     out_file=pjoin(work_dir, 'HCPD_thickness_4mm_HCP_MMP1_linear-fit-age.csv')
    # )

    # row_corr_row(
    #     data_file1=pjoin(work_dir, 'HCPD_myelin_HCP_MMP1.csv'),
    #     cols1=rPath4, idx_col1=None,
    #     data_file2=pjoin(work_dir, 'HCPY_myelin-avg_HCP_MMP1.csv'),
    #     cols2=rPath4, idx_col2=None,
    #     out_file=pjoin(work_dir, 'HCPD-myelin_corr_HCPY-myelin-avg_rPath4.csv'),
    #     index=False, columns=['s1200_avg']
    # )
    # row_corr_row(
    #     data_file1=pjoin(work_dir, 'HCPD-myelin_rPath1-adjacent-minus.csv'),
    #     cols1=None, idx_col1=None,
    #     data_file2=pjoin(work_dir, 'HCPY-myelin-avg_rPath1-adjacent-minus.csv'),
    #     cols2=None, idx_col2=None,
    #     out_file=pjoin(work_dir, 'HCPD-myelin_corr_HCPY-myelin-avg_rPath1-adjacent-minus.csv'),
    #     index=False, columns=['s1200_avg']
    # )

    # col_operate_col(
    #     data_file=pjoin(work_dir, 'HCPD_myelin_HCP_MMP1.csv'),
    #     cols=rPath1, idx_col=None, operation_type='adjacent_pair',
    #     operation_method='-', index=False,
    #     out_file=pjoin(work_dir, 'HCPD-myelin_rPath1-adjacent-minus.csv')
    # )
    # col_operate_col(
    #     data_file=pjoin(work_dir, 'HCPY_thickness-avg_HCP_MMP1.csv'),
    #     cols=rPath1, idx_col=None, operation_type='adjacent_pair',
    #     operation_method='-', index=False,
    #     out_file=pjoin(work_dir, 'HCPY-thickness-avg_rPath1-adjacent-minus.csv')
    # )

    # mask_maps(
    #     data_file=s1200_avg_thickness,
    #     atlas_name='Cole_visual_LR', roi_name='R_cole_visual',
    #     out_file=pjoin(work_dir, 's1200_avg_thickness_mask-R_cole_visual.dscalar.nii')
    # )
    # mask_maps(
    #     data_file=pjoin(work_dir, 'HCPD_thickness_age-map-mean.dscalar.nii'),
    #     atlas_name='Cole_visual_LR', roi_name='R_cole_visual',
    #     out_file=pjoin(work_dir, 'HCPD_thickness_age-map-mean_mask-R_cole_visual.dscalar.nii')
    # )

    # map_operate_map(
    #     data_file1=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #     data_file2=s1200_avg_myelin, operation_method='-',
    #     out_file=pjoin(work_dir, 'HCPD-myelin_minus_HCPY-avg.dscalar.nii')
    # )
