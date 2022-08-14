import os
from os.path import join as pjoin
from cxy_visual_dev.lib.predefine import proj_dir,\
    dataset_name2info, s1200_1096_thickness, s1200_1096_myelin,\
    s1200_avg_thickness, s1200_avg_myelin, get_rois
from cxy_visual_dev.lib.algo import ROI_analysis_on_PC,\
    calc_map_corr, vtx_corr_col, polyfit, row_corr_row,\
    col_operate_col, map_operate_map

work_dir = pjoin(proj_dir, 'analysis/structure')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


if __name__ == '__main__':
    # HCP_MMP1 atlas 包含 Cole_visual_ROI

    # ROI_analysis_on_PC(
    #     data_file=pjoin(proj_dir, 'data/HCP/HCPD_thickness_4mm.dscalar.nii'),
    #     pca_file=pjoin(work_dir, 'HCPD_thickness_4mm_R_cole_visual_PCA-vtx.pkl'),
    #     pc_num=1, mask_atlas='Cole_visual_LR', mask_name='R_cole_visual',
    #     roi_atlas='Cole_visual_ROI',
    #     out_file=pjoin(work_dir, 'HCPD_thickness_4mm_R_cole_visual_PCA-vtx-PC1.csv')
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

    # map_operate_map(
    #     data_file1=pjoin(proj_dir, 'data/HCP/HCPD_myelin.dscalar.nii'),
    #     data_file2=s1200_avg_myelin, operation_method='-',
    #     out_file=pjoin(work_dir, 'HCPD-myelin_minus_HCPY-avg.dscalar.nii')
    # )
