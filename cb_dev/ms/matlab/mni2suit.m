% resample MNI template to SUIT space 
% Created on Mon Jul 12 15:48:49 2021
% @author: Sai Ma

%%
% reset workspeace and variables
clc;
clear;
template_path=fullfile('/usr/local/neurosoft/fsl5.0.10/data/standard/MNI152_T1_1mm.nii.gz');
work_dir=fullfile('/nfs/e1/HCPD_CB/mri/');

%%
% start SPM fmri for SUIT
spm fmri;

%%
% make MNI template folders
mkdir(fullfile(work_dir,'MNI_to_SUIT'));
%%
% copy native images
copyfile(template_path,fullfile(work_dir,'MNI_to_SUIT','MNI_T1_1mm.nii.gz'));
%%
% unzip native images
gunzip(fullfile(work_dir,'MNI_to_SUIT','MNI_T1_1mm.nii.gz'));
%%
% isolate cerebellum from native brain
suit_isolate_seg({fullfile(work_dir,'MNI_to_SUIT','MNI_T1_1mm.nii')})
%%
% normalize native cerebellum
job_nor.subjND.gray={fullfile(work_dir,'MNI_to_SUIT','MNI_T1_1mm_seg1.nii')};
job_nor.subjND.white={fullfile(work_dir,'MNI_to_SUIT','MNI_T1_1mm_seg2.nii')};
job_nor.subjND.isolation={fullfile(work_dir,'MNI_to_SUIT','c_MNI_T1_1mm_pcereb.nii')};
suit_normalize_dartel(job_nor)
%%
% delete native images
delete(fullfile(work_dir,'MNI_to_SUIT','MNI_T1_1mm.nii.gz'));
delete(fullfile(work_dir,'MNI_to_SUIT','MNI_T1_1mm.nii'));
