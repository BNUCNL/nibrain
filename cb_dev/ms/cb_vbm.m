% The workflow of caculating VBM
% Created on Mon Jul 5 15:48:49 2021
% @author: Sai Ma

%%
% reset workspeace and variables
clc;
clear;
hcpd_path=fullfile('/nfs/e1/HCPD/fmriresults01/');
work_dir=fullfile('/nfs/e1/HCPD_CB/mri/');

%%
% read all subject id from subject_list.csv
subid_file = fopen(fullfile(work_dir,'subject_list.csv'));
subject_list=textscan(subid_file,'%s','Delimiter',',');
fclose(subid_file);
subject_id=subject_list{1,1};

%%
% start SPM fmri for SUIT
spm fmri;

%%
% caculate VBM using SUIT
for id=1:length(subject_id)
    % make subject folders
    mkdir(fullfile(work_dir,subject_id{id}));
    mkdir(fullfile(work_dir,subject_id{id},'anat'));
    mkdir(fullfile(work_dir,subject_id{id},'func'));
    % copy native T1w & T2w images
    copyfile(fullfile(hcpd_path,subject_id{id},'T1w','T1w_acpc_dc.nii.gz'),fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    copyfile(fullfile(hcpd_path,subject_id{id},'T1w','T2w_acpc_dc.nii.gz'),fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
    % unzip nifti files
    gunzip(fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    gunzip(fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
    % isolate cerebellum from native brain
    suit_isolate_seg({fullfile(work_dir,subject_id{id},'anat','T1w.nii'),fullfile(work_dir,subject_id{id},'anat','T2w.nii')})
    % normalize native cerebellum
    job_nor.subjND.gray={fullfile(work_dir,subject_id{id},'anat','T1w_seg1.nii')};
    job_nor.subjND.white={fullfile(work_dir,subject_id{id},'anat','T1w_seg2.nii')};
    job_nor.subjND.isolation={fullfile(work_dir,subject_id{id},'anat','c_T1w_pcereb.nii')};
    suit_normalize_dartel(job_nor)
    % caculate VBM
    job_vbm.subj.affineTr={fullfile(work_dir,subject_id{id},'anat','Affine_T1w_seg1.mat')};
    job_vbm.subj.flowfield={fullfile(work_dir,subject_id{id},'anat','u_a_T1w_seg1.nii')};
    job_vbm.subj.resample={fullfile(work_dir,subject_id{id},'anat','T1w_seg1.nii')};
    job_vbm.subj.mask={fullfile(work_dir,subject_id{id},'anat','c_T1w_pcereb.nii')};
    job_vbm.jactransf=1;
    suit_reslice_dartel(job_vbm)
    % delete native images
    delete(fullfile(work_dir,subject_id{id},'anat','T1w.nii'))
    delete(fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    delete(fullfile(work_dir,subject_id{id},'anat','T2w.nii'))
    delete(fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
end