% The workflow of resample alff & falff
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
for id=1:length(subject_id)
    % copy native images
    copyfile(fullfile(hcpd_path,subject_id{id},'T1w','T1wDividedByT2w.nii.gz'),fullfile(work_dir,subject_id{id},'anat','myelin.nii.gz'))
    % unzip nifti files
    gunzip(fullfile(work_dir,subject_id{id},'anat','myelin.nii.gz'))
    % resample myelin data to SUIT speace
    job_myelin.subj.affineTr={fullfile(work_dir,subject_id{id},'anat','Affine_T1w_seg1.mat')};
    job_myelin.subj.flowfield={fullfile(work_dir,subject_id{id},'anat','u_a_T1w_seg1.nii')};
    job_myelin.subj.resample={fullfile(work_dir,subject_id{id},'anat','myelin.nii')};
    job_myelin.subj.mask={fullfile(work_dir,subject_id{id},'anat','c_T1w_pcereb.nii')};
    suit_reslice_dartel(job_myelin)
    % delete native myelin image
    delete(fullfile(work_dir,subject_id{id},'anat','myelin.nii.gz'))
    delete(fullfile(work_dir,subject_id{id},'anat','myelin.nii'))
end

%% 
for id=1:length(subject_id)
    delete(fullfile(work_dir,subject_id{id},'anat','myelin.nii.gz'))
    delete(fullfile(work_dir,subject_id{id},'anat','myelin.nii'))
end