% caculate VBM & Mylin of cerebellum using SUIT
% 2021-6-30
% Author: Sai Ma

%%
% clear variables
clc;
clear;

%%
% get subjid list
hcpd_path=fullfile('/nfs/e1/HCPD/fmriresults01/');
hcpd_dir=dir(hcpd_path);
subject_id={hcpd_dir.name};
subject_id(strcmp(subject_id,'.'))=[];
subject_id(strcmp(subject_id,'..'))=[];
subject_id(strcmp(subject_id,'manifests'))=[];

%%
% prepare data using SUIT
work_dir=fullfile('/nfs/e1/HCPD_CB/mri/');
% for id=1:length(subject_id)
for id=1:2
    mkdir(fullfile(work_dir,subject_id{id}));
    mkdir(fullfile(work_dir,subject_id{id},'anat'));
    mkdir(fullfile(work_dir,subject_id{id},'func'));
    % copy native images
    copyfile(fullfile(hcpd_path,subject_id{id},'T1w','T1w_acpc_dc.nii.gz'),fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    copyfile(fullfile(hcpd_path,subject_id{id},'T1w','T2w_acpc_dc.nii.gz'),fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
    % unzip native images
    gunzip(fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    gunzip(fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
    % isolate cerebellum from native brain
    suit_isolate_seg({fullfile(work_dir,subject_id{id},'anat','T1w.nii'),fullfile(work_dir,subject_id{id},'anat','T2w.nii')})
    % normalize native cerebellum
    % T1w
    job.subjND.gray={fullfile(work_dir,subject_id{id},'anat','T1w_seg1.nii')};
    job.subjND.white={fullfile(work_dir,subject_id{id},'anat','T1w_seg2.nii')};
    job.subjND.isolation={fullfile(work_dir,subject_id{id},'anat','c_T1w_pcereb.nii')};
    suit_normalize_dartel(job)
    % T2w
    job.subjND.gray={fullfile(work_dir,subject_id{id},'anat','T2w_seg1.nii')};
    job.subjND.white={fullfile(work_dir,subject_id{id},'anat','T2w_seg2.nii')};
    job.subjND.isolation={fullfile(work_dir,subject_id{id},'anat','c_T2w_pcereb.nii')};
    suit_normalize_dartel(job)
    % reslicing
    % T1w
    job.subj.affineTr={fullfile(work_dir,subject_id{id},'anat','Affine_T1w_seg1.mat')};
    job.subj.flowfield={fullfile(work_dir,subject_id{id},'anat','u_a_T1w_seg1.nii')};
    job.subj.resample={fullfile(work_dir,subject_id{id},'anat','T1w_seg1.nii')};
    job.subj.mask={fullfile(work_dir,subject_id{id},'anat','c_T1w_pcereb.nii')};
    job.jactransf=1;
    suit_reslice_dartel(job)
    % T2w
    job.subj.affineTr={fullfile(work_dir,subject_id{id},'anat','Affine_T2w_seg1.mat')};
    job.subj.flowfield={fullfile(work_dir,subject_id{id},'anat','u_a_T2w_seg1.nii')};
    job.subj.resample={fullfile(work_dir,subject_id{id},'anat','T2w_seg1.nii')};
    job.subj.mask={fullfile(work_dir,subject_id{id},'anat','c_T2w_pcereb.nii')};
    job.jactransf=1;
    suit_reslice_dartel(job)
    % delete native images
    delete(fullfile(work_dir,subject_id{id},'anat','T1w.nii'))
    delete(fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    delete(fullfile(work_dir,subject_id{id},'anat','T2w.nii'))
    delete(fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
end
