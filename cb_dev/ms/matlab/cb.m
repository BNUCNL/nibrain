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
% caculate VBM & Myelin
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
    % caculate VBM
    job_vbm.subj.affineTr={fullfile(work_dir,subject_id{id},'anat','Affine_T1w_seg1.mat')};
    job_vbm.subj.flowfield={fullfile(work_dir,subject_id{id},'anat','u_a_T1w_seg1.nii')};
    job_vbm.subj.resample={fullfile(work_dir,subject_id{id},'anat','T1w_seg1.nii')};
    job_vbm.subj.mask={fullfile(work_dir,subject_id{id},'anat','c_T1w_pcereb.nii')};
    job_vbm.jactransf=1;
    suit_reslice_dartel(job_vbm)
    % caculate Myelin
    % read T1w & T2w data
    t1w_img=spm_vol(fullfile(work_dir,subject_id{id},'anat','T1w.nii'));
    t2w_img=spm_vol(fullfile(work_dir,subject_id{id},'anat','T2w.nii'));
    t1w_data=spm_read_vols(t1w_img);
    t2w_data=spm_read_vols(t2w_img);
    % T1w devided by T2w
    myelin_data=t1w_data./t2w_data;
    % save as nifti
    t1w_img.fname=fullfile(work_dir,subject_id{id},'anat','myelin_map.nii');
    spm_write_vol(t1w_img,myelin_data);
    % resample to SUIT speace
    job_myelin.subj.affineTr={fullfile(work_dir,subject_id{id},'anat','Affine_T1w_seg1.mat')};
    job_myelin.subj.flowfield={fullfile(work_dir,subject_id{id},'anat','u_a_T1w_seg1.nii')};
    job_myelin.subj.resample={fullfile(work_dir,subject_id{id},'anat','myelin_map.nii')};
    job_myelin.subj.mask={fullfile(work_dir,subject_id{id},'anat','c_T1w_pcereb.nii')};
    suit_reslice_dartel(job_myelin)
    % show flat map
    % map=suit_map2surf(fullfile(work_dir,subject_id{id},'anat','wdmyelin_map.nii'));
    % suit_plotflatmap(map)
    % delete native images
    delete(fullfile(work_dir,subject_id{id},'anat','T1w.nii'))
    delete(fullfile(work_dir,subject_id{id},'anat','T1w.nii.gz'))
    delete(fullfile(work_dir,subject_id{id},'anat','T2w.nii'))
    delete(fullfile(work_dir,subject_id{id},'anat','T2w.nii.gz'))
    % delete(fullfile(work_dir,subject_id{id},'anat','myelin_map.nii'))
end

