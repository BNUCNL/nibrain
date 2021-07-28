% learn SUIT
% 2021-6-29
% author: masai

%%
addpath /usr/local/neurosoft/matlab_tools/spm12
addpath /usr/local/neurosoft/matlab_tools/spm12/compat
addpath /usr/local/neurosoft/matlab_tools/spm12/toolbox/DARTEL
addpath /usr/local/neurosoft/matlab_tools/spm12/toolbox/suit

%%
% isolation
% input
%   <T1w_filename>.nii (Compressed NIfTI files (.nii.gz) are not supported)
% output
%   1. c_<T1w_filename>.nii 
%         The cropped anatomical containing the cerebellum
%   2. c_<T1w_filename>_pcereb.nii 
%         The binarized mask after thresholded(p=0.2)
%   3. <T1w_filename>_seg1.nii
%         Probabilities map of gray matter
%   4. <T1w_filename>_seg1.nii
%         Probabilities map of gray matter

suit_isolate_seg({'T2w_acpc_dc.nii'})

%%
% Normalization
% input
%   job(object)
% output
%   1. Affine matrix for the linear part of the normalization
%   2. The non-linear flowfield (a 4-D nifti)

job.subjND.gray={'T1w_acpc_dc_seg1.nii'};
job.subjND.white={'T1w_acpc_dc_seg2.nii'};
job.subjND.isolation={'c_T1w_acpc_dc_pcereb.nii'};

suit_normalize_dartel(job)

%%
% Re-slicing anat (VBM)
% input
%   job(object)
% output

job.subj.affineTr={'Affine_T1w_acpc_dc_seg1.mat'};
job.subj.flowfield={'u_a_T1w_acpc_dc_seg1.nii'};
job.subj.resample={'T1w_acpc_dc_seg1.nii'};
job.subj.mask={'c_T1w_acpc_dc_pcereb.nii'};
job.jactransf=1;

suit_reslice_dartel(job)

%%
% VBM

job.subj.affineTr={'Affine_T1w_acpc_dc_seg1.mat'};
job.subj.flowfield={'u_a_T1w_acpc_dc_seg1.nii'};
job.subj.resample={'T1w_acpc_dc_seg1.nii'};
job.subj.mask={'c_T1w_acpc_dc_pcereb.nii'};
job.jactransf=1;

suit_reslice_dartel(job)



