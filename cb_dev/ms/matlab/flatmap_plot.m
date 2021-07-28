%
%
%

%%
% reset workspeace and variables
clc;
clear;
work_dir=fullfile('/nfs/e1/HCPD_CB/mri/');
%%
% start spm fmri
spm fmri;

%%
% vbm
map_file = fullfile('/nfs/e2/workingshop/masai/code/cb/python/vbm_masked.nii');
vbm_surf_data=suit_map2surf(map_file, 'space', 'FSL');

fig=figure;

suit_plotflatmap(vbm_surf_data,'cmap',hot)
