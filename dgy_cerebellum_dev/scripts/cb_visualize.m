addpath /usr/local/neurosoft/matlab_tools/spm12
addpath /usr/local/neurosoft/matlab_tools/spm12/compat
addpath /usr/local/neurosoft/matlab_tools/spm12/toolbox/DARTEL
addpath /usr/local/neurosoft/matlab_tools/spm12/toolbox/suit

map_dir = '/nfs/s2/userhome/dengguangyu/workingdir/HCP/data/myelin_grad';
output_dir = '/nfs/s2/userhome/dengguangyu/workingdir/HCP/figures/myelin_grad';

n = 3;
for i = 6: 22
    map_file = fullfile(map_dir, sprintf('CB-Mean-%d.nii', i));
    map = suit_map2surf(map_file, 'space', 'FSL');

    fig = figure;
    suit_plotflatmap(map, 'cmap', jet, 'cscale', [1, 2]);
    % suit_plotflatmap(map, 'cmap',jet)
    % suit_plotflatmap(map, 'cmap', gray)
    save_path = fullfile(output_dir, sprintf('CB-Mean-%d.jpg', i));
    saveas(fig, save_path, 'jpg');
    close(fig);
end