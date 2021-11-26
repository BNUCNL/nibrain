import os
import numpy as np
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import variation
from cxy_visual_dev.lib.predefine import proj_dir, get_rois,\
    Atlas, mmp_map_file, s1200_midthickness_R, s1200_midthickness_L,\
    MedialWall, hemi2stru, mmp_name2label, L_offset_32k, L_count_32k,\
    R_offset_32k, R_count_32k
from magicbox.io.io import save2cifti, CiftiReader, GiftiReader
from magicbox.algorithm.plot import plot_bar
from magicbox.algorithm.triangular_mesh import label_edge_detection

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'variation')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_variation1():
    """
    对于PC1，我们已经认定它是从后到前渐变的梯度，并且用以枕极为锚点的距离作为PC1的理想模型。
    依据该距离分段，在同一距离段内的顶点属于一个层级。可以看到PC1的主要变异是层级间的渐变。
    反映的是从低级视觉到高级视觉这个整体的功能分化。

    PC2作为去除PC1之后的主要成分，开始呈现出层级内的变异，但看起来也存在层级间的变异,
    只是不是渐变，而是和层级内变异一样的波动着的变异。或许和局部功能分化有关。

    分析思路：
    将离枕极的距离从最小值到最大值分为N个层级（N等分），计算第i层的变异（var_i）和均值(mean_i)
    计算N个均值的变异作为层间变异（var_between），计算N个变异的均值作为层内变异（var_within）

    预期结果：
    PC2的层内变异要远大于PC1
    PC1的层间变异要远大于层内
    PC2层内和层间变异都较大，最好是层内大于层间
    """
    # prepare parameters
    n_segment = 10
    method = 'CV3'  # CV1, CV2, CV3, std
    n_pc = 2  # 前N个成分
    pc_names = ('C1', 'C2')
    title = f'segment{n_segment}_{method}'
    out_file1 = pjoin(work_dir, f'{title}_1.jpg')
    out_file2 = pjoin(work_dir, f'{title}_2.jpg')

    # prepare mask
    mask = Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-R'))[0]

    # prepare geodesic distance and segment boundaries
    gdist_file = pjoin(anal_dir, 'gdist/gdist_src-OccipitalPole.dscalar.nii')
    # gdist_file = pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii')  # 效果还是以枕极为锚点比较好
    gdist_map = nib.load(gdist_file).get_fdata()[0, mask]
    min_gdist, max_gdist = np.min(gdist_map), np.max(gdist_map)
    segment_boundaries = np.linspace(min_gdist, max_gdist, n_segment + 1)

    # prepare PC maps
    pc_file = pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii')
    pc_maps = nib.load(pc_file).get_fdata()[:n_pc, mask]
    if method == 'CV1':
        var_func = variation
    elif method == 'CV2':
        # 每个PC都减去各自的最小值
        pc_maps = pc_maps - np.min(pc_maps, 1, keepdims=True)
        var_func = variation
    elif method == 'CV3':
        # 用绝对值计算作为分母的均值（标准差还是基于原数据计算）
        def var_func(arr, axis=None, ddof=0):
            var = np.std(arr, axis, ddof=ddof) /\
                np.mean(np.abs(arr), axis)
            return var
    elif method == 'std':
        var_func = np.std
    else:
        raise ValueError

    # calculating
    segment_means = np.zeros((n_pc, n_segment), np.float64)
    segment_vars = np.zeros((n_pc, n_segment), np.float64)
    for s_idx, s_boundary in enumerate(segment_boundaries[:-1]):
        e_idx = s_idx + 1
        e_boundary = segment_boundaries[e_idx]
        if e_idx == n_segment:
            segment_mask = np.logical_and(
                gdist_map >= s_boundary, gdist_map <= e_boundary)
        else:
            segment_mask = np.logical_and(
                gdist_map >= s_boundary, gdist_map < e_boundary)
        segments = pc_maps[:, segment_mask]
        segment_means[:, s_idx] = np.mean(segments, 1)
        segment_vars[:, s_idx] = var_func(segments, 1)
    var_along = var_func(segment_means, 1)
    var_vertical = np.mean(segment_vars, 1)

    plot_bar(np.array([var_along, var_vertical]), figsize=(4, 4),
             label=('var_along', 'var_vertical'), xticklabel=pc_names,
             ylabel='variation', title=title, mode=out_file1)
    plot_bar(segment_vars, figsize=(8, 4), label=pc_names,
             xticklabel=np.arange(1, n_segment+1),
             ylabel='variation', title=title, mode=out_file2)


# >>>以枕极为原点，以圆环代表层，以长条代表跨层
def get_vis_border():
    """
    找出左视觉皮层边界标记为1，
    找出右视觉皮层边界标记为2。
    """
    hemis = ('lh', 'rh')
    hemi2num = {'lh': 1, 'rh': 2}
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    lbl_tab[1] = nib.cifti2.Cifti2Label(1, 'L_vis_border', 1, 0, 0, 1)
    lbl_tab[2] = nib.cifti2.Cifti2Label(2, 'R_vis_border', 0, 0, 1, 1)
    hemi2geo = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    hemi2loc = {
        'lh': (L_offset_32k, L_count_32k),
        'rh': (R_offset_32k, R_count_32k)}
    hemi2rois = {
        'lh': get_rois('MMP-vis3-L'),
        'rh': get_rois('MMP-vis3-R')}
    out_file = pjoin(work_dir, 'MMP-vis3_border.dlabel.nii')

    reader = CiftiReader(mmp_map_file)
    data = np.zeros(reader.full_data.shape, np.uint8)
    for hemi in hemis:
        idx2vtx = reader.get_data(hemi2stru[hemi])[-1]
        mmp_map = reader.get_data(hemi2stru[hemi], True)[0]
        vis_mask = np.zeros_like(mmp_map, np.uint8)
        for roi in hemi2rois[hemi]:
            idx_vec = mmp_map == mmp_name2label[roi]
            vis_mask[idx_vec] = hemi2num[hemi]

        faces = GiftiReader(hemi2geo[hemi]).faces

        vis_edge = label_edge_detection(vis_mask, faces, 'inner')
        offset, count = hemi2loc[hemi]
        data[0, offset:(offset+count)] = vis_edge[idx2vtx]
        lbl_tab[hemi2num[hemi]]

    save2cifti(out_file, data, reader.brain_models(), label_tables=[lbl_tab])
# 以枕极为原点，以圆环代表层，以长条代表跨层<<<


if __name__ == '__main__':
    get_vis_border()
