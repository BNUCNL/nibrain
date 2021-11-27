import os
import time
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import variation
from cxy_visual_dev.lib.predefine import proj_dir, get_rois,\
    Atlas, mmp_map_file, s1200_midthickness_R, s1200_midthickness_L,\
    MedialWall, hemi2stru, mmp_name2label, L_offset_32k, L_count_32k,\
    R_offset_32k, R_count_32k, s1200_MedialWall, R_OccipitalPole_32k,\
    L_OccipitalPole_32k, hemi2Hemi
from magicbox.io.io import save2cifti, CiftiReader, GiftiReader
from magicbox.algorithm.plot import plot_bar
from magicbox.algorithm.triangular_mesh import label_edge_detection,\
    get_n_ring_neighbor
from magicbox.algorithm.graph import bfs
from magicbox.algorithm.tool import calc_overlap

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
    找出左右视觉皮层边界
    存入dlabel文件时，分别标记为1和2，其它顶点为0
    存入pkl文件时，按照连线顺序排序
    存入dscalar文件时，左右边界的顶点值都是从1按连线顺序增加
    """
    # prepare parameters
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
    reader = CiftiReader(mmp_map_file)
    out_dlabel_file = pjoin(work_dir, 'MMP-vis3_border.dlabel.nii')
    out_pkl_file = pjoin(work_dir, 'MMP-vis3_border.pkl')
    out_dscalar_file = pjoin(work_dir, 'MMP-vis3_border.dscalar.nii')

    # calculating
    out_dlabel = np.zeros(reader.full_data.shape, np.uint8)
    out_pkl = {}
    out_dscalar = np.zeros(reader.full_data.shape, np.uint16)
    for hemi in hemis:
        # prepare visual mask
        mmp_map = reader.get_data(hemi2stru[hemi], True)[0]
        vis_mask = np.zeros_like(mmp_map, np.uint8)
        for roi in hemi2rois[hemi]:
            idx_vec = mmp_map == mmp_name2label[roi]
            vis_mask[idx_vec] = hemi2num[hemi]

        # detect border
        faces = GiftiReader(hemi2geo[hemi]).faces
        vis_edge = label_edge_detection(vis_mask, faces, 'inner')

        # get and sort border vertices
        border_vertices = np.where(vis_edge == hemi2num[hemi])[0]
        border_neighbor_list = get_n_ring_neighbor(faces, mask=vis_edge)
        border_vertices_sort = []  # 按照连线顺序将顶点加入该列表
        pending_queue = []  # 过渡区
        for border_vtx in border_vertices:
            # 找到只有两个近邻的顶点作为起始点
            # 随机删掉一个近邻，留下另一个作为延伸的方向
            if len(border_neighbor_list[border_vtx]) == 2:
                pending_queue.append(border_vtx)
                border_neighbor_list[border_vtx].pop()
                break
        while pending_queue:
            # 按顺序处理完近邻后再加入border_vertices_sort
            for border_vtx in border_vertices:
                border_neighbor_list[border_vtx].difference_update(pending_queue)
            pending_queue_tmp = []
            for pending_vtx in pending_queue:
                pending_queue_tmp.extend(border_neighbor_list[pending_vtx])
            border_vertices_sort.extend(pending_queue)
            pending_queue = pending_queue_tmp
        assert sorted(border_vertices_sort) == sorted(border_vertices)

        # collect data
        offset, count = hemi2loc[hemi]
        idx2vtx = reader.get_data(hemi2stru[hemi])[-1]
        out_dlabel[0, offset:(offset+count)] = vis_edge[idx2vtx]
        out_pkl[hemi] = border_vertices_sort
        vis_edge = vis_edge.astype(np.uint16)
        vis_edge[border_vertices_sort] = \
            np.arange(1, len(border_vertices_sort) + 1)
        out_dscalar[0, offset:(offset+count)] = vis_edge[idx2vtx]

    # save out
    bm_list = reader.brain_models()
    save2cifti(out_dlabel_file, out_dlabel, bm_list, label_tables=[lbl_tab])
    pkl.dump(out_pkl, open(out_pkl_file, 'wb'))
    save2cifti(out_dscalar_file, out_dscalar, bm_list)


def get_radial_line():
    """
    找出从枕极到所有视觉皮层边界上的顶点的最短路径

    产生一个二层嵌套列表（存为pickle）
    第一层的长度等于边界顶点数量
    每个子列表就是从枕极到某个边界顶点的路径，其中第一个元素
    是枕极的顶点号，最后一个元素是该边界顶点的顶点号。
    """
    # prepare parameters
    hemi = 'rh'
    fpath = pjoin(work_dir, 'MMP-vis3_border.pkl')
    out_file = pjoin(work_dir, f'MMP-vis3_RadialLine-{hemi2Hemi[hemi]}.pkl')
    hemi2geo = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    hemi2origin = {
        'lh': L_OccipitalPole_32k,
        'rh': R_OccipitalPole_32k}

    # get border
    border_vertices = pkl.load(open(fpath, 'rb'))[hemi]
    n_vtx = len(border_vertices)

    # get neighbors
    faces = GiftiReader(hemi2geo[hemi]).faces
    faces = MedialWall().remove_from_faces(hemi, faces)
    neighbors_list = get_n_ring_neighbor(faces)

    # calculating
    lines = []
    for border_idx, border_vtx in enumerate(border_vertices, 1):
        time1 = time.time()
        line = bfs(neighbors_list, hemi2origin[hemi], border_vtx)
        lines.append(line)
        print(f'Finished {border_idx}/{n_vtx}: '
              f'cost {time.time() - time1} seconds.')

    # save out
    pkl.dump(lines, open(out_file, 'wb'))


def simplify_radial_line():
    """
    由于有很多line是重叠在一起的，为了减少不必要的计算量，要去掉一些冗余的line
    两两计算重叠的点数占较短的line的百分比，如果超过阈限就去掉短的。
    随后继续辅以间隔N个边界点取一条line的方式使其变得稀疏

    存出更新后的二层嵌套列表
    """
    # prepare parameters
    thr = 0.9
    N = 1
    hemi = 'rh'
    fname = f'MMP-vis3_RadialLine-{hemi2Hemi[hemi]}'
    fpath = pjoin(work_dir, f'{fname}.pkl')
    out_file = pjoin(work_dir, f'{fname}_thr{int(thr*100)}_N{N}.pkl')

    # loading
    lines = pkl.load(open(fpath, 'rb'))

    # 根据重叠度去除冗余
    removed_indices = []
    for i, line1 in enumerate(lines[:-1]):
        if i in removed_indices:
            continue
        for j, line2 in enumerate(lines[i+1:], i+1):
            if j in removed_indices:
                continue
            if len(line1) < len(line2):
                overlap = calc_overlap(line1, line2, index='percent')
                if overlap > thr:
                    removed_indices.append(i)
                    break
            else:
                overlap = calc_overlap(line2, line1, index='percent')
                if overlap > thr:
                    removed_indices.append(j)
                    continue
    lines = [line for idx, line in enumerate(lines)
             if idx not in removed_indices]

    # 间隔N个点取一次line，使其稀疏
    if N > 0:
        reserved_indices = np.arange(0, len(lines), N+1)
        lines = [line for idx, line in enumerate(lines)
                 if idx in reserved_indices]

    # save out
    pkl.dump(lines, open(out_file, 'wb'))


def line_pkl2cii():
    """
    把存在pickle文件中的radial line存到cifti文件中，用以可视化
    存一个dlabel文件，在其中把所有line标记为1，其它为0
    存一个dscalar文件，在其中记录每个顶点被line经过的频率
    """
    hemi = 'rh'
    fname = f'MMP-vis3_RadialLine-{hemi2Hemi[hemi]}_thr90_N1'
    fpath = pjoin(work_dir, f'{fname}.pkl')
    out_dlabel_file = pjoin(work_dir, f'{fname}.dlabel.nii')
    out_dscalar_file = pjoin(work_dir, f'{fname}.dscalar.nii')
    hemi2label = {
        'lh': 'L_radial_line',
        'rh': 'R_radial_line'}

    # loading
    lines = pkl.load(open(fpath, 'rb'))
    n_line = len(lines)

    # get brain model
    bm = CiftiReader(s1200_MedialWall).brain_models([hemi2stru[hemi]])[0]
    bm.index_offset = 0

    # calculating
    out_dlabel = np.zeros((1, bm.index_count), np.uint8)
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    lbl_tab[1] = nib.cifti2.Cifti2Label(1, hemi2label[hemi], 0, 0, 0, 1)
    out_dscalar = np.zeros((1, bm.index_count), np.float64)
    for line in lines:
        out_dlabel[0, line] = 1
        out_dscalar[0, line] += 1
    out_dscalar = out_dscalar / n_line

    # save out
    save2cifti(out_dlabel_file, out_dlabel, [bm], label_tables=[lbl_tab])
    save2cifti(out_dscalar_file, out_dscalar, [bm])
# 以枕极为原点，以圆环代表层，以长条代表跨层<<<


if __name__ == '__main__':
    # get_vis_border()
    # get_radial_line()
    simplify_radial_line()
    line_pkl2cii()
