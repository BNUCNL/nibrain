import os
import time
import gdist
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import variation, sem
from matplotlib import pyplot as plt
from cxy_visual_dev.lib.predefine import proj_dir, get_rois,\
    Atlas, s1200_midthickness_R, s1200_midthickness_L,\
    MedialWall, hemi2stru, s1200_MedialWall, hemi2Hemi
from magicbox.io.io import save2cifti, CiftiReader, GiftiReader
from magicbox.algorithm.plot import plot_bar
from magicbox.algorithm.triangular_mesh import get_n_ring_neighbor
from magicbox.algorithm.graph import bfs
from magicbox.algorithm.tool import calc_overlap
from magicbox.stats import calc_coef_var, calc_cqv

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'variation')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def get_var_func(method):
    """
    选一种计算变异的方式

    Args:
        method (str):

    Returns:
        callable: variation function
    """
    if method == 'CV1':
        # std/mean
        var_func = variation

    elif method == 'CV3':
        # std/取绝对值之后的mean
        var_func = calc_coef_var

    elif method == 'CV4':
        # std/|mean|
        def var_func(arr, axis):
            return np.abs(variation(arr, axis))

    elif method == 'CV5':
        # std/|mean|/n_sample
        def var_func(arr, axis):
            var = np.abs(variation(arr, axis))
            var = var / arr.shape[axis]
            return var

    elif method == 'std':
        # std
        var_func = np.std

    elif method == 'std/n_vtx':
        # std/n_sample
        def var_func(arr, axis, ddof=0):
            var = np.std(arr, axis, ddof=ddof) /\
                arr.shape[axis]
            return var

    elif method == 'CQV':
        # (Q3-Q1)/(Q3+Q1)
        var_func = calc_cqv

    elif method == 'CQV1':
        # |(Q3-Q1)/(Q3+Q1)|
        def var_func(arr, axis):
            return np.abs(calc_cqv(arr, axis))

    else:
        raise ValueError('not supported method')

    return var_func


# ===按照距距状沟的距离分段，代表各个层级（功能特异化的方向）===
# ===以从距状沟辐射出来的长条代表跨层，（层级加工的方向）===
def get_radial_line(hemi, cs_type):
    """
    找出从视觉皮层边界上各顶点到距状沟的最短路径，具体做法：
    对于某个视觉皮层边界上的顶点border_vtx，计算其和距状沟所有顶点的测地距离，
    选出距离最小的那个距状沟顶点cs_vtx，然后用我的BFS算法获取cs_vtx和border_vtx之间的路径。

    产生一个二层嵌套列表（存为pickle）
    第一层的长度等于边界顶点数量
    每个子列表就是从距状沟辐射到某个边界顶点的路径，其中第一个元素
    是距状沟上的起始顶点号，最后一个元素是该边界顶点的顶点号。
    """
    # prepare parameters
    Hemi = hemi2Hemi[hemi]
    if cs_type == 'CS1':
        cs_file = pjoin(proj_dir, f'data/{Hemi}_CalcarineSulcus.label')
    elif cs_type == 'CS2':
        cs_file = pjoin(proj_dir, f'data/{Hemi}_CalcarineSulcus_split.label')
    else:
        raise ValueError(cs_type)
    border_file = pjoin(work_dir, 'MMP-vis3_border.pkl')
    hemi2geo_file = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    out_file = pjoin(work_dir, f'MMP-vis3_RadialLine-{cs_type}_{Hemi}.pkl')

    # get vertices
    cs_vertices = nib.freesurfer.read_label(cs_file)
    cs_vertices = np.asarray(cs_vertices, np.int32)
    border_vertices = pkl.load(open(border_file, 'rb'))[hemi]
    n_border_vtx = len(border_vertices)

    # get geometry information
    gii = GiftiReader(hemi2geo_file[hemi])
    coords = gii.coords.astype(np.float64)
    faces = MedialWall().remove_from_faces(hemi, gii.faces)
    faces = faces.astype(np.int32)
    neighbors_list = get_n_ring_neighbor(faces)

    # calculating
    lines = []
    for border_idx, border_vtx in enumerate(border_vertices, 1):
        time1 = time.time()
        gdists = gdist.compute_gdist(
            coords, faces, np.array([border_vtx], dtype=np.int32), cs_vertices)
        cs_vtx = cs_vertices[gdists.argmin()]
        line = bfs(neighbors_list, cs_vtx, border_vtx)
        lines.append(line)
        print(f'Finished {border_idx}/{n_border_vtx}: '
              f'cost {time.time() - time1} seconds.')

    # save out
    pkl.dump(lines, open(out_file, 'wb'))


def get_radial_line1(hemi, cs_type):
    """
    为每个视觉皮层边界点找出其和所有距状沟顶点的BFS路径，然后取最短的那个。
    如果路径又路过了距状沟的顶点，那这条路径就改成从最后路过的那个距状沟顶点开始！

    产生一个二层嵌套列表（存为pickle）
    第一层的长度等于边界顶点数量
    每个子列表就是从距状沟辐射到某个边界顶点的路径，其中第一个元素
    是距状沟上的起始顶点号，最后一个元素是该边界顶点的顶点号。
    """
    # prepare parameters
    Hemi = hemi2Hemi[hemi]
    if cs_type == 'CS1':
        cs_file = pjoin(proj_dir, f'data/{Hemi}_CalcarineSulcus.label')
    elif cs_type == 'CS2':
        cs_file = pjoin(proj_dir, f'data/{Hemi}_CalcarineSulcus_split.label')
    else:
        raise ValueError(cs_type)
    border_file = pjoin(work_dir, 'MMP-vis3_border.pkl')
    hemi2geo_file = {
        'lh': s1200_midthickness_L,
        'rh': s1200_midthickness_R}
    out_file = pjoin(work_dir, f'MMP-vis3_RadialLine1-{cs_type}_{Hemi}.pkl')

    # get vertices
    cs_vertices = nib.freesurfer.read_label(cs_file)
    border_vertices = pkl.load(open(border_file, 'rb'))[hemi]
    n_border_vtx = len(border_vertices)

    # get geometry information
    gii = GiftiReader(hemi2geo_file[hemi])
    faces = MedialWall().remove_from_faces(hemi, gii.faces)
    neighbors_list = get_n_ring_neighbor(faces)

    # calculating
    lines = []
    for border_idx, border_vtx in enumerate(border_vertices, 1):
        time1 = time.time()
        lines_tmp = []
        line_lens_tmp = []
        for cs_vtx in cs_vertices:
            line = bfs(neighbors_list, cs_vtx, border_vtx)
            line_in_cs = np.isin(line, cs_vertices)
            line_indices_in_cs = np.where(line_in_cs)[0]
            start_idx = line_indices_in_cs[-1]
            assert start_idx == line_indices_in_cs.max()
            line = line[start_idx:]
            lines_tmp.append(line)
            line_lens_tmp.append(len(line))
        lines.append(lines_tmp[np.argmin(line_lens_tmp)])
        print(f'Finished {border_idx}/{n_border_vtx}: '
              f'cost {time.time() - time1} seconds.')
    print('n_line:', len(lines))

    # save out
    pkl.dump(lines, open(out_file, 'wb'))


def simplify_radial_line(hemi, line_type, cs_type, thr, N):
    """
    由于有很多line是重叠在一起的，为了减少不必要的计算量，要去掉一些冗余的line
    两两计算重叠的点数占较短的line的百分比，如果超过阈限就去掉短的。
    随后继续辅以间隔N个边界点取一条line的方式使其变得稀疏

    存出更新后的二层嵌套列表
    """
    # prepare parameters
    fname = f'MMP-vis3_{line_type}-{cs_type}_{hemi2Hemi[hemi]}'
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


def line_pkl2cii(hemi, fname):
    """
    把存在pickle文件中的radial line存到cifti文件中，用以可视化
    存一个dlabel文件，在其中把所有line标记为1，其它为0
    存一个dscalar文件，在其中记录每个顶点被line经过的频率
    """
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


def prepare_ring_bar(hemi, width, mask, line_file, line_gdist_file,
                     CS_gdist_file, out_bar_file, out_ring_file, merge_last=False):
    """
    产生厚度为width的层和长条
    长条的数量由事先精简好的line的数量决定
    层的数量由依据厚度分段得到的结果决定
    据事先观察：距CS1的距离是0~88.1219，距CS2的距离是0~80.2524
    以5为厚度的话，最后剩下的分别是3.1219/0.2524毫米。
    因此CS1最后剩下的不足厚度的部分，单算一层。(merge_last=False)
    因此CS2最后剩下的不足厚度的部分，合并到前一层。(merge_last=True)

    将每个长条分map存到dlabel文件中，标记为1，其它为0
    将所有层以一个map存到dlabel文件中，以序号标记各层，其它为0

    Args:
        width (int, optional): Defaults to 5.
    """
    Hemi = hemi2Hemi[hemi]

    # load line gdist maps
    lines = pkl.load(open(line_file, 'rb'))
    border_vertices = [i[-1] for i in lines]
    reader = CiftiReader(line_gdist_file)
    border_vertices_all = [int(i) for i in reader.map_names()]
    map_indices = [border_vertices_all.index(i) for i in border_vertices]
    line_gdist_maps = reader.get_data()[map_indices]

    # load CS gdist map
    CS_gdist_map = nib.load(CS_gdist_file).get_fdata()
    CS_gdist_map_vis = CS_gdist_map[mask]

    # make bar maps
    width_half = width / 2
    bar_maps = np.zeros_like(line_gdist_maps, np.uint8)
    bar_lbl_tables = []
    bar_label = f'{Hemi}_radial_bar'
    for line_idx, line_gdist_map in enumerate(line_gdist_maps):
        bar_maps[line_idx, line_gdist_map <= width_half] = 1
        lbl_tab = nib.cifti2.Cifti2LabelTable()
        lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
        lbl_tab[1] = nib.cifti2.Cifti2Label(1, bar_label, 0, 0, 0, 1)
        bar_lbl_tables.append(lbl_tab)

    # make ring map
    ring_map = np.zeros_like(CS_gdist_map, np.uint16)
    ring_lbl_tab = nib.cifti2.Cifti2LabelTable()
    ring_lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    CS_gdist_min = np.min(CS_gdist_map_vis)
    CS_gdist_max = np.max(CS_gdist_map_vis)
    ring_boundaries = np.arange(CS_gdist_min, CS_gdist_max, width)
    if merge_last:
        ring_boundaries[-1] = CS_gdist_max
    else:
        ring_boundaries = np.r_[ring_boundaries, CS_gdist_max]
    n_ring = len(ring_boundaries) - 1
    print(f'{n_ring} rings boundaries:\n', ring_boundaries)
    cmap = plt.cm.jet
    color_indices = np.linspace(0, 1, n_ring)
    for s_idx, s_boundary in enumerate(ring_boundaries[:-1]):
        e_idx = s_idx + 1
        e_boundary = ring_boundaries[e_idx]
        if e_idx == n_ring:
            ring_mask = np.logical_and(CS_gdist_map >= s_boundary,
                                       CS_gdist_map <= e_boundary)
        else:
            ring_mask = np.logical_and(CS_gdist_map >= s_boundary,
                                       CS_gdist_map < e_boundary)
        ring_map[ring_mask] = e_idx
        ring_lbl_tab[e_idx] = nib.cifti2.Cifti2Label(
            e_idx, f'{s_boundary}~{e_boundary}', *cmap(color_indices[s_idx])
        )

    # save out
    bm_list = reader.brain_models()
    map_names = [str(i) for i in border_vertices]
    save2cifti(out_bar_file, bar_maps, bm_list, map_names, label_tables=bar_lbl_tables)
    save2cifti(out_ring_file, ring_map, bm_list, label_tables=[ring_lbl_tab])


def calc_var_ring_bar(mask, bar_file, ring_file, pc_file,
                      out_file1, out_file2, out_file3):
    """
    分别在层和长条内计算变异
    每个层和长条都可以计算各自的变异，比较C1和C2在每个层或是长条内的变异
    把所有长条内的变异求平均作为层间变异（层级加工的方向）
    把所有层内的变异求平均作为层内变异（功能特异化的方向）
    """
    # prepare parameters
    method = 'CV3_mean-map'
    ylim = (0.6, 1.4)
    # ylim = None
    n_pc = 2  # 前N个成分
    # out_file1, out_file2, out_file3 = ('go on',) * 3

    # loading
    bar_reader = CiftiReader(bar_file)
    bar_maps = bar_reader.get_data()[:, mask]
    n_bar = bar_maps.shape[0]

    ring_reader = CiftiReader(ring_file)
    ring_map = ring_reader.get_data()[0, mask]
    ring_nums = ring_reader.label_info[0]['key']
    ring_nums.remove(0)
    n_ring = len(ring_nums)

    pc_reader = CiftiReader(pc_file)
    pc_maps = pc_reader.get_data()[:n_pc, mask]
    pc_names = tuple(pc_reader.map_names()[:n_pc])
    means_abs = np.mean(np.abs(pc_maps), 1)  # 整个map的绝对值的均值
    abs_means = np.abs(np.mean(pc_maps, 1))  # 整个map的均值的绝对值
    map_stds = np.std(pc_maps, 1)  # 整个map的标准差
    print('PC1和PC2 map的绝对值的均值分别是：', means_abs)
    print('PC1和PC2 map的均值的绝对值分别是：', abs_means)
    print('PC1和PC2 map的标准差分别是：', map_stds)

    if method == 'CV3_mean-map':
        def var_func(arr, axis, ddof=0):
            # std / 各自map的绝对值的均值
            var = np.std(arr, axis, ddof=ddof) /\
                means_abs
            return var
    elif method == 'CV4_mean-map':
        def var_func(arr, axis, ddof=0):
            # std / 各自map的均值的绝对值
            var = np.std(arr, axis, ddof=ddof) /\
                abs_means
            return var
    elif method == 'std_MapStd':
        def var_func(arr, axis, ddof=0):
            # std / 各自map的std
            var = np.std(arr, axis, ddof=ddof) /\
                map_stds
            return var
    else:
        var_func = get_var_func(method)

    # calculating
    bar_vars = np.zeros((n_pc, n_bar), np.float64)
    for bar_idx in range(n_bar):
        bar_vars[:, bar_idx] = var_func(
            pc_maps[:, bar_maps[bar_idx] == 1], 1)
    var_between_layer_y = np.mean(bar_vars, 1)
    var_between_layer_yerr = sem(bar_vars, 1)

    ring_vars = np.zeros((n_pc, n_ring), np.float64)
    for ring_idx in range(n_ring):
        ring_vars[:, ring_idx] = var_func(
            pc_maps[:, ring_map == ring_nums[ring_idx]], 1)
    var_within_layer_y = np.mean(ring_vars, 1)
    var_within_layer_yerr = sem(ring_vars, 1)

    # plot
    y = np.array([var_between_layer_y, var_within_layer_y])
    yerr = np.array([var_between_layer_yerr, var_within_layer_yerr])
    plot_bar(y, figsize=(3, 3), yerr=yerr,
             label=('hierarchy', 'specialization'),
             xticklabel=pc_names, ylabel='variation',
             mode=out_file1, title=method, ylim=ylim)
    plot_bar(ring_vars, figsize=(7, 3), label=pc_names,
             xlabel='layer number', ylabel='variation',
             mode=out_file2, title='within')
    plot_bar(bar_vars, figsize=(7, 3), label=pc_names,
             xlabel='bar number', ylabel='variation',
             mode=out_file3, title='between')
    if out_file1 == 'go on':
        plt.show()
# 以枕极为原点，以圆环代表层，以长条代表跨层<<<


if __name__ == '__main__':
    # get_radial_line(hemi='rh', cs_type='CS1')
    # get_radial_line(hemi='rh', cs_type='CS2')
    # line_pkl2cii(hemi='rh', fname='MMP-vis3_RadialLine-CS1_R')
    # line_pkl2cii(hemi='rh', fname='MMP-vis3_RadialLine-CS2_R')
    # simplify_radial_line(hemi='rh', line_type='RadialLine', cs_type='CS1', thr=0.9, N=2)
    # simplify_radial_line(hemi='rh', line_type='RadialLine', cs_type='CS2', thr=0.9, N=2)
    # line_pkl2cii(hemi='rh', fname='MMP-vis3_RadialLine-CS1_R_thr90_N2')
    # line_pkl2cii(hemi='rh', fname='MMP-vis3_RadialLine-CS2_R_thr90_N2')
    # get_radial_line1(hemi='rh', cs_type='CS1')
    # get_radial_line1(hemi='rh', cs_type='CS2')
    # line_pkl2cii(hemi='rh', fname='MMP-vis3_RadialLine1-CS1_R')
    # line_pkl2cii(hemi='rh', fname='MMP-vis3_RadialLine1-CS2_R')
    # simplify_radial_line(hemi='rh', line_type='RadialLine1', cs_type='CS1', thr=0.9, N=2)
    # simplify_radial_line(hemi='rh', line_type='RadialLine1', cs_type='CS2', thr=0.9, N=2)
    # line_pkl2cii(hemi='rh', fname='MMP-vis3_RadialLine1-CS1_R_thr90_N2')
    # line_pkl2cii(hemi='rh', fname='MMP-vis3_RadialLine1-CS2_R_thr90_N2')

    # prepare_ring_bar(
    #     hemi='rh', width=5, mask=Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-R')),
    #     line_file=pjoin(work_dir, 'MMP-vis3_RadialLine1-CS1_R_thr90_N2.pkl'),
    #     line_gdist_file = pjoin(anal_dir, 'gdist/gdist_src-MMP-vis3_RadialLine1-CS1_R.dscalar.nii'),
    #     CS_gdist_file = pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus.dscalar.nii'),
    #     out_bar_file = pjoin(work_dir, 'MMP-vis3_RadialBar1-CS1_R_thr90_N2_width5.dlabel.nii'),
    #     out_ring_file = pjoin(work_dir, 'MMP-vis3_ring1-CS1_R_width5.dlabel.nii'),
    #     merge_last=False
    # )
    # prepare_ring_bar(
    #     hemi='rh', width=5, mask=Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-R')),
    #     line_file=pjoin(work_dir, 'MMP-vis3_RadialLine1-CS2_R_thr90_N2.pkl'),
    #     line_gdist_file = pjoin(anal_dir, 'gdist/gdist_src-MMP-vis3_RadialLine1-CS2_R.dscalar.nii'),
    #     CS_gdist_file = pjoin(anal_dir, 'gdist/gdist_src-CalcarineSulcus-split.dscalar.nii'),
    #     out_bar_file = pjoin(work_dir, 'MMP-vis3_RadialBar1-CS2_R_thr90_N2_width5.dlabel.nii'),
    #     out_ring_file = pjoin(work_dir, 'MMP-vis3_ring1-CS2_R_width5.dlabel.nii'),
    #     merge_last=True
    # )

    # calc_var_ring_bar(
    #     mask=Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-R'))[0],
    #     bar_file=pjoin(work_dir, 'MMP-vis3_RadialBar1-CS1_R_thr90_N2_width5.dlabel.nii'),
    #     ring_file=pjoin(work_dir, 'MMP-vis3_ring1-CS1_R_width5.dlabel.nii'),
    #     pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
    #     out_file1=pjoin(work_dir, 'within_between.jpg'),
    #     out_file2=pjoin(work_dir, 'within.jpg'),
    #     out_file3=pjoin(work_dir, 'between.jpg')
    # )
    calc_var_ring_bar(
        mask=Atlas('HCP-MMP').get_mask(get_rois('MMP-vis3-R'))[0],
        bar_file=pjoin(work_dir, 'MMP-vis3_RadialBar1-CS2_R_thr90_N2_width5.dlabel.nii'),
        ring_file=pjoin(work_dir, 'MMP-vis3_ring1-CS2_R_width5.dlabel.nii'),
        pc_file=pjoin(anal_dir, 'decomposition/HCPY-M+T_MMP-vis3-R_zscore1_PCA-subj.dscalar.nii'),
        out_file1=pjoin(work_dir, 'within_between.jpg'),
        out_file2=pjoin(work_dir, 'within.jpg'),
        out_file3=pjoin(work_dir, 'between.jpg')
    )
