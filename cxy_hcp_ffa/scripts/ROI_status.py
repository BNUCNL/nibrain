from os.path import join as pjoin

from numpy import dtype
from magicbox.algorithm.triangular_mesh import get_n_ring_neighbor
from cxy_hcp_ffa.lib.tools import bfs
from cxy_hcp_ffa.lib.predefine import proj_dir

work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')


def get_roi_idx_vec():
    """
    Get index vector with bool values for each ROI.
    The length of each index vector is matched with 1080 subjects.
    True value means the ROI is delineated in the corresponding subject.
    """
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import roi2label

    # inputs
    hemis = ('lh', 'rh')
    roi_file = pjoin(work_dir, 'rois_v3_{hemi}.nii.gz')

    # outputs
    out_file = pjoin(work_dir, 'rois_v3_idx_vec.csv')

    out_dict = {}
    for hemi in hemis:
        roi_maps = nib.load(roi_file.format(hemi=hemi)).get_data().squeeze().T
        for roi, lbl in roi2label.items():
            out_dict[f'{hemi}_{roi}'] = np.any(roi_maps == lbl, axis=1)
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(out_file, index=False)


def count_roi():
    """
    Count valid subjects for each ROI
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(pjoin(work_dir, 'rois_v3_idx_vec.csv'))
    for col in df.columns:
        print(f'#subjects of {col}:', np.sum(df[col]))


def calc_gdist(method='peak'):
    """Calculate geodesic distance between each two ROIs.

    Args:
        method (str, optional): 'peak', 'min', 'min1', 'AP_gap-y', and 'AP_gap-geo'
            If 'peak', use the distance between two vertices
            with peak activation values in two ROIs respectively.

            If 'min', use the minimum distance of pair-wise
            vertices between the two ROIs.

            If 'min1', use the minimum distance of pair-wise
            vertices between the two ROIs. If the two ROIs are continuous, set as 0.

            If 'AP_gap-y', assume the most posterior vertex of mFus-face is P
            and the most anterior vertex of pFus-face is A.
            This distance is calculated as: y coordinate of P - y coordinate of A.

            If AP_gap-geo, use the geodesic distance between P and A.
            Defaults to 'peak'.
    """
    import os
    import time
    import gdist
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import roi2label, hemi2stru
    from magicbox.io.io import CiftiReader

    # inputs
    # rois = ('IOG-face', 'pFus-face', 'mFus-face')
    rois = ('pFus-face', 'mFus-face')
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    roi_file = pjoin(work_dir, 'rois_v3_{}.nii.gz')
    geo_file = '/nfs/m1/hcp/{sid}/T1w/fsaverage_LR32k/' \
               '{sid}.{Hemi}.midthickness_MSMAll.32k_fs_LR.surf.gii'
    geo_file_AP = '/nfs/m1/hcp/{sid}/T1w/fsaverage_LR32k/' \
                  '{sid}.{Hemi}.very_inflated_MSMAll.32k_fs_LR.surf.gii'
    activ_file = pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')

    # outputs
    log_file = pjoin(work_dir, f'gdist_{method}_log')
    out_file = pjoin(work_dir, f'gdist_{method}.csv')

    # preparation
    subj_ids = open(subj_file).read().splitlines()
    n_subj = len(subj_ids)
    activ_reader = CiftiReader(activ_file)
    out_dict = {}
    for hemi in hemis:
        for roi1_idx, roi1 in enumerate(rois[:-1]):
            for roi2 in rois[roi1_idx+1:]:
                k = f"{hemi}_{roi1.split('-')[0]}-{roi2.split('-')[0]}"
                out_dict[k] = np.ones(n_subj, dtype=np.float64) * np.nan
    log_lines = []

    # calculation
    for hemi in hemis:
        roi_maps = nib.load(roi_file.format(hemi)).get_fdata().squeeze().T
        activ_maps = activ_reader.get_data(hemi2stru[hemi], True)
        assert roi_maps.shape == activ_maps.shape
        for subj_idx, subj_id in enumerate(subj_ids):
            time1 = time.time()
            roi_map = roi_maps[subj_idx]
            activ_map = activ_maps[subj_idx]
            g_file = geo_file.format(sid=subj_id, Hemi=hemi2Hemi[hemi])
            if not os.path.exists(g_file):
                log_lines.append(f'{g_file} does not exist.')
                continue
            geo = nib.load(g_file)
            coords = geo.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0]
            coords = coords.data.astype(np.float64)
            faces = geo.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0]
            faces = faces.data.astype(np.int32)

            g_file_AP = geo_file_AP.format(sid=subj_id, Hemi=hemi2Hemi[hemi])
            if not os.path.exists(g_file_AP):
                log_lines.append(f'{g_file_AP} does not exist.')
                continue
            geo_AP = nib.load(g_file_AP)
            coords_AP = geo_AP.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
            for roi1_idx, roi1 in enumerate(rois[:-1]):
                roi1_idx_map = roi_map == roi2label[roi1]
                if np.any(roi1_idx_map):
                    for roi2 in rois[roi1_idx+1:]:
                        roi2_idx_map = roi_map == roi2label[roi2]
                        if np.any(roi2_idx_map):
                            k = f"{hemi}_{roi1.split('-')[0]}-"\
                                f"{roi2.split('-')[0]}"
                            if method == 'peak':
                                roi1_max = np.max(activ_map[roi1_idx_map])
                                roi2_max = np.max(activ_map[roi2_idx_map])
                                roi1_idx_map =\
                                    np.logical_and(roi1_idx_map,
                                                   activ_map == roi1_max)
                                roi2_idx_map =\
                                    np.logical_and(roi2_idx_map,
                                                   activ_map == roi2_max)
                                roi1_vertices = np.where(roi1_idx_map)[0]
                                roi1_vertices = roi1_vertices.astype(np.int32)
                                n_vtx1 = len(roi1_vertices)
                                roi2_vertices = np.where(roi2_idx_map)[0]
                                roi2_vertices = roi2_vertices.astype(np.int32)
                                n_vtx2 = len(roi2_vertices)
                                if n_vtx1 > 1 or n_vtx2 > 1:
                                    msg = f'{subj_id}: {roi1} vs {roi2} '\
                                          f'has multiple peaks.'
                                    log_lines.append(msg)
                                    ds = []
                                    for src_vtx in roi1_vertices:
                                        src_vtx = np.array([src_vtx], np.int32)
                                        ds_tmp = \
                                            gdist.compute_gdist(coords, faces,
                                                                src_vtx,
                                                                roi2_vertices)
                                        ds.extend(ds_tmp)
                                    out_dict[k][subj_idx] = np.mean(ds)
                                elif n_vtx1 == 1 and n_vtx2 == 1:
                                    ds = gdist.compute_gdist(coords, faces,
                                                             roi1_vertices,
                                                             roi2_vertices)
                                    assert len(ds) == 1
                                    out_dict[k][subj_idx] = ds[0]
                                else:
                                    raise RuntimeError("Impossible!")
                            elif method == 'min':
                                roi1_vertices = np.where(roi1_idx_map)[0]
                                roi1_vertices = roi1_vertices.astype(np.int32)
                                roi2_vertices = np.where(roi2_idx_map)[0]
                                roi2_vertices = roi2_vertices.astype(np.int32)
                                ds = gdist.compute_gdist(coords, faces,
                                                         roi1_vertices,
                                                         roi2_vertices)
                                out_dict[k][subj_idx] = np.min(ds)
                            elif method == 'min1':
                                roi1_vertices = np.where(roi1_idx_map)[0]
                                roi2_vertices = np.where(roi2_idx_map)[0]
                                mask = activ_map > 1.65
                                edge_list = get_n_ring_neighbor(faces, mask=mask)

                                continuous_flag = False
                                for vtx1 in roi1_vertices:
                                    for vtx2 in roi2_vertices:
                                        tmp_path = bfs(edge_list, vtx1, vtx2)
                                        if len(tmp_path) != 0:
                                            continuous_flag = True
                                            break
                                    if continuous_flag:
                                        break

                                if continuous_flag:
                                    out_dict[k][subj_idx] = 0
                                else:
                                    roi1_vertices = roi1_vertices.astype(np.int32)
                                    roi2_vertices = roi2_vertices.astype(np.int32)
                                    ds = gdist.compute_gdist(coords, faces,
                                                             roi1_vertices,
                                                             roi2_vertices)
                                    out_dict[k][subj_idx] = np.min(ds)
                            elif method == 'AP_gap-y':
                                ys1 = coords_AP[roi1_idx_map, 1]
                                ys2 = coords_AP[roi2_idx_map, 1]
                                y1 = np.max(ys1)
                                y2 = np.min(ys2)
                                out_dict[k][subj_idx] = y2 - y1
                            elif method == 'AP_gap-geo':
                                ys1 = coords_AP[:, 1].copy()
                                ys2 = coords_AP[:, 1].copy()
                                ys1[~roi1_idx_map] = np.nan
                                ys2[~roi2_idx_map] = np.nan
                                max1 = np.nanmax(ys1)
                                min2 = np.nanmin(ys2)
                                roi1_vertices = np.where(ys1 == max1)[0]
                                roi1_vertices = roi1_vertices.astype(np.int32)
                                n_vtx1 = len(roi1_vertices)
                                roi2_vertices = np.where(ys2 == min2)[0]
                                roi2_vertices = roi2_vertices.astype(np.int32)
                                n_vtx2 = len(roi2_vertices)
                                if n_vtx1 > 1 or n_vtx2 > 1:
                                    msg = f'{subj_id}: {roi1} vs {roi2} ' \
                                          f'has multiple vertices.'
                                    log_lines.append(msg)
                                    ds = []
                                    for src_vtx in roi1_vertices:
                                        src_vtx = np.array([src_vtx], np.int32)
                                        ds_tmp = \
                                            gdist.compute_gdist(coords, faces,
                                                                src_vtx,
                                                                roi2_vertices)
                                        ds.extend(ds_tmp)
                                    out_dict[k][subj_idx] = np.mean(ds)
                                elif n_vtx1 == 1 and n_vtx2 == 1:
                                    ds = gdist.compute_gdist(coords, faces,
                                                             roi1_vertices,
                                                             roi2_vertices)
                                    assert len(ds) == 1
                                    out_dict[k][subj_idx] = ds[0]
                                else:
                                    raise RuntimeError("Impossible!")
                            else:
                                raise ValueError(f'Not supported method: '
                                                 f'{method}')
            print(f'Finished: {subj_idx+1}/{n_subj}, '
                  f'cost {time.time()-time1} seconds.')

    # save out
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(out_file, index=False)
    out_log = '\n'.join(log_lines)
    open(log_file, 'w').write(out_log)


def plot_gdist():
    import numpy as np
    import pandas as pd
    from scipy.stats import sem
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width

    hemis = ('lh', 'rh')
    items = ('pFus-mFus',)
    data_file = pjoin(work_dir, 'gdist_peak.csv')

    n_hemi = len(hemis)
    n_item = len(items)
    df = pd.read_csv(data_file)

    _, ax = plt.subplots()
    x = np.arange(n_hemi)
    width = auto_bar_width(x, n_item)
    offset = -(n_item - 1) / 2
    for item in items:
        ys = np.zeros(n_hemi)
        yerrs = np.zeros(n_hemi)
        for hemi_idx, hemi in enumerate(hemis):
            col = hemi + '_' + item
            data = np.array(df[col])
            data = data[~np.isnan(data)]
            print(f'#{col}: {len(data)}')
            ys[hemi_idx] = np.mean(data)
            yerrs[hemi_idx] = sem(data)
        ax.bar(x + width * offset, ys, width, yerr=yerrs, label=item)
        offset += 1
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(hemis)
    ax.set_ylabel('geodesic distance (mm)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def compare_gdist():
    import numpy as np
    import pandas as pd
    from scipy.stats.stats import ttest_rel

    items = ('pFus-mFus',)
    data_file = pjoin(work_dir, 'gdist_peak.csv')

    df = pd.read_csv(data_file)
    for item in items:
        col1 = 'lh_' + item
        col2 = 'rh_' + item
        data1 = np.array(df[col1])
        data2 = np.array(df[col2])
        nan_vec1 = np.isnan(data1)
        nan_vec2 = np.isnan(data2)
        not_nan_vec = ~np.logical_or(nan_vec1, nan_vec2)
        data1 = data1[not_nan_vec]
        data2 = data2[not_nan_vec]
        print(f'#{item}: {len(data1)}')
        print(f'{col1} vs {col2}:', ttest_rel(data1, data2))


def calc_prob_map(hemi='lh'):
    import numpy as np
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import roi2label
    from magicbox.io.io import save2nifti

    # inputs
    n_roi = len(roi2label)
    print(n_roi)
    roi_file = pjoin(work_dir, f'rois_v3_{hemi}.nii.gz')

    # outputs
    out_file = pjoin(work_dir, f'prob_maps_v3_{hemi}.nii.gz')

    # prepare
    rois = nib.load(roi_file).get_fdata().squeeze().T
    n_vtx = rois.shape[1]

    # calculate
    prob_maps = np.ones((n_roi, n_vtx)) * np.nan
    for idx, roi in enumerate(roi2label.keys()):
        label = roi2label[roi]
        print(roi)

        # get indices of subjects which contain the roi
        indices = rois == label
        subj_indices = np.any(indices, 1)

        # calculate roi probability map among valid subjects
        prob_map = np.mean(indices[subj_indices], 0)
        prob_maps[idx] = prob_map

    # save out
    save2nifti(out_file, prob_maps.T[:, None, None, :])


def get_mpm(hemi='lh'):
    """maximal probability map"""
    import numpy as np
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import roi2label
    from magicbox.io.io import save2nifti

    # inputs
    thr = 0.25
    map_indices = (1, 2)
    idx2roi = {
        0: 'IOG-face',
        1: 'pFus-face',
        2: 'mFus-face'}
    prob_file = pjoin(work_dir, f'prob_maps_v3_{hemi}.nii.gz')

    # outputs
    out_file = pjoin(work_dir, f'MPM_v3_{hemi}_{thr}_FFA.nii.gz')

    # prepare
    prob_maps = nib.load(prob_file).get_fdata()[..., map_indices]
    mpm_map = np.zeros(prob_maps.shape[:3])

    # calculate
    supra_thr_idx_arr = prob_maps > thr
    valid_idx_arr = np.any(supra_thr_idx_arr, 3)
    valid_arr = prob_maps[valid_idx_arr, :]
    mpm_tmp = np.argmax(valid_arr, -1)
    for i, idx in enumerate(map_indices):
        roi = idx2roi[idx]
        idx_arr = np.zeros_like(mpm_map, dtype=np.bool8)
        idx_arr[valid_idx_arr] = mpm_tmp == i
        mpm_map[idx_arr] = roi2label[roi]

    # verification
    valid_supra_thr_idx_arr = supra_thr_idx_arr[valid_idx_arr, :]
    valid_count_vec = np.sum(valid_supra_thr_idx_arr, -1)
    valid_count_vec_uniq = np.zeros_like(valid_count_vec)
    for i in range(len(valid_count_vec)):
        valid_supra_thr_idx_vec = valid_supra_thr_idx_arr[i]
        valid_count_vec_uniq[i] = \
            len(set(valid_arr[i, valid_supra_thr_idx_vec]))
    assert np.all(valid_count_vec == valid_count_vec_uniq)

    # save
    save2nifti(out_file, mpm_map)


def roi2cifti(roi_type):
    import numpy as np
    import nibabel as nib
    from magicbox.io.io import CiftiReader, save2cifti
    from cxy_hcp_ffa.lib.predefine import mmp_map_file

    bms = CiftiReader(mmp_map_file).brain_models()
    idx2vtx_lh = list(bms[0].vertex_indices)
    idx2vtx_rh = list(bms[1].vertex_indices)

    # group ROI
    # data_lh = nib.load(pjoin(work_dir,
    #                          'MPM_v3_lh_0.25.nii.gz')).get_data()[..., 0].T
    # data_rh = nib.load(pjoin(work_dir,
    #                          'MPM_v3_rh_0.25.nii.gz')).get_data()[..., 0].T
    # map_names = [None]
    # out_file = pjoin(work_dir, 'MPM_v3_25.dlabel.nii')

    # individual ROI
    data_lh = nib.load(pjoin(work_dir,
                             'rois_v3_lh.nii.gz')).get_data().squeeze().T
    data_rh = nib.load(pjoin(work_dir,
                             'rois_v3_rh.nii.gz')).get_data().squeeze().T
    map_names = open(pjoin(proj_dir,
                           'analysis/s2/subject_id')).read().splitlines()

    data_lh = data_lh.astype(np.int8)
    data_rh = data_rh.astype(np.int8)
    if roi_type == 'FFA':
        # remove IOG-face
        out_file = pjoin(work_dir, 'HCP_FFA.dlabel.nii')
        data_rh[data_rh == 1] = -1
        data_rh[data_rh == 2] = 1
        data_rh[data_rh == 3] = 2
        data_lh[data_lh == 1] = -1
        data_lh[data_lh == 3] = 4
        data_lh[data_lh == 2] = 3
        key2label = {-1: 'None', 1: 'R_pFus-face', 2: 'R_mFus-face',
                     3: 'L_pFus-face', 4: 'L_mFus-face'}
        key2color = {
            -1: (1.0, 1.0, 1.0, 0.0),
            1: (0.0, 1.0, 0.0, 1.0),
            2: (0.0, 0.0, 1.0, 1.0),
            3: (0.0, 1.0, 0.0, 1.0),
            4: (0.0, 0.0, 1.0, 1.0)
        }
    elif roi_type == 'FSR':
        out_file = pjoin(work_dir, 'rois_v3.dlabel.nii')
        data_lh[data_lh == 3] = 6
        data_lh[data_lh == 2] = 5
        data_lh[data_lh == 1] = 4
        key2label = {-1: 'None',
                     1: 'R_IOG-face', 2: 'R_pFus-face', 3: 'R_mFus-face',
                     4: 'L_IOG-face', 5: 'L_pFus-face', 6: 'L_mFus-face'}
        key2color = {
            -1: (1.0, 1.0, 1.0, 0.0),
            1: (1.0, 0.0, 0.0, 1.0),
            2: (0.0, 1.0, 0.0, 1.0),
            3: (0.0, 0.0, 1.0, 1.0),
            4: (1.0, 0.0, 0.0, 1.0),
            5: (0.0, 1.0, 0.0, 1.0),
            6: (0.0, 0.0, 1.0, 1.0)
        }
    else:
        raise ValueError('Not supported roi_type')

    data_lh = data_lh[:, idx2vtx_lh]
    data_rh = data_rh[:, idx2vtx_rh]
    data = np.concatenate((data_lh, data_rh), axis=1, dtype=np.int8)
    data[data == 0] = -1

    label_tables = []
    for row in data:
        lbl_tb = nib.cifti2.Cifti2LabelTable()
        for key in np.unique(row):
            key = int(key)
            lbl_tb[key] = nib.cifti2.Cifti2Label(key, key2label[key],
                                                 *key2color[key])
        label_tables.append(lbl_tb)

    save2cifti(out_file, data, bms, map_names, label_tables=label_tables)


if __name__ == '__main__':
    # get_roi_idx_vec()
    # count_roi()
    # calc_gdist(method='min')
    # calc_gdist(method='peak')
    # calc_gdist(method='AP_gap-y')
    # calc_gdist(method='AP_gap-geo')
    # calc_gdist(method='min1')
    # plot_gdist()
    # compare_gdist()
    # calc_prob_map(hemi='lh')
    # calc_prob_map(hemi='rh')
    # get_mpm(hemi='lh')
    # get_mpm(hemi='rh')
    roi2cifti(roi_type='FFA')
    roi2cifti(roi_type='FSR')
