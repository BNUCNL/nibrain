from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
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


def calc_gdist():
    """
    Calculate geodesic distance between each two ROIs.
    """
    import os
    import time
    import gdist
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import roi2label

    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    roi_file = pjoin(work_dir, 'rois_v3_{}.nii.gz')
    geo_file = '/nfs/m1/hcp/{sid}/T1w/fsaverage_LR32k/' \
               '{sid}.{Hemi}.midthickness_MSMAll.32k_fs_LR.surf.gii'
    log_file = pjoin(work_dir, 'gdist_error_log')
    out_file = pjoin(work_dir, 'gdist.csv')

    subj_ids = open(subj_file).read().splitlines()
    n_subj = len(subj_ids)

    out_dict = {}
    for hemi in hemis:
        for roi1_idx, roi1 in enumerate(rois[:-1]):
            for roi2 in rois[roi1_idx+1:]:
                k = f"{hemi}_{roi1.split('-')[0]}-{roi2.split('-')[0]}"
                out_dict[k] = np.ones(n_subj, dtype=np.float64) * np.nan
    log_lines = []
    for hemi in hemis:
        roi_maps = nib.load(roi_file.format(hemi)).get_fdata().squeeze().T
        for subj_idx, subj_id in enumerate(subj_ids):
            time1 = time.time()
            roi_map = roi_maps[subj_idx]
            g_file = geo_file.format(sid=subj_id, Hemi=hemi2Hemi[hemi])
            if not os.path.exists(g_file):
                log_lines.append(f'{g_file} does not exist.')
                continue
            geo = nib.load(g_file)
            coords = geo.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
            coords = coords.astype(np.float64)
            faces = geo.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
            faces = faces.astype(np.int32)
            for roi1_idx, roi1 in enumerate(rois[:-1]):
                roi1_idx_map = roi_map == roi2label[roi1]
                if np.any(roi1_idx_map):
                    for roi2 in rois[roi1_idx+1:]:
                        roi2_idx_map = roi_map == roi2label[roi2]
                        if np.any(roi2_idx_map):
                            roi1_vertices = np.where(roi1_idx_map)[0]
                            roi1_vertices = roi1_vertices.astype(np.int32)
                            roi2_vertices = np.where(roi2_idx_map)[0]
                            roi2_vertices = roi2_vertices.astype(np.int32)
                            ds = gdist.compute_gdist(coords, faces, roi1_vertices,
                                                     roi2_vertices)
                            k = f"{hemi}_{roi1.split('-')[0]}-{roi2.split('-')[0]}"
                            out_dict[k][subj_idx] = np.min(ds)
            print(f'Finished: {subj_idx+1}/{n_subj}, '
                  f'cost {time.time()-time1} seconds.')
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
    data_file = pjoin(work_dir, 'gdist.csv')

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
    data_file = pjoin(work_dir, 'gdist.csv')

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


if __name__ == '__main__':
    # get_roi_idx_vec()
    # count_roi()
    # calc_gdist()
    plot_gdist()
    compare_gdist()
