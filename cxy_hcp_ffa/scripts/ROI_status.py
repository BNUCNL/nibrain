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


if __name__ == '__main__':
    # get_roi_idx_vec()
    count_roi()
