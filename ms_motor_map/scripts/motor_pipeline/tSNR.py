"""
calculate the tSNR for pre- and post-denoising fMRI data
"""

import os
import numpy as np
import nibabel as nib

sessidlist = ["sub001", "sub002", "sub003", "sub004", "sub005", \
              "sub006", "sub009", "sub010", "sub014", "sub015", \
              "sub016", "sub017", "sub018", "sub019", "sub020"]

runidlist = ["001", "002", "003", "004", "005", "006", "007", "008"]

smth_cdt = 'unsmth'
hemi = 'rh'
projectdir = "/nfs/e5/studyforrest"
after_funcname = "audiovisual3T_fslpreproc2surface_{0}_denoised".format(
    smth_cdt)
before_funcname = "audiovisual3T_fslpreproc2surface_{0}".format(smth_cdt)
file_name = 'fmcpr.sm0.fsaverage.{0}.nii.gz'.format(hemi)

analysis_dir = '/nfs/e5/studyforrest/data_lxy/result/tSNR/surface'
figure_dir = '/nfs/e5/studyforrest/data_lxy/result/tSNR/figure'
vertex_number = 163842


def tSNR(data):
    """alculate the temporal signal-to-noise ratio (tSNR) for each vertex
    Parameters
    ----------

        data: used to calculate tSNR,
            shape = [n_vertice, m_timepoints].

    Returns
    -------
        data_tSNR: the tSNR of data, shape = [n_vertice, ].

    Notes
    -----
		The tSNR was defined as the ratio between the mean of a timeseries
		and its SD for each vertex
    """

    data_mean = np.mean(data, axis=-1)
    data_std = np.std(data, axis=-1)
    data_tSNR = np.nan_to_num(data_mean / data_std)
    return data_tSNR


# calculate tSNR
for subid in sessidlist:

    after_tSNR = np.zeros([np.size(vertex_number)])
    before_tSNR = np.zeros([np.size(vertex_number)])
    count = 0

    for runid in runidlist:
        before_path = os.path.join(projectdir, subid, before_funcname, runid,
                                   file_name)
        before = nib.load(before_path).get_data()[:, 0, 0, :]
        before_tSNR = before_tSNR + tSNR(before)

        after_path = os.path.join(projectdir, subid, after_funcname, runid,
                                  file_name)
        after_info = nib.load(after_path)
        after = after_info.get_data()[:, 0, 0, :]
        after_tSNR = after_tSNR + tSNR(after)

        print('{0}_{1} done'.format(subid, runid))

    after_tSNR = after_tSNR / len(runidlist)
    before_tSNR = before_tSNR / len(runidlist)

    # save pre-denoising tSNR image
    img = nib.Nifti1Image(before_tSNR.reshape([vertex_number, 1, 1]), None, after_info.get_header())
    result_name = '{0}_{1}_fsaverage_{2}.nii.gz'.format(smth_cdt, subid, hemi)
    save_path = os.path.join(analysis_dir, result_name)
    nib.save(img, save_path)

    # save post-denoising tSNR image
    img = nib.Nifti1Image(after_tSNR.reshape([vertex_number, 1, 1]),
                          None, after_info.get_header())
    result_name = '{0}_{1}_fsaverage_denoised_{2}.nii.gz'.format(
        smth_cdt, subid, hemi)
    save_path = os.path.join(analysis_dir, result_name)
    nib.save(img, save_path)

    print('Saving_{0}'.format(subid))

print('======{0} done======='.format(subid))