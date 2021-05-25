from os.path import join as pjoin

proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'analysis/structure')


def calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='LR',
            zscore_flag=False):
    """
    Calculate thickness or myelination
    """
    import time
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from scipy.stats import zscore
    from cxy_visual_dev.lib.predefine import Atlas, L_offset_32k, L_count_32k
    from cxy_visual_dev.lib.predefine import R_offset_32k, R_count_32k

    # inputs
    info_file = f'/nfs/e1/{proj_name}/{proj_name}_SubjInfo.csv'
    proj2par = {
        'HCPD': '/nfs/e1/HCPD/fmriresults01',
        'HCPA': '/nfs/e1/HCPA/fmriresults01'}
    proj_par = proj2par[proj_name]

    meas2file = {
        'myelin': pjoin(proj_par,
                        '{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
                        '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'),
        'thickness': pjoin(proj_par,
                           '{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
                           '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii')}
    meas_file = meas2file[meas_name]

    # outputs
    if zscore_flag:
        out_file = pjoin(work_dir, f'{proj_name}_{meas_name}_{atlas_name}_zscore.csv')
    else:
        out_file = pjoin(work_dir, f'{proj_name}_{meas_name}_{atlas_name}.csv')

    # prepare
    atlas = Atlas(atlas_name)
    df = pd.read_csv(info_file)
    n_subj = df.shape[0]

    # calculate
    for i, idx in enumerate(df.index, 1):
        time1 = time.time()
        sid = df.loc[idx, 'subID']
        meas_map = nib.load(meas_file.format(sid=sid)).get_fdata()
        if zscore:
            meas_map_L = meas_map[:, L_offset_32k:(L_offset_32k+L_count_32k)]
            meas_map_R = meas_map[:, R_offset_32k:(R_offset_32k+R_count_32k)]
            meas_map_L = zscore(meas_map_L, 1)
            meas_map_R = zscore(meas_map_R, 1)
            meas_map[:, L_offset_32k:(L_offset_32k+L_count_32k)] = meas_map_L
            meas_map[:, R_offset_32k:(R_offset_32k+R_count_32k)] = meas_map_R
        for roi, lbl in atlas.roi2label.items():
            meas = np.mean(meas_map[atlas.maps == lbl])
            df.loc[idx, roi] = meas
        print(f'Finished {i}/{n_subj}: cost {time.time() - time1}')

    # save
    df.to_csv(out_file, index=False)


def plot_line(fpath, rois, ylabel, n_row, n_col):
    import numpy as np
    import pandas as pd
    from scipy.stats.stats import sem
    from matplotlib import pyplot as plt

    # inputs
    Hemis = ('L', 'R')

    # load
    df = pd.read_csv(fpath)
    ages = np.array(df['age in years'])
    age_uniq = np.unique(ages)

    rois_without_hemi = ['_'.join(i.split('_')[1:]) for i in rois]
    rois_uniq = np.unique(rois_without_hemi)
    max_row_idx = int((len(rois_uniq)-1) / n_col)
    _, axes = plt.subplots(n_row, n_col)
    if axes.shape != (n_row, n_col):
        axes = axes.reshape((n_row, n_col))
    for i, roi_without_hemi in enumerate(rois_uniq):
        row_idx = int(i / n_col)
        col_idx = i % n_col
        ax = axes[row_idx, col_idx]
        for Hemi in Hemis:
            roi = f'{Hemi}_{roi_without_hemi}'
            if roi not in rois:
                continue
            meas_vec = np.array(df[roi])
            ys = []
            yerrs = []
            for age in age_uniq:
                meas_tmp = meas_vec[ages == age]
                ys.append(np.mean(meas_tmp))
                yerrs.append(sem(meas_tmp))
            ax.errorbar(age_uniq, ys, yerrs, label=roi)
        if col_idx == 0:
            ax.set_ylabel(ylabel)
        if row_idx == max_row_idx:
            ax.set_xlabel('age in years')
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='Cole_visual_ROI')
    # calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='Cole_visual_ROI')

    # plot line for Cole_visual_ROI
    # from cxy_visual_dev.lib.ColeNet import get_parcel2label_by_ColeName
    # meas_names = ('myelin', 'thickness')
    # fpaths = pjoin(work_dir, 'HCPD_{}_Cole_visual_ROI.csv')
    # net_names = ('Primary Visual', 'Secondary Visual',
    #              'Posterior Multimodal', 'Ventral Multimodal')
    # for meas_name in meas_names:
    #     fpath = fpaths.format(meas_name)
    #     for net_name in net_names:
    #         plot_line(fpath=fpath,
    #                   rois=list(get_parcel2label_by_ColeName(net_name).keys()),
    #                   ylabel=meas_name, n_row=4, n_col=7)

    # calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='LR')
    # calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='LR')

    # plot line for LR
    # meas_names = ('myelin', 'thickness')
    # fpaths = pjoin(work_dir, 'HCPD_{}_LR.csv')
    # rois = ('L_cortex', 'R_cortex')
    # for meas_name in meas_names:
    #     plot_line(fpath=fpaths.format(meas_name),
    #               rois=rois, ylabel=meas_name, n_row=2, n_col=1)

    # calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='Cole_visual_LR')
    # calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='Cole_visual_LR')

    # plot line for Cole_visual_LR
    # meas_names = ('myelin', 'thickness')
    # fpaths = pjoin(work_dir, 'HCPD_{}_Cole_visual_LR.csv')
    # rois = ('L_cole_visual', 'R_cole_visual')
    # for meas_name in meas_names:
    #     plot_line(fpath=fpaths.format(meas_name),
    #               rois=rois, ylabel=meas_name, n_row=2, n_col=1)

    calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='Cole_visual_ROI', zscore_flag=True)
    calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='Cole_visual_ROI', zscore_flag=True)

    # plot line for Cole_visual_ROI after zscore
    from cxy_visual_dev.lib.ColeNet import get_parcel2label_by_ColeName
    meas_names = ('myelin', 'thickness')
    fpaths = pjoin(work_dir, 'HCPD_{}_Cole_visual_ROI_zscore.csv')
    net_names = ('Primary Visual', 'Secondary Visual',
                 'Posterior Multimodal', 'Ventral Multimodal')
    for meas_name in meas_names:
        fpath = fpaths.format(meas_name)
        for net_name in net_names:
            plot_line(fpath=fpath,
                      rois=list(get_parcel2label_by_ColeName(net_name).keys()),
                      ylabel=meas_name, n_row=4, n_col=7)
