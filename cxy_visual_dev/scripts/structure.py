from os.path import join as pjoin

proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'analysis/structure')


def calc_TM(proj_name='HCPD', meas_name='thickness'):
    """
    Calculate thickness or myelination
    """
    import time
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from cxy_visual_dev.lib.ColeNet import get_parcel2label_by_ColeName

    # inputs
    cole_names = ['Primary Visual', 'Secondary Visual',
                  'Posterior Multimodal', 'Ventral Multimodal']
    info_file = f'/nfs/e1/{proj_name}/{proj_name}_SubjInfo.csv'
    mmp_file = '/nfs/p1/atlases/multimodal_glasser/surface/'\
               'MMP_mpmLR32k.dlabel.nii'
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
    out_file = pjoin(work_dir, f'{proj_name}_{meas_name}.csv')

    # prepare
    parcel2label = get_parcel2label_by_ColeName(cole_names)
    mmp_map = nib.load(mmp_file).get_fdata()[0]
    df = pd.read_csv(info_file)
    n_subj = df.shape[0]

    # calculate
    for i, idx in enumerate(df.index, 1):
        time1 = time.time()
        sid = df.loc[idx, 'subID']
        meas_map = nib.load(meas_file.format(sid=sid)).get_fdata()[0]
        for parcel, lbl in parcel2label.items():
            meas = np.mean(meas_map[mmp_map == lbl])
            df.loc[idx, parcel] = meas
        print(f'Finished {i}/{n_subj}: cost {time.time() - time1}')

    # save
    df.to_csv(out_file, index=False)


def plot_line(proj_name='HCPD', meas_name='thickness'):
    import numpy as np
    import pandas as pd
    from scipy.stats.stats import sem
    from matplotlib import pyplot as plt
    from cxy_visual_dev.lib.ColeNet import get_parcel2label_by_ColeName

    # inputs
    Hemis = ('L', 'R')
    net2rc = {
        'Primary Visual': (1, 3),
        'Secondary Visual': (4, 7),
        'Posterior Multimodal': (2, 2),
        'Ventral Multimodal': (1, 2)
    }  # ColeNet name to n_row and n_col
    df_file = pjoin(work_dir, f'{proj_name}_{meas_name}.csv')

    # load
    df = pd.read_csv(df_file)
    ages = np.array(df['age in years'])
    age_uniq = np.unique(ages)

    for net, rc in net2rc.items():
        parcel2label = get_parcel2label_by_ColeName(net)
        parcels_without_hemi = [i.split('_')[1] for i in parcel2label.keys()]
        parcels_uniq = np.unique(parcels_without_hemi)
        _, axes = plt.subplots(rc[0], rc[1])
        if axes.shape != rc:
            axes = axes.reshape(rc)
        for i, parcel_without_hemi in enumerate(parcels_uniq):
            row_idx = int(i / rc[1])
            col_idx = i % rc[1]
            ax = axes[row_idx, col_idx]
            for Hemi in Hemis:
                parcel = f'{Hemi}_{parcel_without_hemi}'
                if parcel not in parcel2label.keys():
                    continue
                meas_vec = np.array(df[parcel])
                ys = []
                yerrs = []
                for age in age_uniq:
                    meas_tmp = meas_vec[ages == age]
                    ys.append(np.mean(meas_tmp))
                    yerrs.append(sem(meas_tmp))
                ax.errorbar(age_uniq, ys, yerrs, label=parcel)
            if col_idx == 0:
                ax.set_ylabel(meas_name)
            if row_idx+1 == rc[0]:
                ax.set_xlabel('age in years')
            ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # calc_TM(proj_name='HCPD', meas_name='thickness')
    # calc_TM(proj_name='HCPD', meas_name='myelin')
    plot_line(proj_name='HCPD', meas_name='thickness')
    plot_line(proj_name='HCPD', meas_name='myelin')
