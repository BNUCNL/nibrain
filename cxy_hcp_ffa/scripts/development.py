import numpy as np
from os.path import join as pjoin

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/development')


def get_subject_info_from_fmriresults01(proj_name='HCPD'):
    """Get subject information from fmriresults01.txt

    Args:
        proj_name (str, optional): Project name. Defaults to 'HCPD'.
    """
    import pandas as pd
    from decimal import Decimal, ROUND_HALF_UP

    # prepare
    src_file = f'/nfs/e1/{proj_name}/fmriresults01.txt'
    trg_file = pjoin(work_dir, f'{proj_name}_SubjInfo.csv')

    # calculate
    df = pd.read_csv(src_file, sep='\t')
    df.drop(labels=[0], axis=0, inplace=True)
    subj_ids = sorted(set(df['src_subject_id']))
    out_dict = {'subID': subj_ids, 'age in months': [],
                'age in years': [], 'gender': []}
    for subj_id in subj_ids:
        idx_vec = df['src_subject_id'] == subj_id
        age = list(set(df.loc[idx_vec, 'interview_age']))
        assert len(age) == 1
        age = int(age[0])
        out_dict['age in months'].append(age)
        age_year = Decimal(str(age/12)).quantize(Decimal('0'),
                                                 rounding=ROUND_HALF_UP)
        out_dict['age in years'].append(int(age_year))

        gender = list(set(df.loc[idx_vec, 'sex']))
        assert len(gender) == 1
        out_dict['gender'].append(gender[0])

    # save
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def get_subject_info_from_completeness(proj_name='HCPD'):
    """Get subject information from HCD_LS_2.0_subject_completeness.csv or
    HCA_LS_2.0_subject_completeness.csv.

    Args:
        proj_name (str, optional): Project name. Defaults to 'HCPD'.
    """
    import pandas as pd
    from decimal import Decimal, ROUND_HALF_UP

    # prepare
    name2file = {
        'HCPD': '/nfs/e1/HCPD/HCD_LS_2.0_subject_completeness.csv',
        'HCPA': '/nfs/e1/HCPA/HCA_LS_2.0_subject_completeness.csv'}
    src_file = name2file[proj_name]
    trg_file = pjoin(work_dir, f'{proj_name}_SubjInfo_completeness.csv')

    # calculate
    df = pd.read_csv(src_file)
    df.drop(labels=[0], axis=0, inplace=True)
    assert df['src_subject_id'].to_list() == sorted(df['src_subject_id'])
    ages_year = [int(Decimal(str(int(age)/12)).quantize(
                 Decimal('0'), rounding=ROUND_HALF_UP))
                 for age in df['interview_age']]
    out_dict = {
        'subID': df['src_subject_id'],
        'age in months': df['interview_age'],
        'age in years': ages_year,
        'gender': df['sex']
    }

    # save
    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(trg_file, index=False)


def plot_age_distribution(proj_name='HCPD'):
    import pandas as pd
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import show_bar_value

    age_type = 'age in years'
    fname = f'{proj_name}_SubjInfo.csv'
    info_file = pjoin(work_dir, fname)
    data = pd.read_csv(info_file)
    ages = data[age_type].to_list()
    ages_uniq = np.unique(ages)

    y = [ages.count(age) for age in ages_uniq]
    rects = plt.bar(ages_uniq, y, edgecolor='k', facecolor='w')
    show_bar_value(rects)
    plt.xlabel(age_type)
    plt.xticks(ages_uniq, ages_uniq, rotation=45)
    plt.ylabel('#subjects')
    plt.title(fname)
    plt.tight_layout()
    plt.show()


def calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='MPM1'):
    """
    Calculate thickness or myelination
    """
    import time
    import pandas as pd
    import nibabel as nib
    from cxy_hcp_ffa.lib.predefine import roi2label, hemi2stru
    from commontool.io.io import CiftiReader

    # inputs
    hemis = ('lh', 'rh')
    info_file = pjoin(work_dir, f'{proj_name}_SubjInfo.csv')

    label2roi = {}
    for k, v in roi2label.items():
        label2roi[v] = k

    atlas2file = {
        'MPM1': pjoin(proj_dir,
                      'analysis/s2/1080_fROI/refined_with_Kevin/'
                      'MPM_v3_{hemi}_0.25.nii.gz')}
    atlas_file = atlas2file[atlas_name]

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
    out_file = pjoin(work_dir, f'{proj_name}_{meas_name}_{atlas_name}.csv')

    # prepare
    df = pd.read_csv(info_file)
    n_subj = df.shape[0]
    hemi2atlas = {}
    hemi2atlas_labels = {}
    for hemi in hemis:
        atlas_map = nib.load(atlas_file.format(hemi=hemi)).get_fdata().squeeze()
        hemi2atlas[hemi] = atlas_map
        atlas_labels = np.unique(atlas_map)
        hemi2atlas_labels[hemi] = atlas_labels[atlas_labels != 0]

    # calculate
    for i, idx in enumerate(df.index, 1):
        time1 = time.time()
        sid = df.loc[idx, 'subID']
        reader = CiftiReader(meas_file.format(sid=sid))
        for hemi in hemis:
            meas_map = reader.get_data(hemi2stru[hemi], True)[0]
            atlas_map = hemi2atlas[hemi]
            atlas_labels = hemi2atlas_labels[hemi]
            for lbl in atlas_labels:
                roi = label2roi[lbl]
                col = f"{roi.split('-')[0]}_{hemi}"
                meas = np.mean(meas_map[atlas_map == lbl])
                df.loc[idx, col] = meas
        print(f'Finished {i}/{n_subj}: cost {time.time() - time1}')

    # save
    df.to_csv(out_file, index=False)


def prepare_plot(proj_name='HCPD', meas_name='thickness', atlas_name='MPM1',
                 n_samples=np.inf, save_out=True):
    import pandas as pd

    # prepare
    hemis = ['lh', 'rh']
    n1, n2 = 'pFus', 'mFus'
    metrics = ('a-b', '(a-b)/std(a-b)', '(a-b)/(a+b)', '2(a-b)/(a+b)')
    fname = pjoin(work_dir, f'{proj_name}_{meas_name}_{atlas_name}.csv')
    data = pd.read_csv(fname)

    # get ages
    age_name = 'age in years'
    ages = np.array(data[age_name])
    ages_uniq = np.unique(ages)

    # select subjects
    indices = []
    for age in ages_uniq:
        indices_tmp = np.where(ages == age)[0]
        n_indices_tmp = len(indices_tmp)
        if n_indices_tmp > n_samples:
            indices_tmp = np.random.choice(indices_tmp, n_samples, False)
        indices.extend(indices_tmp)
    data = data.loc[indices].reset_index(drop=True)
    ages = np.array(data[age_name])
    ages_uniq = np.unique(ages)

    # calculate
    for hemi in hemis:
        col1 = f'{n1}_{hemi}'
        col2 = f'{n2}_{hemi}'
        a = np.array(data[col1])
        b = np.array(data[col2])
        diff = a - b
        for metric in metrics:
            if metric == 'a-b':
                data[f'{n1}-{n2}_{hemi}'] = diff
            elif metric == '(a-b)/std(a-b)':
                age2std = {}
                for age in ages_uniq:
                    age2std[age] = np.std(diff[ages == age])
                data[f'{n1}-{n2}_div-std_{hemi}'] = [diff[i]/age2std[age] for i, age in enumerate(ages)]
            elif metric == '(a-b)/(a+b)':
                data[f'{n1}-{n2}_div-sum_{hemi}'] = diff / (a + b)
            elif metric == '2(a-b)/(a+b)':
                data[f'{n1}-{n2}_div-mean_{hemi}'] = 2*diff / (a + b)
            else:
                raise ValueError('not supported metric:', metric)
    if save_out:
        out_file = pjoin(work_dir, f'{proj_name}_{meas_name}_{atlas_name}_prep_{n_samples}.csv')
        data.to_csv(out_file, index=False)

    return data


def plot_polyfit(meas_name='thickness'):
    import pandas as pd
    from matplotlib import pyplot as plt
    from magicbox.vis.plot import polyfit_plot

    hemi = 'lh'
    cols = [f'pFus_{hemi}', f'mFus_{hemi}']
    fname = f'HCPD_{meas_name}_MPM1_prep_inf.csv'
    data = pd.read_csv(pjoin(work_dir, fname))

    age_name = 'age in years'
    ages = np.array(data[age_name])
    for col in cols:
        print(f'\n---{col}---\n')
        polyfit_plot(ages, np.array(data[col]), 1)
    plt.legend(cols)
    plt.xlabel(age_name)
    plt.title(fname)
    plt.tight_layout()
    plt.show()


def plot_polyfit_box(meas_name='thickness'):
    import pandas as pd
    from matplotlib import pyplot as plt
    from magicbox.vis.plot import polyfit_plot

    # hemi = 'lh'
    # cols = [f'pFus_{hemi}', f'mFus_{hemi}']
    # colors = ['green', 'blue']
    cols = ['pFus-mFus_lh', 'pFus-mFus_rh']
    colors = ['k', 'red']
    fname = f'HCPD_{meas_name}_MPM1_prep_inf.csv'
    data = pd.read_csv(pjoin(work_dir, fname))

    age_name = 'age in years'
    ages = np.array(data[age_name])
    ages_uniq = np.unique(ages)

    for col_idx, col in enumerate(cols):
        print(f'\n---{col}---\n')
        color = colors[col_idx]
        points_list = []
        for age in ages_uniq:
            indices = np.where(ages == age)[0]
            points_list.append(data[col].loc[indices].to_list())
        bplot = plt.boxplot(points_list, positions=ages_uniq, patch_artist=True,
                            showfliers=False, whiskerprops={'color': color},
                            capprops={'color': color}, medianprops={'color': color})
        for patch in bplot['boxes']:
            patch.set_edgecolor(color)
            patch.set_facecolor('w')
        polyfit_plot(ages, np.asarray(data[col]), 1, scatter_plot=False,
                     color=color, label=col)
    plt.legend()
    plt.title(fname)
    plt.xlabel(age_name)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # get_subject_info_from_fmriresults01(proj_name='HCPD')
    # get_subject_info_from_fmriresults01(proj_name='HCPA')
    # get_subject_info_from_completeness(proj_name='HCPD')
    # get_subject_info_from_completeness(proj_name='HCPA')
    # plot_age_distribution(proj_name='HCPD')
    # plot_age_distribution(proj_name='HCPA')
    # calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='MPM1')
    # calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='MPM1')
    # calc_TM(proj_name='HCPA', meas_name='thickness', atlas_name='MPM1')
    # calc_TM(proj_name='HCPA', meas_name='myelin', atlas_name='MPM1')
    # prepare_plot(proj_name='HCPD', meas_name='thickness', atlas_name='MPM1',
    #              n_samples=np.inf, save_out=True)
    # prepare_plot(proj_name='HCPD', meas_name='myelin', atlas_name='MPM1',
    #              n_samples=np.inf, save_out=True)
    # prepare_plot(proj_name='HCPA', meas_name='thickness', atlas_name='MPM1',
    #              n_samples=np.inf, save_out=True)
    # prepare_plot(proj_name='HCPA', meas_name='myelin', atlas_name='MPM1',
    #              n_samples=np.inf, save_out=True)
    # plot_polyfit(meas_name='thickness')
    # plot_polyfit(meas_name='myelin')
    plot_polyfit_box(meas_name='thickness')
    plot_polyfit_box(meas_name='myelin')
