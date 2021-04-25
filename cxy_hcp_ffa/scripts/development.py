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


def calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='MPM1'):
    """
    Calculate thickness or myelination
    """
    import time
    import numpy as np
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


if __name__ == '__main__':
    # get_subject_info_from_fmriresults01(proj_name='HCPD')
    # get_subject_info_from_fmriresults01(proj_name='HCPA')
    # get_subject_info_from_completeness(proj_name='HCPD')
    # get_subject_info_from_completeness(proj_name='HCPA')
    calc_TM(proj_name='HCPD', meas_name='thickness', atlas_name='MPM1')
    calc_TM(proj_name='HCPD', meas_name='myelin', atlas_name='MPM1')
    calc_TM(proj_name='HCPA', meas_name='thickness', atlas_name='MPM1')
    calc_TM(proj_name='HCPA', meas_name='myelin', atlas_name='MPM1')
