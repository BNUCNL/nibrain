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


if __name__ == '__main__':
    get_subject_info_from_fmriresults01(proj_name='HCPD')
    get_subject_info_from_fmriresults01(proj_name='HCPA')
