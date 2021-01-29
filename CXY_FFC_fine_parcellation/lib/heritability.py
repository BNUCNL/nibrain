import numpy as np
import pandas as pd


def get_twins_id(src_file, trg_file=None):
    """
    Get twins ID according to 'ZygosityGT' and pair the twins according to 
    'Family_ID' from HCP restricted information.
    
    Parameters
    ----------
    src_file : str
        HCP restricted information file (CSV format)
    trg_file : str
        If is not None, save twins ID information to a file (CSV format)
    
    Returns
    -------
    df_out : DataFrame
        twins ID information
    """
    assert src_file.endswith('.csv')
    zygosity = ('MZ', 'DZ')
    df_in = pd.read_csv(src_file)

    df_out = {'familyID': [], 'twin1': [], 'twin2': [], 'zygosity': []}
    for zyg in zygosity:
        df_zyg = df_in[df_in['ZygosityGT'] == zyg]
        family_ids = sorted(set(df_zyg['Family_ID']))
        for fam_id in family_ids:
            subjs = df_zyg['Subject'][df_zyg['Family_ID'] == fam_id]
            subjs = subjs.reset_index(drop=True)
            assert len(subjs) == 2
            df_out['familyID'].append(fam_id)
            df_out['twin1'].append(subjs[0])
            df_out['twin2'].append(subjs[1])
            df_out['zygosity'].append(zyg)
    df_out = pd.DataFrame(df_out)
    
    if trg_file is not None:
        assert trg_file.endswith('.csv')
        df_out.to_csv(trg_file, index=False)
    
    return df_out


def count_twins_id(data):
    """
    Count the number of MZ or DZ pairs
    
    Parameters
    ----------
    data : DataFrame | str
        twins ID information
        If is str, it's a CSV file of twins ID information.
    """
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, str):
        data = pd.read_csv(data)
    else:
        raise TypeError('The input data must be a DataFrame or str!')

    zygosity = ('MZ', 'DZ')
    for zyg in zygosity:
        df_zyg = data[data['zygosity'] == zyg]
        print(f'The number of {zyg}:', len(df_zyg))


def filter_twins_id(data, limit_set, trg_file=None):
    """
    The twin pair will be removed as long as anyone of it is not in limit set
    
    Parameters
    ----------
    data : DataFrame | str
        twins ID information
        If is str, it's a CSV file of twins ID information.
    limit_set : collection
        a collection of subject IDs
    trg_file : str, default None
        If is not None, save filtered twins ID to a file (CSV format)
    
    Returns
    -------
    data : DataFrame
        filtered twins ID information
    """
    if isinstance(data, pd.DataFrame):
        data = data.copy()
    elif isinstance(data, str):
        data = pd.read_csv(data)
    else:
        raise TypeError('The input data must be a DataFrame or str!')

    # filter twins ID
    for idx in data.index:
        if (data['twin1'][idx] not in limit_set or 
            data['twin2'][idx] not in limit_set):
            data.drop(index=idx, inplace=True)
    
    if trg_file is not None:
        assert trg_file.endswith('.csv')
        data.to_csv(trg_file, index=False)
    
    return data
