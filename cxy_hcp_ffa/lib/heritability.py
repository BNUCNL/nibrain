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


def icc(x, y=None, n_bootstrap=None, confidence=95):
    '''
    Calculate intraclass correlation between two squences.

    Parameters
    ----------
    x : array-like
        If y is None, x must be a 2D array-like data. The number of rows must
        be two (2 x n matrix), which represent two sequences respectively.
        If y is a sequence, x must be a sequence too.
    y : 1D array-like
        the second sequence
    n_bootstrap : positive integer
        If is not None, do bootstrap with n_bootstrap iterations.
    confidence : a number between 0 and 100
        It is used when n_bootstrap is not None.
        It determines the confidence boundary of the bootstrap. For example,
        when it is 95, the lower and upper boundaries are 2.5- and
        97.5-percentile values.

    Returns
    -------
    r_lb : float
        lower boundary of confidence interval
        Only returned when n_bootstrap is not None.
    r : float
        intraclass correlation
        If n_bootstrap is not None, it is the median correlation across all
        bootstraps.
    r_ub : float
        upper boundary of confidence interval
        Only returned when n_bootstrap is not None.

    References
    ----------
    https://github.com/noahbenson/hcp-lines/blob/master/notebooks/hcp-lines.ipynb
    '''
    x = np.asarray(x if y is None else [x, y])
    assert x.shape[0] == 2

    if n_bootstrap is not None:
        n = x.shape[1]
        rng = np.arange(n)
        rs = [icc(x[:, np.random.choice(rng, n)]) for _ in range(n_bootstrap)]
        lev = 0.5 * (100 - confidence)
        return tuple(np.percentile(rs, [lev, 50, 100-lev]))

    mu0 = np.mean(x, axis=0)
    ms0 = np.sum((x - mu0)**2) / x.shape[1]
    ms1 = np.var(mu0, ddof=1)*2
    r = (ms1 - ms0) / (ms0 + ms1)
    return r


def heritability(mz, dz, n_bootstrap=None, confidence=95):
    '''
    heritability(mz, dz) yields Falconer's heritability index, h^2.

    Parameters
    ----------
    mz : 2D array-like
        The number of rows must be two (2 x n matrix).
    dz : 2D array-like
        The number of rows must be two (2 x n matrix).
    n_bootstrap : positive integer
        If is not None, do bootstrap with n_bootstrap iterations.
    confidence : a number between 0 and 100
        It is used when n_bootstrap is not None.
        It determines the confidence boundary of the bootstrap. For example,
        when it is 95, the lower and upper boundaries are 2.5- and
        97.5-percentile values.

    Returns
    -------
    h2_lb : float
        lower boundary of confidence interval
        Only returned when n_bootstrap is not None.
    h2 : float
        heritability
        If n_bootstrap is not None, it is the median heritability across all
        bootstraps.
    h2_ub : float
        upper boundary of confidence interval
        Only returned when n_bootstrap is not None.

    References
    ----------
    https://github.com/noahbenson/hcp-lines/blob/master/notebooks/hcp-lines.ipynb
    '''
    if n_bootstrap is None:
        r_mz = icc(mz)
        r_dz = icc(dz)
        h2 = 2 * (r_mz - r_dz)
        return h2
    else:
        mz, dz = np.asarray(mz), np.asarray(dz)
        assert mz.shape[0] == 2 and dz.shape[0] == 2
        n_mz, n_dz = mz.shape[1], dz.shape[1]
        mz_rng, dz_rng = np.arange(n_mz), np.arange(n_dz)
        h2s = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            mz_indices = np.random.choice(mz_rng, n_mz)
            dz_indices = np.random.choice(dz_rng, n_dz)
            h2s[i] = heritability(mz[:, mz_indices], dz[:, dz_indices])
        lev = 0.5 * (100 - confidence)
        return tuple(np.percentile(h2s, [lev, 50, 100-lev]))
