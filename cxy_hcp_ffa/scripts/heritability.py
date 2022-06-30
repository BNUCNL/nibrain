import numpy as np
import pandas as pd
import pickle as pkl
from os.path import join as pjoin
from scipy.stats import pearsonr
from magicbox.io.io import CiftiReader
from cxy_hcp_ffa.lib.predefine import zyg2label
from cxy_hcp_ffa.lib import heritability as h2

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'heritability')


def get_twinsID():
    """
    Get twins ID according to 'ZygosityGT' and pair the twins according to
    'Family_ID' from HCP restricted information.
    """
    src_file = '/nfs/m1/hcp/S1200_behavior_restricted.csv'
    trg_file = pjoin(work_dir, 'twins_id.csv')

    h2.get_twins_id(src_file, trg_file)
    h2.count_twins_id(trg_file)


def filter_twinsID_1080():
    """
    Check if it's a subset of 1080 subjects.
    If not, filter twins which are not in 1080 subjects.
    """
    twins_file = pjoin(work_dir, 'twins_id.csv')
    subjs_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    trg_file = pjoin(work_dir, 'twins_id_1080.csv')

    twins_df = pd.read_csv(twins_file)
    subjs_twin = set(np.concatenate([twins_df['twin1'], twins_df['twin2']]))
    subjs_id = set([int(_) for _ in open(subjs_file).read().splitlines()])
    flag = subjs_twin.issubset(subjs_id)
    if flag:
        print('All twins is a subset of 1080 subjects.')
    else:
        print('Filter twins which are not in 1080 subjects.')
        h2.filter_twins_id(data=twins_df, limit_set=subjs_id,
                           trg_file=trg_file)
        h2.count_twins_id(trg_file)


def filter_twinsID_1053():
    """
    Check if it's a subset of 1053 subjects.
    If not, filter twins which are not in 1053 subjects.
    """
    twins_file = pjoin(work_dir, 'twins_id.csv')
    subjs_file = pjoin(anal_dir, 'subj_info/subject_id1.txt')
    trg_file = pjoin(anal_dir, 'NI_R1/data_1053/twins_id_1053.csv')

    twins_df = pd.read_csv(twins_file)
    subjs_twin = set(np.concatenate([twins_df['twin1'], twins_df['twin2']]))
    subjs_id = set([int(_) for _ in open(subjs_file).read().splitlines()])
    flag = subjs_twin.issubset(subjs_id)
    if flag:
        print('All twins is a subset of 1053 subjects.')
    else:
        print('Filter twins which are not in 1053 subjects.')
        h2.filter_twins_id(data=twins_df, limit_set=subjs_id,
                           trg_file=trg_file)
        h2.count_twins_id(trg_file)


def filter_twinsID_rfMRI(twins_file, subjs_file, trg_file):
    """
    Check if a subject have all 4 rfMRI runs.
    If not, filter twins which don't have all 4 rfMRI runs.
    """
    twins_df = pd.read_csv(twins_file)
    subjs_twin = set(np.concatenate([twins_df['twin1'], twins_df['twin2']]))
    subjs_id = set([int(_) for _ in open(subjs_file).read().splitlines()])
    flag = subjs_twin.issubset(subjs_id)
    if flag:
        print('All twins have all 4 rfMRI runs')
    else:
        print("Filter twins which don't have all 4 rfMRI runs.")
        h2.filter_twins_id(data=twins_df, limit_set=subjs_id,
                           trg_file=trg_file)
        h2.count_twins_id(trg_file)


def filter_twinsID_G1G2():
    """
    Check if the subject is in G1 or G2.
    If not, filter twins who are not in G1 or G2.
    """
    import numpy as np
    import pandas as pd

    hemis = ('lh', 'rh')
    twins_file = pjoin(work_dir, 'twins_id.csv')
    gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     'grouping/group_id_{hemi}.npy')
    subjs_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    trg_file = pjoin(work_dir, 'twins_id_G1G2_{hemi}.csv')

    twins_df = pd.read_csv(twins_file)
    subjs_twin = set(np.concatenate([twins_df['twin1'], twins_df['twin2']]))
    subjs_1080 = np.array([int(_) for _ in
                          open(subjs_file).read().splitlines()])
    for hemi in hemis:
        gid_vec = np.load(gid_file.format(hemi=hemi))
        gid_idx_vec = np.logical_or(gid_vec == 1, gid_vec == 2)
        subjs_id = subjs_1080[gid_idx_vec]
        flag = subjs_twin.issubset(subjs_id)
        if flag:
            print('All twins are in G1 and G2')
        else:
            print("Filter twins who are not in G1 or G2.")
            trg_file_tmp = trg_file.format(hemi=hemi)
            h2.filter_twins_id(data=twins_df, limit_set=subjs_id,
                               trg_file=trg_file_tmp)
            h2.count_twins_id(trg_file_tmp)


def count_gender(twin_file):

    # inputs
    gender_file = '/nfs/m1/hcp/S1200_behavior_restricted.csv'

    # prepare
    twin_df = pd.read_csv(twin_file)
    gender_df = pd.read_csv(gender_file, index_col='Subject')
    twin_ids = twin_df['twin1'].to_list() + twin_df['twin2'].to_list()

    # calculate
    gender_df = gender_df.loc[twin_ids]
    genders = gender_df['Gender'].to_list()
    print('#Male:', genders.count('M'))
    print('#Female:', genders.count('F'))


def map_twinsID_to_groupID():
    """
    Find corresponding group ID for each twin.
    """
    import numpy as np
    import pandas as pd

    gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     'grouping/group_id_{hemi}_v2_merged.npy')
    twins_file = pjoin(work_dir, 'twins_id_1080.csv')
    subjs_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    out_file = pjoin(work_dir, 'twins_gid-v2-merged_1080.csv')

    df = pd.read_csv(twins_file)
    subjs_id = [int(_) for _ in open(subjs_file).read().splitlines()]

    for hemi in ('lh', 'rh'):
        gids = np.load(gid_file.format(hemi=hemi))
        for col in ('twin1', 'twin2'):
            col_new = f'{col}_gid_{hemi}'
            for row in df.index:
                subj_idx = subjs_id.index(df.loc[row, col])
                df.loc[row, col_new] = gids[subj_idx]
    df.to_csv(out_file, index=False)


def plot_twinsID_distribution_G0G1G2():
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width, plot_stacked_bar

    hemis = ('lh', 'rh')
    hemi2color = {'lh': (0.33, 0.33, 0.33, 1), 'rh': (0.66, 0.66, 0.66, 1)}
    gids = (0, 1, 2)
    zygosity = ('MZ', 'DZ')
    twins_gid_file = pjoin(work_dir, 'twins_gid_1080.csv')
    gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                     'grouping/group_id_{hemi}.npy')

    n_hemi = len(hemis)
    n_gid = len(gids)
    n_zyg = len(zygosity)
    df = pd.read_csv(twins_gid_file)
    zyg2indices = {}
    for zyg in zygosity:
        zyg2indices[zyg] = df['zygosity'] == zyg

    x = np.arange(n_zyg)
    width = auto_bar_width(x, n_hemi)
    offset = -(n_hemi - 1) / 2
    _, axes = plt.subplots(1, 2, figsize=(12.8, 7))
    for hemi in hemis:
        cols = [f'twin1_gid_{hemi}', f'twin2_gid_{hemi}']
        ys = np.zeros((n_gid, n_zyg))
        gid_vec = np.load(gid_file.format(hemi=hemi))
        n_subjs = np.zeros((n_gid, 1))
        for zyg_idx, zyg in enumerate(zygosity):
            data = np.array(df.loc[zyg2indices[zyg], cols])
            for gid_idx, gid in enumerate(gids):
                ys[gid_idx, zyg_idx] = np.sum(data == gid)
                if zyg_idx == 0:
                    n_subjs[gid_idx, 0] = np.sum(gid_vec == gid)
        x_tmp = x + width * offset
        offset += 1
        labels = [f'G0_{hemi}', f'G1_{hemi}', f'G2_{hemi}']
        face_colors = ['w', 'w', hemi2color[hemi]]
        hatchs = ['//', '*', None]

        ax = axes[0]
        plot_stacked_bar(x_tmp, ys, width, label=labels,
                         ec=hemi2color[hemi], fc=face_colors,
                         hatch=hatchs, ax=ax)

        ax = axes[1]
        plot_stacked_bar(x_tmp, ys / n_subjs, width, label=labels,
                         ec=hemi2color[hemi], fc=face_colors,
                         hatch=hatchs, ax=ax)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(zygosity)
    axes[0].set_ylabel('the number of subjects')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(zygosity)
    axes[1].set_ylabel('the ratio of subjects')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def count_twin_pair_same_group():
    """
    Count the number of twin pairs according to grouping.
    """
    import numpy as np
    import pandas as pd

    hemis = ('lh', 'rh')
    # gids = (-1, 0, 1, 2)
    # rows = ('diff', 'G-1', 'G0', 'G1', 'G2', 'total', 'total_G12')
    gids = (1, 2)
    rows = ('diff', 'G1', 'G2', 'total', 'total_G12')
    zygosity = ('MZ', 'DZ')
    twins_gid_file = pjoin(work_dir, 'twins_gid-v2-merged_1080.csv')
    out_file = pjoin(work_dir, 'count_if_same_group-v2-merged.csv')

    n_row = len(rows)
    df = pd.read_csv(twins_gid_file)
    out_dict = {}
    for hemi in hemis:
        items = [f'twin1_gid_{hemi}', f'twin2_gid_{hemi}']
        for zyg in zygosity:
            col = f'{zyg}_{hemi}'
            out_dict[col] = np.zeros(n_row)
            data = np.array(df.loc[df['zygosity'] == zyg, items])
            out_dict[col][0] = np.sum(data[:, 0] != data[:, 1])
            for gid_idx, gid in enumerate(gids, 1):
                out_dict[col][gid_idx] = np.sum(np.all(data == gid,
                                                       axis=1))
            out_dict[col][gid_idx+1] = data.shape[0]
            out_dict[col][gid_idx+2] = np.sum(~np.any(data < 1, axis=1))
    out_df = pd.DataFrame(out_dict, index=rows)
    out_df.to_csv(out_file, index=True)


def plot_probability_if_twins_belong_to_same_group():
    """
    Plot the probability whether a twin pair both belong to the same group
    """
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width, plot_stacked_bar

    hemis = ('lh', 'rh')
    gids = (0, 1, 2)
    hatchs = [None, '//', '*']  # [None, '//', '*']
    limit_name = f"limit_G{''.join(map(str, gids))}"
    zygosity = ('MZ', 'DZ')
    zyg2color = {'MZ': (0.33, 0.33, 0.33, 1), 'DZ': (0.66, 0.66, 0.66, 1)}
    count_file = pjoin(work_dir, 'count_if_same_group.csv')

    n_hemi = len(hemis)
    n_gid = len(gids)
    n_zyg = len(zygosity)
    df = pd.read_csv(count_file, index_col=0)

    x = np.arange(n_hemi)
    width = auto_bar_width(x, n_zyg)
    offset = -(n_zyg - 1) / 2
    _, ax = plt.subplots(figsize=(12.8, 7))
    for zyg in zygosity:
        ys = np.zeros((n_gid, n_hemi))
        items = [f'{zyg}_lh', f'{zyg}_rh']
        labels = []
        for gid_idx, gid in enumerate(gids):
            ys[gid_idx] = df.loc[f'G{gid}', items]
            labels.append(f'G{gid}_{zyg}')
        ys = ys / np.array([df.loc[limit_name, items]])
        plot_stacked_bar(x + width * offset, ys, width, label=labels,
                         ec=zyg2color[zyg], fc='w', hatch=hatchs, ax=ax)
        offset += 1
        print(f'{zyg}_{hemis}:\n', ys)
        print(f'{zyg}_{hemis}:\n', np.sum(ys, 0))
    ax.set_xticks(x)
    ax.set_xticklabels(hemis)
    ax.set_ylabel('the ratio of twins')
    ax.legend()
    plt.tight_layout()
    plt.show()


def prepare_heritability_calculation_TMAV():
    """
    Preparation for heritability calculation for
    thickness, myelin, activation, and vertex area.
    """
    import pandas as pd
    import pickle as pkl

    hemis = ('lh', 'rh')
    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    zyg2label = {'MZ': 1, 'DZ': 3}
    subjs_1080_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    meas2file = {
        'thickness': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin'
                           '/structure/individual_thickness_{hemi}.pkl'),
        'myelin': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                        'structure/individual_myelin_{hemi}.pkl'),
        'activ': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                       'tfMRI/individual_activ_{hemi}.pkl'),
        'va': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                    'structure/individual_va_{hemi}.pkl')
    }
    twins_file = pjoin(work_dir, 'twins_id_1080.csv')
    out_file = pjoin(work_dir, 'pre-heritability_TMAV_individual.csv')

    # prepare data
    df = pd.read_csv(twins_file)
    subjs_1080 = [int(i) for i in open(subjs_1080_file).read().splitlines()]
    subj_indices1 = [subjs_1080.index(_) for _ in df['twin1']]
    subj_indices2 = [subjs_1080.index(_) for _ in df['twin2']]

    # preparing
    df_out = {'zyg': [zyg2label[_] for _ in df['zygosity']]}
    for meas_name, meas_file in meas2file.items():
        for hemi in hemis:
            data = pkl.load(open(meas_file.format(hemi=hemi), 'rb'))
            for roi in rois:
                col1 = f"{roi.split('-')[0]}_{meas_name}_{hemi}1"
                col2 = f"{roi.split('-')[0]}_{meas_name}_{hemi}2"
                roi_idx = data['roi'].index(roi)
                df_out[col1] = data['meas'][roi_idx][subj_indices1]
                df_out[col2] = data['meas'][roi_idx][subj_indices2]
            col1 = f'pFus_mFus_{meas_name}_{hemi}1'
            col2 = f'pFus_mFus_{meas_name}_{hemi}2'
            df_out[col1] = df_out[f'pFus_{meas_name}_{hemi}1'] - df_out[
                f'mFus_{meas_name}_{hemi}1']
            df_out[col2] = df_out[f'pFus_{meas_name}_{hemi}2'] - df_out[
                f'mFus_{meas_name}_{hemi}2']
    df_out = pd.DataFrame(df_out)
    df_out.to_csv(out_file, index=False)


def prepare_heritability_calculation_RSFC():
    """
    Preparation for heritability calculation for RSFC
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    hemis = ('lh', 'rh')
    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    zyg2label = {'MZ': 1, 'DZ': 3}
    subjs_1080_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    rsfc_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                      'rfMRI/rsfc_mpm2Cole_{hemi}.pkl')
    twins_file = pjoin(work_dir, 'twins_id_rfMRI.csv')
    out_file = pjoin(work_dir, 'pre-heritability_rsfc.csv')

    # prepare data
    df = pd.read_csv(twins_file)
    subjs_1080 = [int(i) for i in open(subjs_1080_file).read().splitlines()]
    subj_indices1 = [subjs_1080.index(_) for _ in df['twin1']]
    subj_indices2 = [subjs_1080.index(_) for _ in df['twin2']]

    # preparing
    df_out = {'zyg': [zyg2label[_] for _ in df['zygosity']]}
    for hemi in hemis:
        rsfc = pkl.load(open(rsfc_file.format(hemi=hemi), 'rb'))
        assert subjs_1080 == [int(_) for _ in rsfc['subject']]
        for trg_idx, trg_lbl in enumerate(rsfc['trg_label']):
            for roi in rois:
                col1 = f"{roi.split('-')[0]}_trg{trg_lbl}_{hemi}1"
                col2 = f"{roi.split('-')[0]}_trg{trg_lbl}_{hemi}2"
                df_out[col1] = rsfc[roi][subj_indices1, trg_idx]
                df_out[col2] = rsfc[roi][subj_indices2, trg_idx]
                assert np.all(~np.isnan(df_out[col1]))
                assert np.all(~np.isnan(df_out[col2]))
            col1 = f'pFus_mFus_trg{trg_lbl}_{hemi}1'
            col2 = f'pFus_mFus_trg{trg_lbl}_{hemi}2'
            df_out[col1] = df_out[f'pFus_trg{trg_lbl}_{hemi}1'] - df_out[
                f'mFus_trg{trg_lbl}_{hemi}1']
            df_out[col2] = df_out[f'pFus_trg{trg_lbl}_{hemi}2'] - df_out[
                f'mFus_trg{trg_lbl}_{hemi}2']
    df_out = pd.DataFrame(df_out)
    df_out.to_csv(out_file, index=False)


def prepare_heritability_calculation_meanRSFC():
    """
    Preparation for heritability calculation for mean RSFC across ColeNet
    """
    import numpy as np
    import pandas as pd
    import pickle as pkl

    hemis = ('lh', 'rh')
    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    zyg2label = {'MZ': 1, 'DZ': 3}
    subjs_1080_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    rsfc_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                      'rfMRI/rsfc_mpm2Cole_{hemi}.pkl')
    twins_file = pjoin(work_dir, 'twins_id_rfMRI.csv')
    out_file = pjoin(work_dir, 'pre-heritability_rsfc-mean.csv')

    # prepare data
    df = pd.read_csv(twins_file)
    subjs_1080 = [int(i) for i in open(subjs_1080_file).read().splitlines()]
    subj_indices1 = [subjs_1080.index(_) for _ in df['twin1']]
    subj_indices2 = [subjs_1080.index(_) for _ in df['twin2']]

    # preparing
    df_out = {'zygosity': df['zygosity'],
              'zyg': [zyg2label[_] for _ in df['zygosity']]}
    for hemi in hemis:
        rsfc = pkl.load(open(rsfc_file.format(hemi=hemi), 'rb'))
        for roi in rois:
            col1 = f"{roi.split('-')[0]}_{hemi}1"
            col2 = f"{roi.split('-')[0]}_{hemi}2"
            df_out[col1] = np.mean(rsfc[roi][subj_indices1], 1)
            df_out[col2] = np.mean(rsfc[roi][subj_indices2], 1)
        col1 = f'pFus_mFus_{hemi}1'
        col2 = f'pFus_mFus_{hemi}2'
        df_out[col1] = df_out[f'pFus_{hemi}1'] - df_out[f'mFus_{hemi}1']
        df_out[col2] = df_out[f'pFus_{hemi}2'] - df_out[f'mFus_{hemi}2']
    df_out = pd.DataFrame(df_out)
    df_out.to_csv(out_file, index=False)


def calc_Falconer_h2():
    """
    Calculate Falconer's heritability by using ICC.
    """
    import time
    import numpy as np
    import pandas as pd

    n_bootstrap = 10000
    confidence = 95
    data_file = pjoin(work_dir, 'pre-heritability_rsfc-mean.csv')
    out_file = pjoin(work_dir, 'heritability_icc_rsfc-mean.csv')

    data = pd.read_csv(data_file)
    mz_indices = data['zyg'] == 1
    dz_indices = data['zyg'] == 3
    var_names = [_[:-1] for _ in data.columns if _[-1] == '1']
    n_var = len(var_names)
    indices = ['ICC_MZ', 'ICC_DZ', 'r_mz_lb', 'r_mz', 'r_mz_ub', 'r_dz_lb',
               'r_dz', 'r_dz_ub', 'H2', 'h2_lb', 'h2', 'h2_ub']
    out_dict = {}
    for k in indices:
        out_dict[k] = np.zeros(n_var)
    for var_idx, var_name in enumerate(var_names):
        time1 = time.time()
        pair_name = [var_name + '1', var_name + '2']
        mz = np.array(data.loc[mz_indices, pair_name]).T
        mz = mz[:, ~np.any(np.isnan(mz), axis=0)]
        dz = np.array(data.loc[dz_indices, pair_name]).T
        dz = dz[:, ~np.any(np.isnan(dz), axis=0)]
        print(f'#MZ in {var_name}:', mz.shape[1])
        print(f'#DZ in {var_name}:', dz.shape[1])
        out_dict['ICC_MZ'][var_idx] = h2.icc(mz)
        out_dict['ICC_DZ'][var_idx] = h2.icc(dz)
        r_mz_lb, r_mz, r_mz_ub = h2.icc(mz,
                                        n_bootstrap=n_bootstrap,
                                        confidence=confidence)
        r_dz_lb, r_dz, r_dz_ub = h2.icc(dz,
                                        n_bootstrap=n_bootstrap,
                                        confidence=confidence)
        out_dict['r_mz_lb'][var_idx] = r_mz_lb
        out_dict['r_mz'][var_idx] = r_mz
        out_dict['r_mz_ub'][var_idx] = r_mz_ub
        out_dict['r_dz_lb'][var_idx] = r_dz_lb
        out_dict['r_dz'][var_idx] = r_dz
        out_dict['r_dz_ub'][var_idx] = r_dz_ub
        out_dict['H2'][var_idx] = h2.heritability(mz, dz)
        h2_lb, h, h2_ub = h2.heritability(mz, dz, n_bootstrap=n_bootstrap,
                                          confidence=confidence)
        out_dict['h2_lb'][var_idx] = h2_lb
        out_dict['h2'][var_idx] = h
        out_dict['h2_ub'][var_idx] = h2_ub
        print(f'Finish {var_idx+1}/{n_var}, spend {time.time()-time1} seconds')
    out_df = pd.DataFrame(np.array(list(out_dict.values())),
                          index=out_dict.keys(), columns=var_names)
    out_df.to_csv(out_file, index=True)


def plot_Falconer_h2():
    """
    thickness, myelin, face-avg, mean RSFC
    """
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width

    # figsize = (9, 4)
    figsize = (7, 4.8)
    out_file = pjoin(work_dir, 'Falconer_h2.jpg')
    meas2file = {
        'thickness': pjoin(work_dir, 'heritability_icc_TMA.csv'),
        'myelin': pjoin(work_dir, 'heritability_icc_TMA.csv'),
        'activ': pjoin(work_dir, 'heritability_icc_TMA.csv'),
        'rsfc': pjoin(work_dir, 'heritability_icc_rsfc-mean.csv')}
    zygosity = ('mz', 'dz')
    hemis = ('lh', 'rh')
    # rois = ('pFus', 'mFus', 'pFus_mFus')
    rois = ('pFus', 'mFus')
    roi2color = {
        'pFus': 'limegreen',
        'mFus': 'cornflowerblue',
        'pFus_mFus': 'black'}
    roi2label = {'pFus': 'pFus', 'mFus': 'mFus', 'pFus_mFus': 'pFus-mFus'}
    meas2title = {
        'thickness': 'thickness',
        'myelin': 'myelination',
        'activ': 'face selectivity',
        'va': 'surface area',
        'rsfc': 'RSFC'}
    meas2str = {
        'thickness': '_thickness_',
        'myelin': '_myelin_',
        'activ': '_activ_',
        'va': '_va_',
        'rsfc': '_'}

    n_roi = len(rois)
    n_hemi = len(hemis)
    n_meas = len(meas2file)
    x = np.arange(n_hemi)
    _, axes = plt.subplots(2, n_meas, figsize=figsize)

    # plot ICC
    n_item0 = 2 * n_roi
    width0 = auto_bar_width(x, n_item0)
    for meas_idx, meas_name in enumerate(meas2file.keys()):
        df = pd.read_csv(meas2file[meas_name], index_col=0)
        ax = axes[0, meas_idx]
        offset = -(n_item0 - 1) / 2
        for roi in rois:
            cols = [f'{roi}{meas2str[meas_name]}{hemi}' for hemi in hemis]
            for zyg in zygosity:
                lbl = roi2label[roi] + '_' + zyg
                y = np.array(df.loc[f'r_{zyg}', cols])
                low_err = y - np.array(df.loc[f'r_{zyg}_lb', cols])
                high_err = np.array(df.loc[f'r_{zyg}_ub', cols]) - y
                yerr = np.array([low_err, high_err])
                if zyg == 'mz':
                    ax.bar(x + width0 * offset, y, width0, yerr=yerr,
                           label=lbl, ec=roi2color[roi], fc='w',
                           hatch='//')
                    # ax.plot(x + width0 * offset,
                    #         np.array(df.loc['ICC_MZ', cols]),
                    #         linestyle='', marker='*', ms=10,
                    #         markeredgecolor='k', markerfacecolor='w')
                else:
                    ax.bar(x + width0 * offset, y, width0, yerr=yerr,
                           label=lbl, ec=roi2color[roi], fc='w',
                           hatch='\\')
                    # ax.plot(x + width0 * offset,
                    #         np.array(df.loc['ICC_DZ', cols]),
                    #         linestyle='', marker='*', ms=10,
                    #         markeredgecolor='k', markerfacecolor='w')
                offset += 1
        ax.set_title(meas2title[meas_name])
        ax.set_xticks(x)
        ax.set_xticklabels(hemis)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if meas_idx == 0:
            ax.set_ylabel('ICC')
        if meas_idx == 1:
            ax.legend()

    # plot heritability
    n_item1 = n_roi
    width1 = auto_bar_width(x, n_item1)
    for meas_idx, meas_name in enumerate(meas2file.keys()):
        df = pd.read_csv(meas2file[meas_name], index_col=0)
        ax = axes[1, meas_idx]
        offset = -(n_item1 - 1) / 2
        for roi in rois:
            cols = [f'{roi}{meas2str[meas_name]}{hemi}' for hemi in hemis]
            y = np.array(df.loc['h2', cols])
            low_err = y - np.array(df.loc['h2_lb', cols])
            high_err = np.array(df.loc['h2_ub', cols]) - y
            yerr = np.array([low_err, high_err])
            # ax.bar(x + width1 * offset, y, width1, yerr=yerr,
            #        label=roi2label[roi], ec=roi2color[roi], fc='w',
            #        hatch='//')
            ax.bar(x + width1 * offset, y, width1, yerr=yerr,
                   label=roi2label[roi], color=roi2color[roi])
            # ax.plot(x + width1 * offset,
            #         np.array(df.loc['H2', cols]),
            #         linestyle='', marker='*', ms=10,
            #         markeredgecolor='k', markerfacecolor='w')
            offset += 1
        ax.set_xticks(x)
        ax.set_xticklabels(hemis)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if meas_idx == 0:
            ax.set_ylabel('heritability')
        # if meas_idx == 1:
        #     ax.legend()

    plt.tight_layout()
    plt.savefig(out_file)
    # plt.show()


def calc_pattern_corr_between_twins(meas_name, mpm_file, twins_id_file, out_file):
    """
    Calculate spatial pattern correlation between each twin pair.
    """
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    rois = ('pFus-faces', 'mFus-faces')

    key2name = {0: '???', 1: 'R_pFus-faces', 2: 'R_mFus-faces',
                3: 'L_pFus-faces', 4: 'L_mFus-faces'}
    name2key = {}
    for k, n in key2name.items():
        name2key[n] = k

    meas2file = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
        'activ': pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    }

    df = pd.read_csv(twins_id_file)
    meas_reader = CiftiReader(meas2file[meas_name])
    meas_ids = [int(name.split('_')[0]) for name in meas_reader.map_names()]
    twin1_indices = [meas_ids.index(i) for i in df['twin1']]
    twin2_indices = [meas_ids.index(i) for i in df['twin2']]
    meas_maps = meas_reader.get_data()
    meas1 = meas_maps[twin1_indices]
    meas2 = meas_maps[twin2_indices]
    mpm_reader = CiftiReader(mpm_file)
    mpm_map = mpm_reader.get_data()[0]
    assert mpm_map.shape[0] == meas_maps.shape[1]
    lbl_tab = mpm_reader.label_tables()[0]

    out_df = pd.DataFrame()
    out_df['zygosity'] = df['zygosity']
    out_df['zyg'] = [zyg2label[zyg] for zyg in df['zygosity']]
    for hemi in hemis:
        for roi in rois:
            name = f'{hemi2Hemi[hemi]}_{roi}'
            key = name2key[name]
            assert lbl_tab[key].label == name
            idx_vec = mpm_map == key
            out_df[f"{hemi}_{roi.split('-')[0]}"] = [
                pearsonr(i[idx_vec], j[idx_vec])[0]
                for i, j in zip(meas1, meas2)]

    out_df.to_csv(out_file, index=False)


def calc_pattern_corr_between_twins_rsfc(subj_id_file, twins_id_file, rsfc_file, out_file):
    """
    Calculate connectivity pattern correlation between each twin pair.
    """
    # inputs
    hemis = ('lh', 'rh')
    rois = ('pFus-face', 'mFus-face')

    # prepare
    df = pd.read_csv(twins_id_file)
    subj_ids = [int(i) for i in open(subj_id_file).read().splitlines()]
    twin1_indices = [subj_ids.index(i) for i in df['twin1']]
    twin2_indices = [subj_ids.index(i) for i in df['twin2']]

    # calculate
    out_df = pd.DataFrame()
    out_df['zygosity'] = df['zygosity']
    out_df['zyg'] = [zyg2label[zyg] for zyg in df['zygosity']]
    for hemi in hemis:
        rsfc_dict = pkl.load(open(rsfc_file.format(hemi=hemi), 'rb'))
        assert subj_ids == [int(_) for _ in rsfc_dict['subject']]
        for roi in rois:
            rsfc1 = rsfc_dict[roi][twin1_indices]
            rsfc2 = rsfc_dict[roi][twin2_indices]
            out_df[f"{hemi}_{roi.split('-')[0]}"] = [
                pearsonr(i, j)[0] for i, j in zip(rsfc1, rsfc2)]

    # save
    out_df.to_csv(out_file, index=False)


def pattern_corr_fisher_z(meas='thickness'):
    import numpy as np
    import pandas as pd

    # inputs
    hemis = ('lh', 'rh')
    rois = ('IOG', 'pFus', 'mFus')
    df_file = pjoin(work_dir, f'twins_pattern-corr_{meas}.csv')

    # outputs
    out_file = df_file.split('.')[0] + '_fisherZ.csv'

    df = pd.read_csv(df_file)
    for hemi in hemis:
        for roi in rois:
            col = f'{hemi}_{roi}'
            df[col] = np.arctanh(df[col])

    df.to_csv(out_file, index=False)


def plot_pattern_corr1():
    import numpy as np
    import pandas as pd
    from scipy.stats import sem
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width

    rois = ('pFus', 'mFus')
    meas_names = ('thickness', 'myelin', 'activ', 'rsfc')
    meas2ylabel = {
        'thickness': 'thickness',
        'myelin': 'myelin',
        'activ': 'face-avg',
        'rsfc': 'RSFC'}
    zygosity = ('MZ', 'DZ')
    zyg2color = {'MZ': (0.33, 0.33, 0.33, 1), 'DZ': (0.66, 0.66, 0.66, 1)}
    hemis = ('lh', 'rh')
    df_file = pjoin(work_dir, 'twins_pattern-corr_{}.csv')

    # outputs
    out_file = pjoin(work_dir, 'pattern_corr1.jpg')

    n_zyg = len(zygosity)
    x = np.arange(len(rois))
    width = auto_bar_width(x, n_zyg)
    _, axes = plt.subplots(len(hemis), len(meas_names), figsize=(6.4, 4.8))
    for meas_idx, meas_name in enumerate(meas_names):
        df = pd.read_csv(df_file.format(meas_name))
        for hemi_idx, hemi in enumerate(hemis):
            ax = axes[hemi_idx, meas_idx]
            offset = -(n_zyg - 1) / 2
            for zyg in zygosity:
                indices = df['zygosity'] == zyg
                cols = [f'{hemi}_{roi}' for roi in rois]
                data = np.array(df.loc[indices, cols])
                y = np.mean(data, 0)
                yerr = sem(data, 0)
                ax.bar(x + width * offset, y, width, yerr=yerr,
                       label=zyg, color=zyg2color[zyg])
                offset += 1
            ax.set_ylabel(meas2ylabel[meas_name])
            # if meas_idx == 1:
            #     ax.set_title(hemi)
            #     if hemi_idx == 0:
            #         ax.legend()
            ax.set_xticks(x)
            ax.set_xticklabels(rois)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_file)
    # plt.show()


def plot_pattern_corr2():
    import numpy as np
    import pandas as pd
    from scipy.stats import ttest_ind
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width
    from commontool.stats import EffectSize

    rois = ('pFus', 'mFus')
    roi2color = {'pFus': 'limegreen', 'mFus': 'cornflowerblue'}
    meas_names = ('thickness', 'myelin', 'activ')
    meas2ylabel = {
        'thickness': 'thickness',
        'myelin': 'myelin',
        'activ': 'face-avg'
    }
    hemis = ('lh', 'rh')

    n_roi = len(rois)
    df_file = pjoin(work_dir, 'twins_pattern-corr_{}.csv')

    x = np.arange(len(hemis))
    width = auto_bar_width(x, n_roi)
    _, axes = plt.subplots(2, len(meas_names))
    es = EffectSize()
    for meas_idx, meas_name in enumerate(meas_names):
        print(f'---{meas2ylabel[meas_name]}---')
        ax1 = axes[0, meas_idx]
        ax2 = axes[1, meas_idx]
        df = pd.read_csv(df_file.format(meas_name))
        offset = -(n_roi - 1) / 2
        for roi in rois:
            ts = []
            ps = []
            ds = []
            for hemi in hemis:
                col = f'{hemi}_{roi}'
                s1 = np.array(df.loc[df['zygosity'] == 'MZ', col])
                s2 = np.array(df.loc[df['zygosity'] == 'DZ', col])
                t, p = ttest_ind(s1, s2)
                d = es.cohen_d(s1, s2)
                ts.append(t)
                ps.append(p)
                ds.append(d)
                print(f'P value of {col}:', p)
            ax1.bar(x + width * offset, ts, width, label=roi,
                    color=roi2color[roi])
            ax2.bar(x + width * offset, ds, width, label=roi,
                    color=roi2color[roi])
            offset += 1
        ax1.set_ylabel(meas2ylabel[meas_name])
        ax1.set_xticks(x)
        ax1.set_xticklabels(hemis)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.set_ylabel(meas2ylabel[meas_name])
        ax2.set_xticks(x)
        ax2.set_xticklabels(hemis)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        if meas_idx == 1:
            # ax1.legend()
            ax1.set_title('t-value')
            ax2.set_title("Cohen's D")
    plt.tight_layout()
    plt.show()


def plot_SEM_h2_TMA():
    """
    thickness, myelin, and activation
    """
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width

    df_file = pjoin(work_dir, 'AE-h2estimate_TMA.csv')
    hemis = ('lh', 'rh')
    rois = ('pFus', 'mFus', 'pFus_mFus')
    roi2color = {
        'pFus': 'limegreen',
        'mFus': 'cornflowerblue',
        'pFus_mFus': 'black'
    }
    meas2title = {
        'thickness': 'thickness',
        'myelin': 'myelin',
        'activ': 'face-avg'
    }

    n_hemi = len(hemis)
    n_roi = len(rois)
    n_meas = len(meas2title)
    df = pd.read_csv(df_file)

    x = np.arange(n_hemi)
    width = auto_bar_width(x, n_roi)
    _, axes = plt.subplots(1, n_meas)
    for meas_idx, meas_name in enumerate(meas2title.keys()):
        ax = axes[meas_idx]
        offset = -(n_roi - 1) / 2
        for roi in rois:
            cols = [f'{roi}_{meas_name}_{hemi}' for hemi in hemis]
            data = np.array(df[cols])
            y = data[1]
            low_err = y - data[0]
            high_err = data[2] - y
            yerr = np.array([low_err, high_err])
            ax.bar(x + width * offset, y, width, yerr=yerr,
                   label=roi, ec=roi2color[roi], fc='w',
                   hatch='//')
            offset += 1
        ax.set_title(meas2title[meas_name])
        ax.set_xticks(x)
        ax.set_xticklabels(hemis)
        if meas_idx == 0:
            ax.set_ylabel('heritability')
        if meas_idx == 1:
            ax.legend()
    plt.tight_layout()
    plt.show()


def plot_SEM_h2_RSFC():
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from nibrain.util.plotfig import auto_bar_width

    df_file = pjoin(work_dir, 'AE-h2estimate_rsfc.csv')
    hemis = ('lh', 'rh')
    rois = ('pFus', 'mFus', 'pFus_mFus')
    roi2color = {
        'pFus': 'limegreen',
        'mFus': 'cornflowerblue',
        'pFus_mFus': 'black'
    }
    trg_config_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/'\
                      'network_labelfile.txt'

    n_hemi = len(hemis)
    n_roi = len(rois)
    df = pd.read_csv(df_file)
    rf = open(trg_config_file)
    trg_names = []
    trg_labels = []
    while True:
        trg_name = rf.readline()
        if trg_name == '':
            break
        trg_names.append(trg_name.rstrip('\n'))
        trg_labels.append(int(rf.readline().split(' ')[0]))
    indices_sorted = np.argsort(trg_labels)
    trg_names = np.array(trg_names)[indices_sorted].tolist()
    trg_labels = np.array(trg_labels)[indices_sorted].tolist()
    n_trg = len(trg_names)
    print(trg_names)

    x = np.arange(n_hemi * n_trg)
    width = auto_bar_width(x, n_roi) / 1.5
    offset = -(n_roi - 1) / 2
    for roi in rois:
        cols = [f'{roi}_trg{trg_lbl}_{hemi}' for trg_lbl in trg_labels
                for hemi in hemis]
        data = np.array(df[cols])
        y = data[1]
        low_err = y - data[0]
        high_err = data[2] - y
        yerr = np.array([low_err, high_err])
        plt.bar(x + width * offset, y, width, yerr=yerr,
                label=roi, ec=roi2color[roi], fc='w',
                hatch='//')
        offset += 1
    plt.xticks(x, hemis * n_trg)
    plt.ylabel('heritability')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # get_twinsID()
    # filter_twinsID_1080()
    # filter_twinsID_1053()
    # filter_twinsID_rfMRI(
    #     twins_file=pjoin(work_dir, 'twins_id.csv'),
    #     subjs_file=pjoin(anal_dir, 'rfMRI/rfMRI_REST_id'),
    #     trg_file = pjoin(work_dir, 'twins_id_rfMRI.csv'))
    # filter_twinsID_rfMRI(
    #     twins_file=pjoin(work_dir, 'twins_id.csv'),
    #     subjs_file=pjoin(anal_dir, 'subj_info/subject_id2.txt'),
    #     trg_file = pjoin(anal_dir, 'NI_R1/data_1053/twins_id_rfMRI.csv'))
    # filter_twinsID_G1G2()
    # count_gender(
    #     twin_file=pjoin(work_dir, 'twins_id_1080.csv'))
    count_gender(
        twin_file=pjoin(anal_dir, 'NI_R1/data_1053/twins_id_1053.csv'))
    # map_twinsID_to_groupID()
    # plot_twinsID_distribution_G0G1G2()
    # count_twin_pair_same_group()
    # plot_probability_if_twins_belong_to_same_group()
    # prepare_heritability_calculation_TMAV()
    # prepare_heritability_calculation_RSFC()
    # prepare_heritability_calculation_meanRSFC()
    # calc_Falconer_h2()
    # plot_Falconer_h2()

    # calc_pattern_corr_between_twins(meas_name='thickness')
    # calc_pattern_corr_between_twins(meas_name='myelin')
    # calc_pattern_corr_between_twins(meas_name='activ')
    # calc_pattern_corr_between_twins(
    #     meas_name='thickness',
    #     mpm_file=pjoin(anal_dir, 'HCP-YA_FFA-MPM_thr-25.32k_fs_LR.dlabel.nii'),
    #     twins_id_file=pjoin(work_dir, 'twins_id_1080.csv'),
    #     out_file=pjoin(work_dir, 'twins_pattern-corr_thickness_new.csv'))
    # calc_pattern_corr_between_twins(
    #     meas_name='thickness',
    #     mpm_file=pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-MPM_thr-25.32k_fs_LR.dlabel.nii'),
    #     twins_id_file=pjoin(anal_dir, 'NI_R1/data_1053/twins_id_1053.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/twins_pattern-corr_thickness.csv'))
    # calc_pattern_corr_between_twins(
    #     meas_name='myelin',
    #     mpm_file=pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-MPM_thr-25.32k_fs_LR.dlabel.nii'),
    #     twins_id_file=pjoin(anal_dir, 'NI_R1/data_1053/twins_id_1053.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/twins_pattern-corr_myelin.csv'))
    # calc_pattern_corr_between_twins(
    #     meas_name='activ',
    #     mpm_file=pjoin(anal_dir, 'NI_R1/data_1053/HCP-YA_FFA-MPM_thr-25.32k_fs_LR.dlabel.nii'),
    #     twins_id_file=pjoin(anal_dir, 'NI_R1/data_1053/twins_id_1053.csv'),
    #     out_file=pjoin(anal_dir, 'NI_R1/data_1053/twins_pattern-corr_activ.csv'))

    # pattern_corr_fisher_z(meas='thickness')
    # pattern_corr_fisher_z(meas='myelin')
    # pattern_corr_fisher_z(meas='activ')
    # plot_pattern_corr1()
    # plot_pattern_corr2()
    # plot_SEM_h2_TMA()
    # plot_SEM_h2_RSFC()

    # calc_pattern_corr_between_twins_rsfc(
    #     subj_id_file=pjoin(proj_dir, 'analysis/s2/subject_id'),
    #     twins_id_file=pjoin(work_dir, 'twins_id_rfMRI.csv'),
    #     rsfc_file=pjoin(anal_dir, 'rfMRI/rsfc_mpm2Cole_{hemi}.pkl'),
    #     out_file = pjoin(work_dir, 'twins_pattern-corr_rsfc.csv'))
    # calc_pattern_corr_between_twins_rsfc(
    #     subj_id_file=pjoin(proj_dir, 'analysis/s2/subject_id'),
    #     twins_id_file=pjoin(anal_dir, 'NI_R1/data_1053/twins_id_rfMRI.csv'),
    #     rsfc_file=pjoin(anal_dir, 'rfMRI/rsfc_1053mpm2Cole_{hemi}.pkl'),
    #     out_file = pjoin(anal_dir, 'NI_R1/data_1053/twins_pattern-corr_rsfc.csv'))
