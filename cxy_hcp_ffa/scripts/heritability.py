# %% Initialization
import time
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import pearsonr, sem
from matplotlib import pyplot as plt
from nibrain.util.plotfig import auto_bar_width, plot_stacked_bar
from cxy_hcp_ffa.lib import heritability as h2
from commontool.io.io import CiftiReader

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir,
                 'analysis/s2/1080_fROI/refined_with_Kevin/heritability')

# %% acquire twins ID
h2.get_twins_id(src_file='/nfs/m1/hcp/S1200_behavior_restricted.csv',
                trg_file=pjoin(work_dir, 'twins_id.csv'))
h2.count_twins_id(data=pjoin(work_dir, 'twins_id.csv'))

# %% filter twins ID
twins_df = pd.read_csv(pjoin(work_dir, 'twins_id.csv'))
subjs_twin = set(np.concatenate([twins_df['twin1'], twins_df['twin2']]))

# %%% check if it's a subset of 1080 subjects
subjs_file = pjoin(proj_dir, 'analysis/s2/subject_id')
subjs_id = set([int(_) for _ in open(subjs_file).read().splitlines()])
flag = subjs_twin.issubset(subjs_id)
if flag:
    print('All twins is a subset of 1080 subjects.')
else:
    print('Filter twins which are not in 1080 subjects.')
    h2.filter_twins_id(data=twins_df, limit_set=subjs_id,
                       trg_file=pjoin(work_dir, 'twins_id_1080.csv'))
    h2.count_twins_id(pjoin(work_dir, 'twins_id_1080.csv'))

# %%% check if the subject have all 4 rfMRI runs.
subjs_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                   'rfMRI/rfMRI_REST_id')
subjs_id = set([int(_) for _ in open(subjs_file).read().splitlines()])
flag = subjs_twin.issubset(subjs_id)
if flag:
    print('All twins have all 4 rfMRI runs')
else:
    print("Filter twins which don't have all 4 rfMRI runs.")
    h2.filter_twins_id(data=twins_df, limit_set=subjs_id,
                       trg_file=pjoin(work_dir, 'twins_id_rfMRI.csv'))
    h2.count_twins_id(pjoin(work_dir, 'twins_id_rfMRI.csv'))

# %%% check if the subject is in G1 or G2
hemis = ('lh', 'rh')
gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                'grouping/group_id_{hemi}.npy')
subjs_file = pjoin(proj_dir, 'analysis/s2/subject_id')
subjs_1080 = np.array([int(_) for _ in open(subjs_file).read().splitlines()])
for hemi in hemis:
    gid_vec = np.load(gid_file.format(hemi=hemi))
    gid_idx_vec = np.logical_or(gid_vec==1, gid_vec==2)
    subjs_id = subjs_1080[gid_idx_vec]
    flag = subjs_twin.issubset(subjs_id)
    if flag:
        print('All twins are in G1 and G2')
    else:
        print("Filter twins who are not in G1 or G2.")
        h2.filter_twins_id(data=twins_df, limit_set=subjs_id,
                           trg_file=pjoin(work_dir, f'twins_id_G1G2_{hemi}.csv'))
        h2.count_twins_id(pjoin(work_dir, f'twins_id_G1G2_{hemi}.csv'))

# %% twins ID distribution in G0, G1, and G2
gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                 'grouping/group_id_{hemi}.npy')
twins_id_file = pjoin(work_dir, 'twins_id_1080.csv')
subjs_file = pjoin(proj_dir, 'analysis/s2/subject_id')
out_file = pjoin(work_dir, 'twins_gid_1080.csv')

df = pd.read_csv(twins_id_file)
subjs_id = [int(_) for _ in open(subjs_file).read().splitlines()]

for hemi in ('lh', 'rh'):
    gids = np.load(gid_file.format(hemi=hemi))
    for col in ('twin1', 'twin2'):
        col_new = f'{col}_gid_{hemi}'
        for row in df.index:
            subj_idx = subjs_id.index(df.loc[row, col])
            df.loc[row, col_new] = gids[subj_idx]
df.to_csv(out_file, index=False)

# %% plot gourp info about twins
hemis = ('lh', 'rh')
n_hemi = len(hemis)
hemi2color = {'lh': (0.33, 0.33, 0.33, 1),
              'rh': (0.66, 0.66, 0.66, 1)}
gids = (0, 1, 2)
n_gid = len(gids)
zygosity = ('MZ', 'DZ')
n_zyg = len(zygosity)
twins_gid_file = pjoin(work_dir, 'twins_gid_1080.csv')
gid_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                 'grouping/group_id_{hemi}.npy')

df = pd.read_csv(twins_gid_file)
zyg2indices = {}
for zyg in zygosity:
    zyg2indices[zyg] = df['zygosity'] == zyg

x = np.arange(n_zyg)
width = auto_bar_width(x, n_hemi)
offset = -(n_hemi - 1) / 2
fig, axes = plt.subplots(1, 2, figsize=(12.8, 7))
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
    plot_stacked_bar(x_tmp, ys, width, label=labels, ec=hemi2color[hemi],
                     fc=face_colors, hatch=hatchs, ax=ax)

    ax = axes[1]
    plot_stacked_bar(x_tmp, ys/n_subjs, width, label=labels,
                     ec=hemi2color[hemi], fc=face_colors, hatch=hatchs,
                     ax=ax)
axes[0].set_xticks(x)
axes[0].set_xticklabels(zygosity)
axes[0].set_ylabel('the number of subjects')
axes[1].set_xticks(x)
axes[1].set_xticklabels(zygosity)
axes[1].set_ylabel('the ratio of subjects')
axes[1].legend()
plt.tight_layout()
plt.show()

# %% count the number of twin pairs according to grouping
hemis = ('lh', 'rh')
gids = (0, 1, 2)
rows = ('diff', 'G0', 'G1', 'G2', 'limit_G012', 'limit_G12')
n_row = len(rows)
zygosity = ('MZ', 'DZ')
twins_gid_file = pjoin(work_dir, 'twins_gid_1080.csv')
out_file = pjoin(work_dir, 'count_if_same_group.csv')

df = pd.read_csv(twins_gid_file)
out_dict = {}
for hemi in hemis:
    items = [f'twin1_gid_{hemi}', f'twin2_gid_{hemi}']
    for zyg in zygosity:
        col = f'{zyg}_{hemi}'
        out_dict[col] = np.zeros(n_row)
        data = np.array(df.loc[df['zygosity']==zyg, items])
        out_dict[col][0] = np.sum(data[:, 0] != data[:, 1])
        for gid_idx, gid in enumerate(gids):
            out_dict[col][gid_idx+1] = np.sum(np.all(data == gid, axis=1))
        out_dict[col][4] = data.shape[0]
        out_dict[col][5] = np.sum(~np.any(data == 0, axis=1))
out_df = pd.DataFrame(out_dict, index=rows)
out_df.to_csv(out_file, index=True)

# %% plot the probability whether a twin pair both belong to the same group
hemis = ('lh', 'rh')
n_hemi = len(hemis)
gids = (0, 1, 2)
n_gid = len(gids)
hatchs = [None, '//', '*']  # [None, '//', '*']
limit_name = f"limit_G{''.join(map(str, gids))}"
zygosity = ('MZ', 'DZ')
n_zyg = len(zygosity)
zyg2color = {'MZ': (0.33, 0.33, 0.33, 1),
             'DZ': (0.66, 0.66, 0.66, 1)}
df = pd.read_csv(pjoin(work_dir, 'count_if_same_group.csv'), index_col=0)

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
    plot_stacked_bar(x+width*offset, ys, width, label=labels, ec=zyg2color[zyg],
                     fc='w', hatch=hatchs, ax=ax)
    offset += 1
    print(f'{zyg}_{hemis}:\n', ys)
    print(f'{zyg}_{hemis}:\n', np.sum(ys, 0))
ax.set_xticks(x)
ax.set_xticklabels(hemis)
ax.set_ylabel('the ratio of twins')
ax.legend()
plt.tight_layout()
plt.show()

# %% preparation for heritability calculation
hemis = ('lh', 'rh')
rois = ('IOG-face', 'pFus-face', 'mFus-face')
zyg2label = {'MZ': 1, 'DZ': 3}
subjs_1080_file = pjoin(proj_dir, 'analysis/s2/subject_id')
subjs_1080 = [int(i) for i in open(subjs_1080_file).read().splitlines()]

# %%% thickness, myelin, activation
# prepare parameters
meas2file = {
        'thickness': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin'
                           '/structure/MPM_v3_{hemi}_0.25_thickness.pkl'),
        'myelin': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                        'structure/MPM_v3_{hemi}_0.25_myelin.pkl'),
        'activ': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                       'tfMRI/MPM_v3_{hemi}_0.25_activ.pkl')
        }
twins_file = pjoin(work_dir, 'twins_id_1080.csv')
out_file = pjoin(work_dir, 'pre-heritability_TMA.csv')

# prepare data
df = pd.read_csv(twins_file)
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
        df_out[col1] = df_out[f'pFus_{meas_name}_{hemi}1'] - df_out[f'mFus_{meas_name}_{hemi}1']
        df_out[col2] = df_out[f'pFus_{meas_name}_{hemi}2'] - df_out[f'mFus_{meas_name}_{hemi}2']
df_out = pd.DataFrame(df_out)
df_out.to_csv(out_file, index=False)

# %%% RSFC
# prepare parameters
rsfc_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                  'rfMRI/rsfc_mpm2Cole_{hemi}.pkl')
twins_file = pjoin(work_dir, 'twins_id_rfMRI.csv')
out_file = pjoin(work_dir, 'pre-heritability_rsfc.csv')

# prepare data
df = pd.read_csv(twins_file)
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
        df_out[col1] = df_out[f'pFus_trg{trg_lbl}_{hemi}1'] - df_out[f'mFus_trg{trg_lbl}_{hemi}1']
        df_out[col2] = df_out[f'pFus_trg{trg_lbl}_{hemi}2'] - df_out[f'mFus_trg{trg_lbl}_{hemi}2']
df_out = pd.DataFrame(df_out)
df_out.to_csv(out_file, index=False)

# %% calculate heritability by using ICC
n_bootstrap = 10000
confidence = 95
data_file = pjoin(work_dir, 'pre-heritability_rsfc.csv')
out_file = pjoin(work_dir, 'heritability_icc_rsfc.csv')

data = pd.read_csv(data_file)
mz_indices = data['zyg'] == 1
dz_indices = data['zyg'] == 3
var_names = [_[:-1] for _ in data.columns if _[-1] == '1']
n_var = len(var_names)
indices = ['ICC_MZ', 'ICC_DZ',
           'r_mz_lb', 'r_mz', 'r_mz_ub',
           'r_dz_lb', 'r_dz', 'r_dz_ub',
           'H2', 'h2_lb', 'h2', 'h2_ub']
out_dict = {}
for k in indices:
    out_dict[k] = np.zeros(n_var)
for var_idx, var_name in enumerate(var_names):
    time1 = time.time()
    pair_name = [var_name+'1', var_name+'2']
    mz = np.array(data.loc[mz_indices, pair_name]).T
    dz = np.array(data.loc[dz_indices, pair_name]).T
    out_dict['ICC_MZ'][var_idx] = h2.icc(mz)
    out_dict['ICC_DZ'][var_idx] = h2.icc(dz)
    r_mz_lb, r_mz, r_mz_ub = h2.icc(mz, n_bootstrap=n_bootstrap,
                                    confidence=confidence)
    r_dz_lb, r_dz, r_dz_ub = h2.icc(dz, n_bootstrap=n_bootstrap,
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
    print(f'Finish {var_idx+1}/{n_var}, spend {time.time()-time1} seconds.')
out_df = pd.DataFrame(np.array(list(out_dict.values())),
                      index=out_dict.keys(), columns=var_names)
out_df.to_csv(out_file, index=True)

# %% plot results from ICC heritability
# %%% thickness, myelin, face-avg
df = pd.read_csv(pjoin(work_dir, 'heritability_icc_TMA.csv'), index_col=0)
zygosity = ('mz', 'dz')
hemis = ('lh', 'rh')
n_hemi = len(hemis)
rois = ('pFus', 'mFus', 'pFus_mFus')
roi2color = {'pFus': 'limegreen', 'mFus': 'cornflowerblue', 'pFus_mFus': 'black'}
roi2label = {'pFus': 'pFus', 'mFus': 'mFus', 'pFus_mFus': 'pFus-mFus'}
n_roi = len(rois)
meas2title = {'thickness': 'thickness', 'myelin': 'myelin', 'activ': 'face-avg'}
n_meas = len(meas2title)

x = np.arange(n_hemi)
fig, axes = plt.subplots(2, n_meas)

# plot ICC
n_item0 = 2 * n_roi
width0 = auto_bar_width(x, n_item0)
for meas_idx, meas_name in enumerate(meas2title.keys()):
    ax = axes[0, meas_idx]
    offset = -(n_item0 - 1) / 2
    for roi in rois:
        cols = [f'{roi}_{meas_name}_{hemi}' for hemi in hemis]
        for zyg in zygosity:
            lbl = roi2label[roi] + '_' + zyg
            y = np.array(df.loc[f'r_{zyg}', cols])
            low_err = y - np.array(df.loc[f'r_{zyg}_lb', cols])
            high_err = np.array(df.loc[f'r_{zyg}_ub', cols]) - y
            yerr = np.array([low_err, high_err])
            if zyg == 'mz':
                ax.bar(x+width0*offset, y, width0, yerr=yerr, label=lbl,
                       ec=roi2color[roi], fc='w', hatch='//')
                ax.plot(x+width0*offset, np.array(df.loc['ICC_MZ', cols]),
                        linestyle='', marker='*', ms=10,
                        markeredgecolor='k', markerfacecolor='w')
            else:
                ax.bar(x+width0*offset, y, width0, yerr=yerr, label=lbl,
                       ec=roi2color[roi], fc='w', hatch='\\')
                ax.plot(x+width0*offset, np.array(df.loc['ICC_DZ', cols]),
                        linestyle='', marker='*', ms=10,
                        markeredgecolor='k', markerfacecolor='w')
            offset += 1
    ax.set_title(meas2title[meas_name])
    ax.set_xticks(x)
    ax.set_xticklabels(hemis)
    if meas_idx == 0:
        ax.set_ylabel('ICC')
    if meas_idx == 1:
        ax.legend()

# plot heritability
n_item1 = n_roi
width1 = auto_bar_width(x, n_item1)
for meas_idx, meas_name in enumerate(meas2title.keys()):
    ax = axes[1, meas_idx]
    offset = -(n_item1 - 1) / 2
    for roi in rois:
        cols = [f'{roi}_{meas_name}_{hemi}' for hemi in hemis]
        y = np.array(df.loc['h2', cols])
        low_err = y - np.array(df.loc['h2_lb', cols])
        high_err = np.array(df.loc['h2_ub', cols]) - y
        yerr = np.array([low_err, high_err])
        ax.bar(x+width1*offset, y, width1, yerr=yerr, label=roi2label[roi],
               ec=roi2color[roi], fc='w', hatch='//')
        ax.plot(x+width1*offset, np.array(df.loc['H2', cols]),
                linestyle='', marker='*', ms=10,
                markeredgecolor='k', markerfacecolor='w')
        offset += 1
    ax.set_xticks(x)
    ax.set_xticklabels(hemis)
    if meas_idx == 0:
        ax.set_ylabel('heritability')
    if meas_idx == 1:
        ax.legend()

plt.tight_layout()
plt.show()

# %% calculate correlation within each twin pair.
twins_id_file = pjoin(work_dir, 'twins_id_1080.csv')
meas_name = 'activ'
meas2file = {
    'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                 'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                 'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
    'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
              'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
              'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
    'activ': pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')}
hemis = ('lh', 'rh')
hemi2stru = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'}
mpm_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'
                 'MPM_v3_{hemi}_0.25.nii.gz')
roi2label = {'IOG-face': 1, 'pFus-face': 2, 'mFus-face': 3}
zyg2label = {'MZ': 1, 'DZ': 2}
out_file = pjoin(work_dir, f'twins_pattern-corr_{meas_name}.csv')

df = pd.read_csv(twins_id_file)
meas_reader = CiftiReader(meas2file[meas_name])
meas_ids = [int(name.split('_')[0]) for name in meas_reader.map_names()]
twin1_indices = [meas_ids.index(i) for i in df['twin1']]
twin2_indices = [meas_ids.index(i) for i in df['twin2']]

out_df = pd.DataFrame()
out_df['zygosity'] = df['zygosity']
out_df['zyg'] = [zyg2label[zyg] for zyg in df['zygosity']]
for hemi in hemis:
    meas1 = meas_reader.get_data(hemi2stru[hemi], True)[twin1_indices]
    meas2 = meas_reader.get_data(hemi2stru[hemi], True)[twin2_indices]
    mpm = nib.load(mpm_file.format(hemi=hemi)).get_data().squeeze()
    for roi, lbl in roi2label.items():
        idx_vec = mpm == lbl
        out_df[f"{hemi}_{roi.split('-')[0]}"] = \
        [pearsonr(i[idx_vec], j[idx_vec])[0] for i, j in zip(meas1, meas2)]
out_df.to_csv(out_file, index=False)

# %% plot pattern correlation
rois = ('pFus', 'mFus')
meas_names = ('thickness', 'myelin', 'activ')
meas2ylabel = {'thickness': 'thickness', 'myelin': 'myelin',
               'activ': 'face-avg'}
zygosity = ('MZ', 'DZ')
n_zyg = len(zygosity)
zyg2color = {'MZ': (0.33, 0.33, 0.33, 1),
             'DZ': (0.66, 0.66, 0.66, 1)}
hemis = ('lh', 'rh')
df_file = pjoin(work_dir, 'twins_pattern-corr_{}.csv')

x = np.arange(len(rois))
width = auto_bar_width(x, n_zyg)
fig, axes = plt.subplots(len(hemis), len(meas_names))
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
            ax.bar(x+width*offset, y, width, yerr=yerr, label=zyg,
                   color=zyg2color[zyg])
            offset += 1
        ax.set_ylabel(meas2ylabel[meas_name])
        if meas_idx == 1:
            ax.set_title(hemi)
            if hemi_idx == 0:
                ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels(rois)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# %% plot results from Twin_study_heritability.R
# %%% plot thickness, myelin, and activation
df = pd.read_csv(pjoin(work_dir, 'AE-h2estimate_TMA.csv'))
hemis = ('lh', 'rh')
n_hemi = len(hemis)
rois = ('pFus', 'mFus', 'pFus_mFus')
roi2color = {'pFus': 'limegreen', 'mFus': 'cornflowerblue', 'pFus_mFus': 'black'}
n_roi = len(rois)
meas2title = {'thickness': 'thickness', 'myelin': 'myelin', 'activ': 'face-avg'}
n_meas = len(meas2title)

x = np.arange(n_hemi)
width = auto_bar_width(x, n_roi)
fig, axes = plt.subplots(1, n_meas)
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
        ax.bar(x+width*offset, y, width, yerr=yerr,
               label=roi, ec=roi2color[roi], fc='w', hatch='//')
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

# %%% plot RSFC
df = pd.read_csv(pjoin(work_dir, 'AE-h2estimate_rsfc.csv'))
hemis = ('lh', 'rh')
n_hemi = len(hemis)
rois = ('pFus', 'mFus', 'pFus_mFus')
roi2color = {'pFus': 'limegreen', 'mFus': 'cornflowerblue', 'pFus_mFus': 'black'}
n_roi = len(rois)
trg_config_file = '/nfs/p1/atlases/ColeAnticevicNetPartition/network_labelfile.txt'

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
    cols = [f'{roi}_trg{trg_lbl}_{hemi}' for trg_lbl in trg_labels for hemi in hemis]
    data = np.array(df[cols])
    y = data[1]
    low_err = y - data[0]
    high_err = data[2] - y
    yerr = np.array([low_err, high_err])
    plt.bar(x+width*offset, y, width, yerr=yerr,
            label=roi, ec=roi2color[roi], fc='w', hatch='//')
    offset += 1
plt.xticks(x, hemis*n_trg)
plt.ylabel('heritability')
plt.legend()
plt.tight_layout()
plt.show()
