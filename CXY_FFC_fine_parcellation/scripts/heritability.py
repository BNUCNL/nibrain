import numpy as np
import pandas as pd
import pickle as pkl
from os.path import join as pjoin
from matplotlib import pyplot as plt
from nibrain.util.plotfig import auto_bar_width
from work_progress.CXY.FFC_fine_parcellation.codes import heritability as h2

proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
work_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/heritability')

# start===acquire twins ID===
h2.get_twins_id(src_file='/nfs/m1/hcp/S1200_behavior_restricted.csv',
                trg_file=pjoin(work_dir, 'twins_id.csv'))
h2.count_twins_id(data=pjoin(work_dir, 'twins_id.csv'))
# ===acquire twins ID===end

# start===filter twins ID===
twins_df = pd.read_csv(pjoin(work_dir, 'twins_id.csv'))
subjs_twin = set(np.concatenate([twins_df['twin1'], twins_df['twin2']]))

# check if it's a subset of 1080 subjects
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

# check if the subject have all 4 rfMRI runs.
subjs_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/rfMRI/rfMRI_REST_id')
subjs_id = set([int(_) for _ in open(subjs_file).read().splitlines()])
flag = subjs_twin.issubset(subjs_id)
if flag:
    print('All twins have all 4 rfMRI runs')
else:
    print("Filter twins which don't have all 4 rfMRI runs.")
    h2.filter_twins_id(data=twins_df, limit_set=subjs_id, 
                       trg_file=pjoin(work_dir, 'twins_id_rfMRI.csv'))
    h2.count_twins_id(pjoin(work_dir, 'twins_id_rfMRI.csv'))
# ===filter twins ID===end

# start===prepare input for Twin_study_heritability.R===
hemis = ('lh', 'rh')
rois = ('IOG-face', 'pFus-face', 'mFus-face')
zyg2label = {'MZ': 1, 'DZ': 3}
subjs_1080_file = pjoin(proj_dir, 'analysis/s2/subject_id')
subjs_1080 = [int(i) for i in open(subjs_1080_file).read().splitlines()]

# start---thickness, myelin, activation---
# prepare parameters
meas2file = {
        'thickness': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'\
                           'structure/MPM_v3_{hemi}_0.25_thickness.pkl'),
        'myelin': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'\
                        'structure/MPM_v3_{hemi}_0.25_myelin.pkl'),
        'activ': pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'\
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
# ---thickness, myelin, activation---end

# start---RSFC---
# prepare parameters
rsfc_file = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin/'\
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
# ---RSFC---end
# ===prepare input for Twin_study_heritability.R===end


# start===plot thickness, myelin, and activation===
df = pd.read_csv(pjoin(work_dir, 'ACE-h2estimate_TMA.csv'))
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
        ax.bar(x+width*offset, data[1], width, yerr=data[[0, 2]],
               label=roi, color=roi2color[roi])
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
# ===plot thickness, myelin, and activation===end


# start===plot RSFC===
df = pd.read_csv(pjoin(work_dir, 'ACE-h2estimate_rsfc.csv'))
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
    plt.bar(x+width*offset, data[1], width, yerr=data[[0, 2]],
            label=roi, color=roi2color[roi])
    offset += 1
plt.xticks(x, hemis*n_trg)
plt.ylabel('heritability')
plt.legend()
plt.tight_layout()
plt.show()
# ===plot RSFC===end
