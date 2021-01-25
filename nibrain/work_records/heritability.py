def get_twins_id():
    import pandas as pd

    zygosity = ('MZ', 'DZ')
    df_in = pd.read_csv('/nfs/m1/hcp/S1200_behavior_restricted.csv')

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
    df_out.to_csv('twins_id.csv', index=False)


def twins_id_stats():
    import numpy as np
    import pandas as pd

    twins_id_file = 'twins_id_rfMRI.csv'
    subjs_1080_file = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis/s2/subject_id'
    subjs_rfMRI_file = '../rfMRI/rfMRI_REST_id'

    df = pd.read_csv(twins_id_file)
    subjs_twin = set(np.concatenate([df['twin1'], df['twin2']]))
    subjs_1080 = set([int(_) for _ in open(subjs_1080_file).read().splitlines()])
    subjs_rfMRI = set([int(_) for _ in open(subjs_rfMRI_file).read().splitlines()])

    flag_1080 = subjs_twin.issubset(subjs_1080)
    flag_rfMRI = subjs_twin.issubset(subjs_rfMRI)

    zygosity = ('MZ', 'DZ')
    for zyg in zygosity:
        df_zyg = df[df['zygosity'] == zyg]
        print(f'The number of {zyg}:', len(df_zyg))
    print(f"Is subject_ids in {twins_id_file} a subset of 1080 subjects?", flag_1080)
    print(f"Is subject_ids in {twins_id_file} a subset of rfMRI_REST_id?", flag_rfMRI)


def get_twins_id_1080():
    """
    留下属于1080被试群里的那些双生子（一对中，只要有一个不满足就不要）。
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv('twins_id.csv')
    subjs_1080_file = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis/s2/subject_id'
    subjs_1080 = [int(i) for i in open(subjs_1080_file).read().splitlines()]

    for idx in df.index:
        if df['twin1'][idx] not in subjs_1080 or df['twin2'][idx] not in subjs_1080:
            df.drop(index=idx, inplace=True)
    df.to_csv('twins_id_1080.csv', index=False)


def get_twins_id_rfMRI():
    """
    留下同时拥有4个静息run的双生子（一对中，只要有一个不满足就不要）。
    需要注意的是，我当时找同时拥有4个静息run的被试ID的时候是限制在1080名被试内的。
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv('twins_id.csv')
    subjs_rfMRI_file = '../rfMRI/rfMRI_REST_id'
    subjs_rfMRI = [int(i) for i in open(subjs_rfMRI_file).read().splitlines()]

    for idx in df.index:
        if df['twin1'][idx] not in subjs_rfMRI or df['twin2'][idx] not in subjs_rfMRI:
            df.drop(index=idx, inplace=True)
    df.to_csv('twins_id_rfMRI.csv', index=False)


def pre_heritability():
    import pickle as pkl
    import pandas as pd
    from os.path import join as pjoin

    zyg2label = {'MZ': 1, 'DZ': 3}
    proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis/s2/1080_fROI/refined_with_Kevin'
    meas2file = {
        'thickness': pjoin(proj_dir, 'structure/MPM_v3_{hemi}_0.25_thickness.pkl'),
        'myelin': pjoin(proj_dir, 'structure/MPM_v3_{hemi}_0.25_myelin.pkl'),
        'activ': pjoin(proj_dir, 'tfMRI/MPM_v3_{hemi}_0.25_activ.pkl')}
    hemis = ('lh', 'rh')
    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    twins_id_file = 'twins_id_1080.csv'
    subjs_1080_file = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis/s2/subject_id'
    out_file = 'pre-heritability_TMA.csv'

    df = pd.read_csv(twins_id_file)
    subjs_1080 = [int(i) for i in open(subjs_1080_file).read().splitlines()]
    subj_indices1 = [subjs_1080.index(_) for _ in df['twin1']]
    subj_indices2 = [subjs_1080.index(_) for _ in df['twin2']]

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


def plot_TMA():
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import auto_bar_width

    df = pd.read_csv('ACE-h2estimate_TMA.csv')
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


def pre_heritability_rsfc():
    import numpy as np
    import pickle as pkl
    import pandas as pd

    zyg2label = {'MZ': 1, 'DZ': 3}
    twins_id_file = 'twins_id_rfMRI.csv'
    hemis = ('lh', 'rh')
    rois = ('IOG-face', 'pFus-face', 'mFus-face')
    rsfc_file = '../rfMRI/rsfc_mpm2Cole_{hemi}.pkl'
    subjs_1080_file = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis/s2/subject_id'
    out_file = 'pre-heritability_rsfc.csv'

    df = pd.read_csv(twins_id_file)
    subjs_1080 = [int(i) for i in open(subjs_1080_file).read().splitlines()]
    subj_indices1 = [subjs_1080.index(_) for _ in df['twin1']]
    subj_indices2 = [subjs_1080.index(_) for _ in df['twin2']]

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


def plot_rsfc():
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import auto_bar_width

    df = pd.read_csv('ACE-h2estimate_rsfc.csv')
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


if __name__ == '__main__':
    # get_twins_id()
    # get_twins_id_1080()
    # get_twins_id_rfMRI()
    # twins_id_stats()
    # pre_heritability()
    plot_TMA()
    # pre_heritability_rsfc()
    plot_rsfc()
