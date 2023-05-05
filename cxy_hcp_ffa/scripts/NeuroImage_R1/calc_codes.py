import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.io import loadmat, savemat
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from magicbox.io.io import CiftiReader, save2cifti
from cxy_hcp_ffa.lib.predefine import proj_dir, L_offset_32k,\
    L_count_32k, R_offset_32k, R_count_32k, LR_count_32k, mmp_map_file,\
    mmp_name2label

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'NI_R1')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_snr1(meas_name):
    """
    直接从fixextended目录下的RunName_Atlas_stats.dscalar.nii
    文件中取出CNR, TSNR, BOLDVar/UnstructNoiseVar指标
    但后来发现这些统计指标是针对ICA-FIX之前的时间序列的。
    我们还是得按照审稿人的办法重新算CNR。由此，新建calc_snr2函数
    另外，这里用的CIFTI文件名也没写MSMAll，所以定是不能用这个了~

    Args:
        meas_name (str):
            CNR: CNR
            TSNR: TSNR
            BOLD_CNR: BOLDVar/UnstructNoiseVar
    """
    roi_names = ['R_pFus', 'R_mFus', 'L_pFus', 'L_mFus']
    roi_file = pjoin(anal_dir, 'HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    meas_files = '/nfs/m1/hcp/{sid}/MNINonLinear/Results/'\
        '{run}/{run}_Atlas_stats.dscalar.nii'
    log_file = pjoin(work_dir, f'{meas_name}1_log')
    out_file = pjoin(work_dir, f'{meas_name}1.pkl')

    reader = CiftiReader(roi_file)
    subj_ids = reader.map_names()
    lbl_tabs = reader.label_tables()
    roi_maps = reader.get_data()

    n_subj, n_vtx = roi_maps.shape
    n_run = len(runs)
    log_writer = open(log_file, 'w')
    out_dict = {'shape': 'n_subj x n_run', 'run_name': runs}
    for roi_name in roi_names:
        out_dict[roi_name] = np.ones((n_subj, n_run)) * np.nan
    for sidx, sid in enumerate(subj_ids):
        time1 = time.time()
        roi2mask = {}
        for roi_key in lbl_tabs[sidx].keys():
            if roi_key == 0:
                continue
            roi_name = lbl_tabs[sidx][roi_key].label.split('-')[0]
            roi2mask[roi_name] = roi_maps[sidx] == roi_key

        for run_idx, run in enumerate(runs):
            meas_file = meas_files.format(sid=sid, run=run)
            try:
                meas_reader = CiftiReader(meas_file)
            except OSError:
                msg = f'{meas_file} meets OSError.'
                print(msg)
                log_writer.write(f'{msg}\n')
                continue
            if meas_name == 'BOLD_CNR':
                cnr_idx1 = meas_reader.map_names().index('BOLDVar')
                cnr_idx2 = meas_reader.map_names().index('UnstructNoiseVar')
                meas_maps = meas_reader.get_data()[:, :n_vtx]
                cnr_map = meas_maps[cnr_idx1] / meas_maps[cnr_idx2]
            else:
                cnr_idx = meas_reader.map_names().index(meas_name)
                cnr_map = meas_reader.get_data()[cnr_idx][:n_vtx]
            for roi_name, mask in roi2mask.items():
                out_dict[roi_name][sidx, run_idx] = np.mean(cnr_map[mask])
        print(f'Finished {sidx + 1}/{n_subj}, cost {time.time() - time1} seconds.')

    log_writer.close()
    pkl.dump(out_dict, open(out_file, 'wb'))


def calc_snr2(meas_name):
    """
    基于RunName_Atlas_MSMAll_hp2000_clean.dtseries.nii计算信噪比

    Args:
        meas_name (str): 信噪比种类
            TSNR: 直接计算tSNR = mean(timeseries) / std(timeseries)
    Notes:
        我发现一个trick，就是如果先计算ROI的平均时间序列的话会抹掉部分甚至全部的随机噪声，
        这样再去计算SNR的话，就不好说这块区域的信噪比到底怎么样了。
        所以先基于vertex-wise的时间序列计算SNR，然后计算ROI内的平均才比较合理
        之前HCP提供的tSNR这些指标就都是vertex-wise的
    """
    roi_names = ['R_pFus', 'R_mFus', 'L_pFus', 'L_mFus']
    roi_file = pjoin(anal_dir, 'HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    src_files = '/nfs/m1/hcp/{sid}/MNINonLinear/Results/'\
        '{run}/{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    log_file = pjoin(work_dir, f'{meas_name}2_log')
    out_file = pjoin(work_dir, f'{meas_name}2.pkl')

    reader = CiftiReader(roi_file)
    subj_ids = reader.map_names()
    lbl_tabs = reader.label_tables()
    roi_maps = reader.get_data()

    n_subj, n_vtx = roi_maps.shape
    n_run = len(runs)
    log_writer = open(log_file, 'w')
    out_dict = {'shape': 'n_subj x n_run', 'run_name': runs}
    for roi_name in roi_names:
        out_dict[roi_name] = np.ones((n_subj, n_run)) * np.nan
    for sidx, sid in enumerate(subj_ids):
        time1 = time.time()
        roi2mask = {}
        for roi_key in lbl_tabs[sidx].keys():
            if roi_key == 0:
                continue
            roi_name = lbl_tabs[sidx][roi_key].label.split('-')[0]
            roi2mask[roi_name] = roi_maps[sidx] == roi_key

        for run_idx, run in enumerate(runs):
            src_file = src_files.format(sid=sid, run=run)
            try:
                src_maps = nib.load(src_file).get_fdata()[:, :n_vtx]
            except OSError:
                msg = f'{src_file} meets OSError.'
                print(msg)
                log_writer.write(f'{msg}\n')
                continue
            if meas_name == 'TSNR':
                cnr_map = np.mean(src_maps, 0) / np.std(src_maps, 0, ddof=1)
            else:
                raise ValueError
            for roi_name, mask in roi2mask.items():
                out_dict[roi_name][sidx, run_idx] = np.mean(cnr_map[mask])
        print(f'Finished {sidx + 1}/{n_subj}, cost {time.time() - time1} seconds.')

    log_writer.close()
    pkl.dump(out_dict, open(out_file, 'wb'))


def make_fus_mask(mask_name='union1'):
    """
    制作一个包含pFus和mFus在内的足够大的mask

    Args:
        mask_name (str, optional): Defaults to 'union1'.
            union1: 用pFus和mFus的概率图的1%阈限以上的部分做并集
            MMP1: 用HCP MMP中的VVC+FFC+V8+PIT
    """
    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1.0, 1.0, 1.0, 0.0)
    lbl_tab[1] = nib.cifti2.Cifti2Label(1, 'R_Fus', 0.0, 1.0, 0.0, 1.0)
    lbl_tab[2] = nib.cifti2.Cifti2Label(2, 'L_Fus', 0.0, 0.0, 1.0, 1.0)
    fus_mask = np.zeros((1, LR_count_32k), np.uint8)
    out_file = pjoin(work_dir, f'Fus-{mask_name}.32k_fs_LR.dlabel.nii')

    L_mask = np.zeros(LR_count_32k, bool)
    L_mask[L_offset_32k:(L_offset_32k + L_count_32k)] = True
    R_mask = np.zeros(LR_count_32k, bool)
    R_mask[R_offset_32k:(R_offset_32k + R_count_32k)] = True
    if mask_name.startswith('union'):
        if mask_name == 'union1':
            thr = 0.01
        else:
            raise ValueError
        prob_file = pjoin(anal_dir, 'HCP-YA_FFA-prob.32k_fs_LR.dscalar.nii')
        prob_maps = nib.load(prob_file).get_fdata() > thr
        union_mask = np.logical_or(prob_maps[0], prob_maps[1])
        union_mask_L = np.logical_and(union_mask, L_mask)
        union_mask_R = np.logical_and(union_mask, R_mask)
        fus_mask[0, union_mask_L] = 2
        fus_mask[0, union_mask_R] = 1
    elif mask_name.startswith('MMP'):
        if mask_name == 'MMP1':
            rois = ['VVC', 'FFC', 'V8', 'PIT']
        else:
            raise ValueError
        mmp_map = nib.load(mmp_map_file).get_fdata()
        for Hemi_idx, Hemi in enumerate(('R', 'L'), 1):
            for roi in rois:
                mmp_mask = mmp_map == mmp_name2label[f'{Hemi}_{roi}']
                fus_mask[mmp_mask] = Hemi_idx
    else:
        raise ValueError

    save2cifti(out_file, fus_mask, CiftiReader(mmp_map_file).brain_models(),
               label_tables=[lbl_tab])


def calc_fus_pattern_corr(mask_name='union1', meas_name='myelin'):
    """
    计算两两被试之间在mask内的空间pattern的相关

    Args:
        mask_name (str, optional): Defaults to 'union1'.
            See "make_fus_mask" for details
        meas_name (str, optional): Defaults to 'myelin'.
    """
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    mask_file = pjoin(work_dir, f'Fus-{mask_name}.32k_fs_LR.dlabel.nii')
    meas2file = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii',
        'va': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
              'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
              'S1200.All.midthickness_MSMAll_va.32k_fs_LR.dscalar.nii',
        'curv': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                'S1200.All.curvature_MSMAll.32k_fs_LR.dscalar.nii',
        'GBC': '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev/'
               'data/HCP/HCPY-GBC_cortex.dscalar.nii',
        'activ': pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    }
    out_file = pjoin(work_dir, f'Fus-pattern-corr_{mask_name}_{meas_name}.pkl')

    # prepare meas
    reader = CiftiReader(meas2file[meas_name])
    subj_ids = open(subj_file).read().splitlines()
    if meas_name in ('thickness', 'myelin', 'va', 'curv', 'GBC'):
        meas_id_file = pjoin(proj_dir, 'data/HCP/subject_id_1096')
        meas_ids = open(meas_id_file).read().splitlines()
        meas_indices = [meas_ids.index(i) for i in subj_ids]
        meas_maps = reader.get_data()[meas_indices]
        if meas_name == 'GBC':
            meas_maps = meas_maps[:, :LR_count_32k]
    elif meas_name == 'activ':
        map_names = [i.split('_')[0] for i in reader.map_names()]
        assert subj_ids == map_names
        meas_maps = reader.get_data()
    else:
        raise ValueError

    # calculate
    mask_reader = CiftiReader(mask_file)
    lbl_tab = mask_reader.label_tables()[0]
    mask_map = mask_reader.get_data()[0]
    out_dict = {}
    for k in lbl_tab.keys():
        if k == 0:
            continue
        mask = mask_map == k
        meas_arr = meas_maps[:, mask]
        out_dict[lbl_tab[k].label] = 1 - cdist(meas_arr, meas_arr, 'correlation')

    pkl.dump(out_dict, open(out_file, 'wb'))


def MT_gradient():
    """
    We defined the gradient simply as the difference between the pFus-faces/FFA-1 
    and mFus-faces/FFA-2 (“pFus – mFus” for myelin; “mFus – pFus” for thickness). 
    For left or right hemisphere, We calculated the gradients for subjects who both 
    have the two ROIs individually to obtain a distribution of individual gradients. 
    We also used these paired individual ROIs to extract the gradients on the group 
    average map to obtain a distribution of group gradients.
    """
    key2name = {0: '???', 1: 'R_pFus-faces', 2: 'R_mFus-faces',
                3: 'L_pFus-faces', 4: 'L_mFus-faces'}
    name2key = {}
    for k, n in key2name.items():
        name2key[n] = k
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    meas_names = ('myelin', 'thickness')
    meas2rois = {
        'myelin': ('pFus-faces', 'mFus-faces'),
        'thickness': ('mFus-faces', 'pFus-faces')}
    roi_file = pjoin(anal_dir, 'HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')
    gid_file=pjoin(anal_dir, 'grouping/group_id_v2_012.csv')
    subj_file = pjoin(proj_dir, 'analysis/s2/subject_id')
    meas_id_file = pjoin(proj_dir, 'data/HCP/subject_id_1096')
    meas2file = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'}
    out_file = pjoin(work_dir, 'MT_gradient.pkl')

    reader = CiftiReader(roi_file)
    subj_ids_tmp = reader.map_names()
    lbl_tabs = reader.label_tables()
    roi_maps = reader.get_data()

    subj_ids = open(subj_file).read().splitlines()
    assert subj_ids == subj_ids_tmp
    n_subj = len(subj_ids)
    meas_ids = open(meas_id_file).read().splitlines()
    meas_indices = [meas_ids.index(i) for i in subj_ids]

    out_dict = {}
    gid_df = pd.read_csv(gid_file)
    for meas_name in meas_names:
        roi_pair = meas2rois[meas_name]
        meas_maps = nib.load(meas2file[meas_name]).get_fdata()[meas_indices]
        for hemi in hemis:
            roi1 = f'{hemi2Hemi[hemi]}_{roi_pair[0]}'
            roi2 = f'{hemi2Hemi[hemi]}_{roi_pair[1]}'
            gid_vec = np.array(gid_df[hemi])
            gid_idx_vec = np.logical_or(gid_vec == 1, gid_vec == 2)
            meas_map_grp = np.mean(meas_maps[gid_idx_vec], 0)
            grad_name_ind = f'{hemi2Hemi[hemi]}_{meas_name}_grad_ind'
            grad_name_grp = f'{hemi2Hemi[hemi]}_{meas_name}_grad_grp'
            out_dict[grad_name_ind] = np.ones(n_subj) * np.nan
            out_dict[grad_name_grp] = np.ones(n_subj) * np.nan
            for subj_idx, subj_id in enumerate(subj_ids):
                if not gid_idx_vec[subj_idx]:
                    continue
                roi1_key = name2key[roi1]
                assert lbl_tabs[subj_idx][roi1_key].label == roi1
                roi2_key = name2key[roi2]
                assert lbl_tabs[subj_idx][roi2_key].label == roi2
                roi1_mask = roi_maps[subj_idx] == roi1_key
                roi2_mask = roi_maps[subj_idx] == roi2_key
                grad_ind = np.mean(meas_maps[subj_idx][roi1_mask]) - \
                    np.mean(meas_maps[subj_idx][roi2_mask])
                grad_grp = np.mean(meas_map_grp[roi1_mask]) - \
                    np.mean(meas_map_grp[roi2_mask])
                out_dict[grad_name_ind][subj_idx] = grad_ind
                out_dict[grad_name_grp][subj_idx] = grad_grp

    pkl.dump(out_dict, open(out_file, 'wb'))


def select_data(subj_mask, out_dir):
    """
    把指定被试的数据节选出来(目前限定于从1080中取出1053个被试)
    也有部分是从1096中取出1053个，都已经额外写好在它们各自的模块里了

    Args:
        subj_mask (bool vector):
        out_dir (str):
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    data_files1 = [
        # pjoin(anal_dir, 'gdist_peak.csv'),
        # pjoin(anal_dir, 'gdist_min1.csv'),
        # pjoin(anal_dir, 'structure/FFA_va.csv'),
        # pjoin(anal_dir, 'structure/FFA_myelin.csv'),
        # pjoin(anal_dir, 'structure/FFA_thickness.csv'),
        # pjoin(anal_dir, 'tfMRI/FFA_activ-emo.csv'),
        # pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole-mean.csv'),
        # pjoin(anal_dir, 'grouping/group_id_v2_012.csv')
    ]
    for data_file1 in data_files1:
        out_file1 = os.path.basename(data_file1)
        print(f'Doing: {out_file1}')
        out_file1 = pjoin(out_dir, out_file1)
        df1 = pd.read_csv(data_file1)
        df1 = df1.loc[subj_mask]
        df1.to_csv(out_file1, index=False)

    data_files2 = [
        # pjoin(anal_dir, 'HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii'),
        # pjoin(anal_dir, 'HCP-YA_FFA-indiv.164k_fsavg_LR.dlabel.nii')
    ]
    for data_file2 in data_files2:
        out_file2 = os.path.basename(data_file2)
        print(f'Doing: {out_file2}')
        out_file2 = pjoin(out_dir, out_file2)
        reader2 = CiftiReader(data_file2)
        data2 = reader2.get_data()[subj_mask]
        map_names2 = reader2.map_names()
        map_names2 = [j for i, j in enumerate(map_names2) if subj_mask[i]]
        lbl_tabs2 = reader2.label_tables()
        lbl_tabs2 = [j for i, j in enumerate(lbl_tabs2) if subj_mask[i]]
        save2cifti(out_file2, data2, reader2.brain_models(), map_names2,
                   reader2.volume, lbl_tabs2)

    data_files3 = [
        # pjoin(anal_dir, 'rfMRI/rsfc_FFA2MMP.mat'),
        # pjoin(anal_dir, 'rfMRI/rsfc_FFA2Cole.mat')
    ]
    for data_file3 in data_files3:
        rois3 = ['lh_pFus', 'lh_mFus', 'rh_pFus', 'rh_mFus']
        out_file3 = os.path.basename(data_file3)
        print(f'Doing: {out_file3}')
        out_file3 = pjoin(out_dir, out_file3)
        data3 = loadmat(data_file3)
        out_data3 = {'target_label': data3['target_label']}
        for roi3 in rois3:
            out_data3[roi3] = data3[roi3][subj_mask]
        savemat(out_file3, out_data3)  # 验证一下

    data_files4 = [
        # pjoin(anal_dir, 'NI_R1/TSNR2.pkl')
    ]
    for data_file4 in data_files4:
        rois4 = ['L_pFus', 'L_mFus', 'R_pFus', 'R_mFus']
        out_file4 = os.path.basename(data_file4)
        print(f'Doing: {out_file4}')
        out_file4 = pjoin(out_dir, out_file4)
        data4 = pkl.load(open(data_file4, 'rb'))
        for roi4 in rois4:
            data4[roi4] = data4[roi4][subj_mask]
        pkl.dump(data4, open(out_file4, 'wb'))

    data_files5 = [
        # pjoin(anal_dir, 'NI_R1/Fus-pattern-corr_MMP1_curv.pkl')
    ]
    for data_file5 in data_files5:
        out_file5 = os.path.basename(data_file5)
        print(f'Doing: {out_file5}')
        out_file5 = pjoin(out_dir, out_file5)
        data5 = pkl.load(open(data_file5, 'rb'))
        for k, v in data5.items():
            data5[k] = v[subj_mask][:, subj_mask]
        pkl.dump(data5, open(out_file5, 'wb'))

    data_files6 = [
        pjoin(proj_dir, 'analysis/s2/activation.dscalar.nii')
    ]
    for data_file6 in data_files6:
        print(f'Doing: {os.path.basename(data_file6)}')
        out_file6 = pjoin(out_dir, 'S1200_1053_tfMRI_WM_level2_FACE-AVG_hp200_s2_MSMAll.32k_fs_LR.dscalar.nii')
        reader6 = CiftiReader(data_file6)
        map_names6 = [j for i, j in enumerate(reader6.map_names()) if subj_mask[i]]
        data6 = reader6.get_data()[subj_mask]
        save2cifti(out_file6, data6, reader6.brain_models(), map_names6, reader6.volume)

    data_files7 = [
        '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/'
        'HCP_S1200_GroupAvg_v1/S1200.All.curvature_MSMAll.32k_fs_LR.dscalar.nii'
    ]
    for data_file7 in data_files7:
        out_file7 = os.path.basename(data_file7)
        print(f'Doing: {out_file7}')
        out_file7 = pjoin(out_dir, out_file7.replace('.All.', '_1053_'))
        subj_file7 = pjoin(proj_dir, 'analysis/s2/subject_id')
        meas_id_file7 = pjoin(proj_dir, 'data/HCP/subject_id_1096')

        reader7 = CiftiReader(data_file7)
        subj_ids7 = open(subj_file7).read().splitlines()
        meas_ids7 = open(meas_id_file7).read().splitlines()
        meas_indices7 = [meas_ids7.index(i) for i in subj_ids7]
        map_names7 = reader7.map_names()
        map_names7 = [map_names7[i] for i in meas_indices7]
        map_names7 = [j for i, j in enumerate(map_names7) if subj_mask[i]]
        data7 = reader7.get_data()[meas_indices7][subj_mask]
        save2cifti(out_file7, data7, reader7.brain_models(), map_names7, reader7.volume)


def snr_regression(src_file, snr_file, out_file):
    """
    将snr的个体变异从数据中回归掉，四个ROI做所有个体上的值连在一起做。
    这是为了在回归掉snr之后也能使得组间，半球间，roi间的比较是有意义的。

    注意：目前这里的回归是连带着截距也减掉了，只要在做拟合之前去掉regressor（此处指tSNR），
    在做拟合时带着截距，最后从因变量（此处为src_file中的数据）中只减去需要被
    回归掉的regressor和其系数的乘积（即保留截距）。这样似乎可以保留因变量原来的scale。
    HCP从thickness中回归curvature时就是这样做的。
    在拟合的时候去掉截距项是无法保留原来的scale的。

    Args:
        src_file (_type_): _description_
        snr_file (_type_): _description_
        out_file (_type_): _description_
    """
    hemis = ('lh', 'rh')
    hemi2Hemi = {'lh': 'L', 'rh': 'R'}
    rois = ('pFus', 'mFus')

    df = pd.read_csv(src_file)
    snr_data = pkl.load(open(snr_file, 'rb'))
    col2mask = {}
    col2indices = {}
    meas = np.zeros(0)
    snr = np.zeros((0, 1))
    start_idx = 0
    for hemi in hemis:
        Hemi = hemi2Hemi[hemi]
        for roi in rois:
            col = f'{hemi}_{roi}'
            meas_vec = np.array(df[col])
            mask = ~np.isnan(meas_vec)
            snr_vec = np.nanmean(snr_data[f'{Hemi}_{roi}'], 1, keepdims=True)

            meas = np.r_[meas, meas_vec[mask]]
            snr = np.r_[snr, snr_vec[mask]]
            col2mask[col] = mask
            end_idx = start_idx + np.sum(mask)
            col2indices[col] = (start_idx, end_idx)
            start_idx = end_idx
    assert np.all(~np.isnan(snr))

    reg = LinearRegression().fit(snr, meas)
    meas_reg = meas - reg.predict(snr)

    for hemi in hemis:
        for roi in rois:
            col = f'{hemi}_{roi}'
            start_idx, end_idx = col2indices[col]
            df.loc[col2mask[col], col] = meas_reg[start_idx:end_idx]

    df.to_csv(out_file, index=False)


def MT_random_maps():
    """
    计算subj_file中所有被试的平均map，并随机挑选
    20个被试的个体map(包括thickness, myelin, FFA)
    """
    n = 20
    meas_names = ('thickness', 'myelin')
    subj_file = pjoin(anal_dir, 'subj_info/subject_id1.txt')
    meas_id_file = pjoin(proj_dir, 'data/HCP/subject_id_1096')
    meas2file = {
        'thickness': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                     'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                     'S1200.All.thickness_MSMAll.32k_fs_LR.dscalar.nii',
        'myelin': '/nfs/p1/public_dataset/datasets/hcp/DATA/'
                  'HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1/'
                  'S1200.All.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'}
    mpm_file = pjoin(work_dir, 'data_1053/HCP-YA_FFA-MPM_thr-25.32k_fs_LR.dlabel.nii')
    roi_file = pjoin(work_dir, 'data_1053/HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')

    # prepare subject information
    subj_ids = open(subj_file).read().splitlines()
    n_subj = len(subj_ids)
    meas_ids = open(meas_id_file).read().splitlines()
    meas_indices = [meas_ids.index(i) for i in subj_ids]

    # random choices
    subj_selected_indices = np.random.choice(range(n_subj), n, replace=False)
    subj_selected_ids = [subj_ids[i] for i in subj_selected_indices]
    map_names = [f'{n_subj}_avg'] + subj_selected_ids

    # select thickness and myelin maps
    for meas_name in meas_names:
        reader = CiftiReader(meas2file[meas_name])
        meas_maps = reader.get_data()[meas_indices]
        avg_map = np.mean(meas_maps, 0, keepdims=True)
        ind_maps = meas_maps[subj_selected_indices]
        out_maps = np.r_[avg_map, ind_maps]
        out_file = pjoin(work_dir, f'avg+ind_{meas_name}_{n_subj}.dscalar.nii')
        save2cifti(out_file, out_maps, reader.brain_models(), map_names, reader.volume)

    # select FFA maps
    out_file = pjoin(work_dir, f'avg+ind_FFA_{n_subj}.dlabel.nii')
    reader = CiftiReader(mpm_file)
    out_maps = reader.get_data()
    lbl_tabs = reader.label_tables()
    reader = CiftiReader(roi_file)
    assert subj_ids == reader.map_names()
    out_maps = np.r_[out_maps, reader.get_data()[subj_selected_indices]]
    lbl_tabs_tmp = reader.label_tables()
    lbl_tabs.extend([lbl_tabs_tmp[i] for i in subj_selected_indices])
    for lbl_tab in lbl_tabs:
        for k, lbl in lbl_tab.items():
            if 'pFus' in lbl.label:
                lbl_tab[k].red = 0.0
                lbl_tab[k].green = 0.0
                lbl_tab[k].blue = 0.0
            elif 'mFus' in lbl.label:
                lbl_tab[k].red = 1.0
                lbl_tab[k].green = 1.0
                lbl_tab[k].blue = 1.0
    save2cifti(out_file, out_maps, reader.brain_models(), map_names, reader.volume, lbl_tabs)


def get_MMP_area(rois, out_file):
    """
    从HCP MMP中选取rois
    """
    Hemis = ('L', 'R')
    reader = CiftiReader(mmp_map_file)
    mmp_map = reader.get_data()
    lbl_tab_mmp = reader.label_tables()[0]

    lbl_tab = nib.cifti2.Cifti2LabelTable()
    lbl_tab[0] = lbl_tab_mmp[0]
    out_map = np.zeros_like(mmp_map, dtype=np.uint16)
    for Hemi in Hemis:
        for roi in rois:
            k = mmp_name2label[f'{Hemi}_{roi}']
            lbl_tab[k] = lbl_tab_mmp[k]
            out_map[mmp_map == k] = k

    save2cifti(out_file, out_map, reader.brain_models(),
               label_tables=[lbl_tab])


if __name__ == '__main__':
    # calc_snr2(meas_name='TSNR')
    # make_fus_mask(mask_name='union1')
    # make_fus_mask(mask_name='MMP1')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='myelin')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='thickness')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='curv')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='GBC')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='activ')
    # calc_fus_pattern_corr(mask_name='MMP1', meas_name='curv')
    # MT_gradient()
    select_data(
        subj_mask=np.load(pjoin(anal_dir, 'subj_info/subject_id1.npy')),
        out_dir=pjoin(work_dir, 'data_1053'))
    # snr_regression(
    #     src_file=pjoin(work_dir, 'data_1053/FFA_activ-emo.csv'),
    #     snr_file=pjoin(work_dir, 'data_1053/TSNR2.pkl'),
    #     out_file=pjoin(work_dir, 'data_1053/FFA_activ-emo_clean-TSNR2.csv'))
    # snr_regression(
    #     src_file=pjoin(work_dir, 'data_1053/rsfc_FFA2Cole-mean.csv'),
    #     snr_file=pjoin(work_dir, 'data_1053/TSNR2.pkl'),
    #     out_file=pjoin(work_dir, 'data_1053/rsfc_FFA2Cole-mean_clean-TSNR2.csv'))

    # MT_random_maps()
    # get_MMP_area(
    #     rois=['VVC', 'FFC', 'V8', 'PIT'],
    #     out_file=pjoin(work_dir, '4areas_around-Fus.32k_fs_LR.dlabel.nii'))
