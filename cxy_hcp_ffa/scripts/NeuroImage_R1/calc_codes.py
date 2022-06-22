import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.spatial.distance import cdist
from magicbox.io.io import CiftiReader, save2cifti
from cxy_hcp_ffa.lib.predefine import proj_dir, L_offset_32k,\
    L_count_32k, R_offset_32k, R_count_32k, LR_count_32k, mmp_map_file,\
    mmp_name2label

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'NI_R1')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_cnr(meas_name='CNR'):
    """
    直接从fixextended目录下的RunName_Atlas_stats.dscalar.nii
    文件中取出CNR, TSNR, BOLDVar/UnstructNoiseVar指标
    但后来发现这些统计指标是针对ICA-FIX之前的时间序列的。
    我们还是得按照审稿人的办法重新算CNR。由此，新建calc_cnr1函数
    另外，这里用的CIFTI文件名也没写MSMAll，所以定是不能用这个了~

    Args:
        meas_name (str, optional): Defaults to 'CNR'.
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
    log_file = pjoin(work_dir, f'CNR_log')
    out_file = pjoin(work_dir, f'{meas_name}.pkl')
    
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


def calc_cnr1(meas_name='CNR'):
    """

    Args:
        meas_name (str, optional): Defaults to 'CNR'.
            CNR: BOLDVar/UnstructNoiseVar
                These two quantities can be determined by regressing out
                the signal spatial ICA component timeseries (from the sICA+FIX
                processing run by the HCP) from the cleaned resting state timeseries
                (to compute the Unstructured Noise Variance) and then taking the
                difference between the Cleaned Timeseries Variance and the Unstructured
                Noise Variance to compute the BOLD Variance.
            TSNR: 直接计算rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii
                中时间序列的tSNR
    Notes:
        我发现一个trick，就是如果先计算ROI的平均时间序列的话会抹掉部分甚至全部的非结构噪声，
        这样再去计算tSNR或是CNR的话，就不好说这块区域的信噪比到底怎么样了。
        所以先基于vertex-wise的时间序列计算tSNR或是CNR，然后计算ROI内的平均才比较合理
        之前HCP提供的CNR这些指标就都是vertex-wise的

    """
    roi_names = ['R_pFus', 'R_mFus', 'L_pFus', 'L_mFus']
    roi_file = pjoin(anal_dir, 'HCP-YA_FFA-indiv.32k_fs_LR.dlabel.nii')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    src_files = '/nfs/m1/hcp/{sid}/MNINonLinear/Results/'\
        '{run}/{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    log_file = pjoin(work_dir, f'CNR1_log')
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


if __name__ == '__main__':
    # calc_cnr(meas_name='CNR')
    # calc_cnr(meas_name='TSNR')
    # calc_cnr(meas_name='BOLD_CNR')
    # calc_cnr1(meas_name='TSNR')
    # make_fus_mask(mask_name='union1')
    # make_fus_mask(mask_name='MMP1')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='myelin')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='thickness')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='curv')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='GBC')
    # calc_fus_pattern_corr(mask_name='union1', meas_name='activ')
    # calc_fus_pattern_corr(mask_name='MMP1', meas_name='curv')
    MT_gradient()
