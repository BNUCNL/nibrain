import os
import time
import numpy as np
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, save2cifti
from cxy_hcp_ffa.lib.predefine import proj_dir, L_offset_32k,\
    L_count_32k, R_offset_32k, R_count_32k, LR_count_32k, mmp_map_file

anal_dir = pjoin(proj_dir, 'analysis/s2/1080_fROI/refined_with_Kevin')
work_dir = pjoin(anal_dir, 'NI_R1')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def calc_cnr(meas_name='CNR'):

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


def make_fus_mask(mask_name='union1'):
    """
    制作一个包含pFus和mFus在内的足够大的mask

    Args:
        mask_name (str, optional): Defaults to 'union1'.
            union1: 用pFus和mFus的概率图的1%阈限以上的部分做并集
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
    else:
        raise ValueError
    
    save2cifti(out_file, fus_mask, CiftiReader(mmp_map_file).brain_models(),
               label_tables=[lbl_tab])


if __name__ == '__main__':
    # calc_cnr(meas_name='CNR')
    # calc_cnr(meas_name='TSNR')
    # calc_cnr(meas_name='BOLD_CNR')
    make_fus_mask(mask_name='union1')
