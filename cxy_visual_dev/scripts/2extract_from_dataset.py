import os
import time
import glob
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import LR_count_32k,\
    mmp_map_file, dataset_name2dir, dataset_name2info,\
    All_count_32k
from cxy_visual_dev.lib.algo import calc_alff

proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'data/HCP')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def merge_data(dataset_name, meas_name):
    """
    把所有被试的数据合并到一个cifti文件里

    Args:
        dataset_name (str): HCPD | HCPA
        meas_name (str): thickness | myelin
    """
    # outputs
    out_file = pjoin(work_dir, f'{dataset_name}_{meas_name}.dscalar.nii')

    # prepare
    dataset_dir = dataset_name2dir[dataset_name]
    meas2file = {
        'myelin': pjoin(
            dataset_dir,
            'fmriresults01/{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
            '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
        ),
        'thickness': pjoin(
            dataset_dir,
            'fmriresults01/{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
            '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
        )
    }

    df = pd.read_csv(dataset_name2info[dataset_name])
    n_subj = df.shape[0]

    data = np.zeros((n_subj, LR_count_32k), np.float64)

    # calculate
    for subj_idx, subj_id in enumerate(df['subID']):
        time1 = time.time()
        meas_file = meas2file[meas_name].format(sid=subj_id)
        data[subj_idx] = nib.load(meas_file).get_fdata()[0]
        print(f'Finished: {subj_idx+1}/{n_subj},'
              f'cost: {time.time() - time1} seconds.')

    # save
    mmp_reader = CiftiReader(mmp_map_file)
    save2cifti(out_file, data, mmp_reader.brain_models(), df['subID'])


def smooth_data(dataset_name, meas_name, sigma):
    """
    对原数据进行平滑

    Args:
        dataset_name (str): HCPD | HCPA
        meas_name (str): thickness | myelin
        sigma (float): the size of the gaussian surface smoothing kernel in mm
    """
    # outputs
    out_dir = pjoin(work_dir, f'{dataset_name}_{meas_name}_{sigma}mm')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = pjoin(out_dir, '{sid}.dscalar.nii')
    log_file = pjoin(out_dir, 'cifti_smoothing_log')
    stderr_file = pjoin(out_dir, 'cifti_smoothing_stderr')
    stdout_file = pjoin(out_dir, 'cifti_smoothing_stdout')

    # prepare
    sigma = str(sigma)
    dataset_dir = dataset_name2dir[dataset_name]
    meas2file = {
        'myelin': pjoin(
            dataset_dir,
            'fmriresults01/{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
            '{sid}_V1_MR.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
        ),
        'thickness': pjoin(
            dataset_dir,
            'fmriresults01/{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
            '{sid}_V1_MR.thickness_MSMAll.32k_fs_LR.dscalar.nii'
        )
    }
    meas_file = meas2file[meas_name]
    geo_file = pjoin(
        dataset_dir,
        'fmriresults01/{sid}_V1_MR/T1w/fsaverage_LR32k/'
        '{sid}_V1_MR.{Hemi}.midthickness_MSMAll.32k_fs_LR.surf.gii'
    )

    df = pd.read_csv(dataset_name2info[dataset_name])
    n_subj = df.shape[0]

    # calculate
    log = open(log_file, 'w')
    stderr = open(stderr_file, 'w')
    stdout = open(stdout_file, 'w')
    for idx, subj_id in enumerate(df['subID'], 1):
        time1 = time.time()
        cmd = [
            'wb_command', '-cifti-smoothing', meas_file.format(sid=subj_id),
            sigma, sigma, 'COLUMN', out_file.format(sid=subj_id),
            '-left-surface', geo_file.format(sid=subj_id, Hemi='L'),
            '-right-surface', geo_file.format(sid=subj_id, Hemi='R')
        ]
        log.write('Running: ' + ' '.join(cmd) + '\n')
        subprocess.run(cmd, stderr=stderr, stdout=stdout)
        print(f'Finished: {idx}/{n_subj}, cost: {time.time()-time1} seconds.')
    log.write('done')
    log.close()


def merge_smoothed_data(dataset_name, meas_name, sigma):
    """
    合并我平滑过后的cifti文件

    Args:
        dataset_name (str): HCPD | HCPA
        meas_name (str): thickness | myelin
        sigma (float): the size of the gaussian surface smoothing kernel in mm
    """
    # outputs
    out_file = pjoin(work_dir,
                     f'{dataset_name}_{meas_name}_{sigma}mm.dscalar.nii')

    # prepare
    src_dir = pjoin(work_dir, f'{dataset_name}_{meas_name}_{sigma}mm')
    src_file = pjoin(src_dir, '{sid}.dscalar.nii')
    df = pd.read_csv(dataset_name2info[dataset_name])
    n_subj = df.shape[0]
    data = np.zeros((n_subj, LR_count_32k), np.float64)

    # calculate
    for subj_idx, subj_id in enumerate(df['subID']):
        time1 = time.time()
        meas_file = src_file.format(sid=subj_id)
        data[subj_idx] = nib.load(meas_file).get_fdata()[0]
        print(f'Finished: {subj_idx+1}/{n_subj},'
              f'cost: {time.time() - time1} seconds.')

    # save
    mmp_reader = CiftiReader(mmp_map_file)
    save2cifti(out_file, data, mmp_reader.brain_models(), df['subID'])


def alff(subj_par, subj_ids, stem_path, base_path, tr,
         low_freq_band=(0.01, 0.08), linear_detrend=True):

    # prepare
    n_subj = len(subj_ids)
    alff_all_file = pjoin(subj_par, 'alff.dscalar.nii')
    falff_all_file = pjoin(subj_par, 'falff.dscalar.nii')

    # start
    first_flag = True
    brain_models = None
    volume = None
    alff_all = np.ones((n_subj, All_count_32k), dtype=np.float64) * np.nan
    falff_all = np.ones((n_subj, All_count_32k), dtype=np.float64) * np.nan
    for subj_idx, subj_id in enumerate(subj_ids):
        time1 = time.time()

        # prepare path
        stem_dir = pjoin(subj_par, subj_id, stem_path)
        fpath_ = pjoin(stem_dir, base_path)
        fpaths = glob.glob(fpath_)
        n_run = len(fpaths)
        if n_run == 0:
            continue
        alff_sub_file = pjoin(stem_dir, 'alff.dscalar.nii')
        falff_sub_file = pjoin(stem_dir, 'falff.dscalar.nii')

        # loop all runs
        alff_sub = np.zeros((n_run, All_count_32k), dtype=np.float64)
        falff_sub = np.zeros((n_run, All_count_32k), dtype=np.float64)
        for run_idx, fpath in enumerate(fpaths):
            # prepare path
            run_dir = os.path.dirname(fpath)
            base_name = os.path.basename(fpath)
            base_name = '.'.join(base_name.split('.')[:-2])
            alff_run_file = pjoin(run_dir, f'{base_name}_alff.dscalar.nii')
            falff_run_file = pjoin(run_dir, f'{base_name}_falff.dscalar.nii')

            # get data
            if first_flag:
                reader = CiftiReader(fpath)
                brain_models = reader.brain_models()
                volume = reader.volume
                data = reader.get_data()
                first_flag = False
            else:
                data = nib.load(fpath).get_fdata()
            assert data.shape[1] == All_count_32k

            # calculate alff and falff
            alff_run, falff_run = calc_alff(data, tr, 0,
                                            low_freq_band, linear_detrend)
            alff_sub[run_idx] = alff_run
            falff_sub[run_idx] = falff_run

            # save run
            save2cifti(alff_run_file, alff_run[None, :], brain_models,
                       volume=volume)
            save2cifti(falff_run_file, falff_run[None, :], brain_models,
                       volume=volume)
            print(f'Finish subj-{subj_idx+1}/{n_subj}_run-{run_idx+1}/{n_run}')
        alff_sub = np.mean(alff_sub, 0, keepdims=True)
        falff_sub = np.mean(falff_sub, 0, keepdims=True)
        alff_all[subj_idx] = alff_sub
        falff_all[subj_idx] = falff_sub

        # save subject
        save2cifti(alff_sub_file, alff_sub, brain_models, volume=volume)
        save2cifti(falff_sub_file, falff_sub, brain_models, volume=volume)
        print(f'Finish subj-{subj_idx+1}/{n_subj}, '
              f'cost {time.time()-time1} seconds')

    # save subjects
    save2cifti(alff_all_file, alff_all, brain_models, subj_ids, volume)
    save2cifti(falff_all_file, falff_all, brain_models, subj_ids, volume)


if __name__ == '__main__':
    # merge_data(dataset_name='HCPD', meas_name='thickness')
    # merge_data(dataset_name='HCPD', meas_name='myelin')
    # merge_data(dataset_name='HCPA', meas_name='thickness')
    # merge_data(dataset_name='HCPA', meas_name='myelin')
    # smooth_data(dataset_name='HCPD', meas_name='thickness', sigma=4)
    # smooth_data(dataset_name='HCPD', meas_name='myelin', sigma=4)
    # merge_smoothed_data(dataset_name='HCPD', meas_name='thickness', sigma=4)
    # merge_smoothed_data(dataset_name='HCPD', meas_name='myelin', sigma=4)

    # subj_par = '/nfs/z1/HCP/HCPD/fmriresults01'
    # subj_ids = sorted([i for i in os.listdir(subj_par) if i.startswith('HCD')])
    # alff(
    #     subj_par=subj_par, subj_ids=subj_ids,
    #     stem_path='MNINonLinear/Results',
    #     base_path='rfMRI_REST?_??/'
    #               'rfMRI_REST?_??_Atlas_MSMAll_hp0_clean.dtseries.nii',
    #     tr=0.8, low_freq_band=(0.008, 0.1), linear_detrend=True
    # )

    # subj_par = '/nfs/e1/HCPD/fmriresults01'
    # subj_ids = sorted([i for i in os.listdir(subj_par) if i.startswith('HCD')])
    # alff(
    #     subj_par=subj_par, subj_ids=subj_ids,
    #     stem_path='MNINonLinear/Results',
    #     base_path='rfMRI_REST?_??/'
    #               'rfMRI_REST?_??_Atlas_MSMAll_hp0_clean.dtseries.nii',
    #     tr=0.8, low_freq_band=(0.008, 0.1), linear_detrend=True
    # )

    subj_par = '/nfs/z1/HCP/HCPA/fmriresults01'
    subj_ids = sorted([i for i in os.listdir(subj_par) if i.startswith('HCA')])
    alff(
        subj_par=subj_par, subj_ids=subj_ids,
        stem_path='MNINonLinear/Results',
        base_path='rfMRI_REST?_??/'
                  'rfMRI_REST?_??_Atlas_MSMAll_hp0_clean.dtseries.nii',
        tr=0.8, low_freq_band=(0.008, 0.1), linear_detrend=True
    )
