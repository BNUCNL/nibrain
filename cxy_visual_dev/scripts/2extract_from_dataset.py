import os
import time
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from scipy.spatial.distance import cdist
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import Atlas, LR_count_32k,\
    mmp_map_file, dataset_name2dir, dataset_name2info,\
    All_count_32k, mmp_vis2_name2label
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


def alff(subj_par, check_file, stem_path, base_path, tr,
         low_freq_band=(0.01, 0.08), linear_detrend=True):

    # prepare
    df = pd.read_csv(check_file, sep='\t')
    n_subj = df.shape[0]
    all_runs = df.columns.drop('subID')
    if 'rfMRI_REST' in all_runs:
        all_runs = all_runs.drop('rfMRI_REST')
    alff_all_file = pjoin(subj_par, 'alff.dscalar.nii')
    falff_all_file = pjoin(subj_par, 'falff.dscalar.nii')

    # start
    first_flag = True
    brain_models = None
    volume = None
    alff_all = np.ones((n_subj, All_count_32k), dtype=np.float64) * np.nan
    falff_all = np.ones((n_subj, All_count_32k), dtype=np.float64) * np.nan
    for subj_idx, idx in enumerate(df.index):
        time1 = time.time()

        # check valid runs
        runs = []
        for run in all_runs:
            run_status = df.loc[idx, run]
            if isinstance(run_status, str) and run_status.startswith('ok'):
                runs.append(run)
        n_run = len(runs)
        if n_run == 0:
            continue

        # prepare path
        stem_dir = pjoin(subj_par, str(df.loc[idx, 'subID']), stem_path)
        alff_sub_file = pjoin(stem_dir, 'alff.dscalar.nii')
        falff_sub_file = pjoin(stem_dir, 'falff.dscalar.nii')

        # loop all runs
        alff_sub = np.zeros((1, All_count_32k), dtype=np.float64)
        falff_sub = np.zeros((1, All_count_32k), dtype=np.float64)
        for run_idx, run in enumerate(runs, 1):
            # prepare path
            fpath = pjoin(stem_dir, base_path.format(run=run))
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
            alff_sub[0] += alff_run
            falff_sub[0] += falff_run

            # save run
            save2cifti(alff_run_file, alff_run[None, :], brain_models,
                       volume=volume)
            save2cifti(falff_run_file, falff_run[None, :], brain_models,
                       volume=volume)
            print(f'Finish subj-{subj_idx+1}/{n_subj}_run-{run_idx}/{n_run}')
        alff_sub = alff_sub / n_run
        falff_sub = falff_sub / n_run
        alff_all[subj_idx] = alff_sub
        falff_all[subj_idx] = falff_sub

        # save subject
        save2cifti(alff_sub_file, alff_sub, brain_models, volume=volume)
        save2cifti(falff_sub_file, falff_sub, brain_models, volume=volume)
        print(f'Finish subj-{subj_idx+1}/{n_subj}, '
              f'cost {time.time()-time1} seconds')

    # save subjects
    save2cifti(alff_all_file, alff_all, brain_models, df['subID'].astype(str), volume)
    save2cifti(falff_all_file, falff_all, brain_models, df['subID'].astype(str), volume)


def ColeParcel_fc_vtx(subj_par, check_file, stem_path, base_path):

    # prepare
    df_ck = pd.read_csv(check_file, sep='\t')
    n_subj = df_ck.shape[0]
    all_runs = df_ck.columns.drop('subID')
    if 'rfMRI_REST' in all_runs:
        all_runs = all_runs.drop('rfMRI_REST')

    # load atlas
    cap_file = '/nfs/z1/atlas/ColeAnticevicNetPartition/' \
               'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii'
    cap_LabelKey_file = '/nfs/z1/atlas/ColeAnticevicNetPartition/' \
                        'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
    map = nib.load(cap_file).get_fdata()[0]
    df = pd.read_csv(cap_LabelKey_file, sep='\t', usecols=['KEYVALUE', 'LABEL'])
    n_parcel = df.shape[0]

    # start
    first_flag = True
    brain_models = None
    volume = None
    for subj_idx, idx in enumerate(df_ck.index, 1):
        time1 = time.time()

        # check valid runs
        runs = []
        for run in all_runs:
            run_status = df_ck.loc[idx, run]
            if isinstance(run_status, str) and run_status.startswith('ok'):
                runs.append(run)
        n_run = len(runs)
        if n_run == 0:
            continue

        # prepare path
        stem_dir = pjoin(subj_par, str(df_ck.loc[idx, 'subID']), stem_path)
        out_file = pjoin(stem_dir, 'rsfc_ColeParcel2Vertex.dscalar.nii')

        # loop all runs
        fc_sub = np.zeros((n_parcel, All_count_32k), dtype=np.float64)
        for run_idx, run in enumerate(runs, 1):

            fpath = pjoin(stem_dir, base_path.format(run=run))

            # get data
            if first_flag:
                reader = CiftiReader(fpath)
                brain_models = reader.brain_models()
                volume = reader.volume
                data = reader.get_data()
                first_flag = False
            else:
                data = nib.load(fpath).get_fdata()
            data = data.T
            assert data.shape[0] == All_count_32k

            # prepare ROI time series
            data_roi = np.zeros((n_parcel, data.shape[1]), dtype=np.float64)
            for k_idx, k in enumerate(df['KEYVALUE']):
                data_roi[k_idx] = np.mean(data[map == k], 0)

            # calculate RSFC
            fc_run = 1 - cdist(data_roi, data, metric='correlation')
            fc_sub += fc_run
            print(f'Finish subj-{subj_idx}/{n_subj}_run-{run_idx}/{n_run}')

        fc_sub /= n_run
        save2cifti(out_file, fc_sub, brain_models, df['LABEL'], volume)
        print(f'Finish subj-{subj_idx}/{n_subj}, '
              f'cost {time.time()-time1} seconds')


def get_HCPY_alff():
    """
    只选用1096名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试
    """
    info_df = pd.read_csv(dataset_name2info['HCPY'])
    check_df = pd.read_csv(pjoin(
        proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv'
    ), sep='\t')
    reader = CiftiReader('/nfs/m1/hcp/alff.dscalar.nii')
    reader_mmp = CiftiReader(mmp_map_file)
    out_file = pjoin(proj_dir, 'data/HCP/HCPY-alff.dscalar.nii')

    data_1206 = reader.get_data()
    subj_ids_1206 = check_df['subID'].to_list()
    assert subj_ids_1206 == [int(i) for i in reader.map_names()]
    ok_idx_vec = np.all(check_df[
        ['rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR', 'rfMRI_REST2_LR']
    ] == 'ok=(1200, 91282)', 1)

    data = np.ones((info_df.shape[0], LR_count_32k), np.float64) * np.nan
    for idx in info_df.index:
        idx_1206 = subj_ids_1206.index(info_df.loc[idx, 'subID'])
        if ok_idx_vec[idx_1206]:
            data[idx] = data_1206[idx_1206, :LR_count_32k]

    save2cifti(out_file, data, reader_mmp.brain_models(),
               [str(i) for i in info_df['subID']])


def get_HCPY_GBC():
    """
    只选用1096名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试
    GBC计算的是一个MMP-vis2的顶点和HCP-MMP1_visual-cortex2以外的所有parcel的连接的均值
    """
    src_file = '/nfs/m1/hcp/{subj_id}/MNINonLinear/Results/'\
        'rsfc_ColeParcel2Vertex.dscalar.nii'
    info_df = pd.read_csv(dataset_name2info['HCPY'])
    check_df = pd.read_csv(pjoin(
        proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv'), sep='\t')
    cap_LabelKey_file = '/nfs/z1/atlas/ColeAnticevicNetPartition/' \
                        'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
    cap_df = pd.read_csv(cap_LabelKey_file, sep='\t')
    reader_mmp = CiftiReader(mmp_map_file)

    atlas = Atlas('MMP-vis2-LR')
    mask_map = np.zeros(LR_count_32k, bool)
    for _, lbl in atlas.roi2label.items():
        mask_map = np.logical_or(mask_map, atlas.maps[0] == lbl)

    out_file = pjoin(proj_dir, 'data/HCP/HCPY-GBC_MMP-vis2.dscalar.nii')

    n_subj = info_df.shape[0]
    subj_ids_1206 = check_df['subID'].to_list()
    ok_idx_vec = np.all(check_df[
        ['rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR', 'rfMRI_REST2_LR']
    ] == 'ok=(1200, 91282)', 1)
    vis_labels = [cap_df.loc[cap_df['GLASSERLABELNAME'] == f'{i}_ROI', 'LABEL'].item()
                  for i in mmp_vis2_name2label.keys()]

    data = np.ones((n_subj, LR_count_32k), np.float64) * np.nan
    first_flag = True
    map_names = []
    for idx in info_df.index:
        time1 = time.time()
        subj_id = info_df.loc[idx, 'subID']
        idx_1206 = subj_ids_1206.index(subj_id)
        if ok_idx_vec[idx_1206]:
            reader = CiftiReader(src_file.format(subj_id=subj_id))
            if first_flag:
                map_names = reader.map_names()
                vis_indices = [map_names.index(i) for i in vis_labels]
                first_flag = False
            else:
                assert map_names == reader.map_names()
            src_data = reader.get_data()[:, :LR_count_32k]
            src_data = np.delete(src_data, vis_indices, 0)
            print(src_data.shape)
            data[idx, mask_map] = np.mean(src_data, 0)[mask_map]
        print(f'Finished {idx+1}/{n_subj}, cost: {time.time()-time1} seconds.')

    save2cifti(out_file, data, reader_mmp.brain_models(),
               [str(i) for i in info_df['subID']])


if __name__ == '__main__':
    # merge_data(dataset_name='HCPD', meas_name='thickness')
    # merge_data(dataset_name='HCPD', meas_name='myelin')
    # merge_data(dataset_name='HCPA', meas_name='thickness')
    # merge_data(dataset_name='HCPA', meas_name='myelin')
    # smooth_data(dataset_name='HCPD', meas_name='thickness', sigma=4)
    # smooth_data(dataset_name='HCPD', meas_name='myelin', sigma=4)
    # merge_smoothed_data(dataset_name='HCPD', meas_name='thickness', sigma=4)
    # merge_smoothed_data(dataset_name='HCPD', meas_name='myelin', sigma=4)

    # subj_par = '/nfs/e1/HCPA/fmriresults01'
    # alff(
    #     subj_par=subj_par,
    #     check_file=pjoin(work_dir, 'HCPA_rfMRI_file_check.tsv'),
    #     stem_path='MNINonLinear/Results',
    #     base_path='{run}/{run}_Atlas_MSMAll_hp0_clean.dtseries.nii',
    #     tr=0.8, low_freq_band=(0.008, 0.1), linear_detrend=True
    # )

    # subj_par = '/nfs/e1/HCPA/fmriresults01'
    # ColeParcel_fc_vtx(
    #     subj_par=subj_par,
    #     check_file=pjoin(work_dir, 'HCPA_rfMRI_file_check.tsv'),
    #     stem_path='MNINonLinear/Results',
    #     base_path='{run}/{run}_Atlas_MSMAll_hp0_clean.dtseries.nii'
    # )

    # subj_par = '/nfs/m1/hcp'
    # alff(
    #     subj_par=subj_par,
    #     check_file=pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv'),
    #     stem_path='MNINonLinear/Results',
    #     base_path='{run}/{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii',
    #     tr=0.8, low_freq_band=(0.008, 0.1), linear_detrend=True
    # )

    # subj_par = '/nfs/m1/hcp'
    # ColeParcel_fc_vtx(
    #     subj_par=subj_par,
    #     check_file=pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv'),
    #     stem_path='MNINonLinear/Results',
    #     base_path='{run}/{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    # )

    # get_HCPY_alff()
    get_HCPY_GBC()
