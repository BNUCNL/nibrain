import os
import time
import gzip
import subprocess
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import zscore
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import pairwise_distances
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import Atlas, LR_count_32k, get_rois,\
    mmp_map_file, dataset_name2dir, All_count_32k, proj_dir, hemi2Hemi,\
    L_offset_32k, L_count_32k, R_count_32k, R_offset_32k, hemi2stru,\
    s1200_1096_myelin, s1200_1096_corrThickness, s1200_1096_thickness,\
    s1200_1096_curv
from cxy_visual_dev.lib.algo import calc_alff

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
    info_file = pjoin(work_dir, f'{dataset_name}_SubjInfo.csv')
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
        ),
        'corrThickness': pjoin(
            dataset_dir,
            'fmriresults01/{sid}_V1_MR/MNINonLinear/fsaverage_LR32k/'
            '{sid}_V1_MR.corrThickness_MSMAll.32k_fs_LR.dscalar.nii'
        )
    }
    df = pd.read_csv(info_file)
    n_subj = df.shape[0]

    # calculate
    bms = None
    data = None
    for subj_idx, subj_id in enumerate(df['subID']):
        time1 = time.time()
        meas_file = meas2file[meas_name].format(sid=subj_id)
        reader = CiftiReader(meas_file)
        assert reader.full_data.shape[0] == 1
        if data is None:
            bms = reader.brain_models()
            data = np.zeros((n_subj, reader.full_data.shape[1]))
        data[subj_idx] = reader.get_data()[0]
        print(f'Finished: {subj_idx+1}/{n_subj},'
              f'cost: {time.time() - time1} seconds.')

    # save
    save2cifti(out_file, data, bms, df['subID'])


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


def fc_strength(subj_par, df_ck, stem_path, base_path, stru='cortex', batch_size=0):
    """
    stru=cortex时，计算cortex各顶点和其它所有cortical vertex的相关
    stru=grayordinate时，计算全grayordinate的点和其它所有grayordinate点的相关
    一个点和其它点的相关经fisherZ和取绝对值后求平均作为该点的FC strength。
    对于每个被试，会用上所有check为ok的run，然后求run间平均放到Results目录下。
    batch_size=0时，直接用所有点构造一个大相关矩阵，最耗内存，可达30G
    但速度最快，为了适应写代码时的机器内存，把数据降成了单精度节约内存。
    batch size=1时，逐点计算FC strength，最省内存，但也最慢。
    batch size>1时，逐batch_size个点计算FC strength，目前设置为6000，最高占用7G内存，
    速度虽然比0时慢，但是同时开4个进程内存也才28G，这样速度就比0快一倍。
    """
    # prepare
    n_subj = df_ck.shape[0]
    all_runs = df_ck.columns.drop('subID')
    if 'rfMRI_REST' in all_runs:
        all_runs = all_runs.drop('rfMRI_REST')

    if stru == 'cortex':
        n_vtx = LR_count_32k
    elif stru == 'grayordinate':
        n_vtx = All_count_32k
    else:
        raise ValueError('not supported stru:', stru)

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
        out_file = pjoin(stem_dir, f'rsfc_strength-{stru}.dscalar.nii')

        # loop all runs
        fc_strength_sub = np.zeros((1, n_vtx), dtype=np.float64)
        for run_idx, run in enumerate(runs, 1):

            fpath = pjoin(stem_dir, base_path.format(run=run))

            # get data
            if first_flag:
                reader = CiftiReader(fpath)
                if stru == 'cortex':
                    brain_models = reader.brain_models()[:2]
                    data = reader.get_data()[:, :n_vtx]
                elif stru == 'grayordinate':
                    brain_models = reader.brain_models()
                    volume = reader.volume
                    data = reader.get_data()
                else:
                    raise ValueError('not supported stru:', stru)
                first_flag = False
            else:
                if stru == 'cortex':
                    data = nib.load(fpath).get_fdata()[:, :n_vtx]
                elif stru == 'grayordinate':
                    data = nib.load(fpath).get_fdata()
                else:
                    raise ValueError('not supported stru:', stru)
            data = data.T
            assert data.shape[0] == n_vtx

            # calculate RSFC
            if batch_size == 0:
                # stru=cortex时，这个办法一个被试需要3.42个小时
                # 内存可占到30G，而且还损失了精度
                # 和batch_size=6000那个方法最大差异会在小数点后第7位体现
                fcs = 1 - pdist(data, 'correlation')
                fcs = np.abs(np.arctanh(fcs), dtype=np.float32)
                triu = np.tri(n_vtx, k=-1, dtype=bool).T
                arr = np.zeros((n_vtx, n_vtx), np.float32)
                arr[triu] = fcs
                del fcs, triu
                arr += arr.T
                arr[np.eye(n_vtx, dtype=bool)] = np.nan
                fc_strength_run = np.nanmean(arr, 0, keepdims=True)
            elif batch_size == 1:
                # stru=cortex时，这个办法一个被试需要6天以上
                fc_strength_run = np.zeros((1, n_vtx), dtype=np.float64)
                for idx in range(n_vtx):
                    X1 = data[[idx]]
                    X2 = np.delete(data, idx, 0)
                    rs = 1 - cdist(X1, X2, 'correlation')[0]
                    fc_strength_run[0, idx] = np.mean(np.abs(np.arctanh(rs)))
            elif batch_size > 1:
                # stru=cortex，batch_size=6000时，一个被试需要6.67个小时
                # 内存最高占用到7G
                fc_strength_run = np.zeros((1, n_vtx), dtype=np.float64)
                batch_indices = list(range(0, n_vtx, batch_size))
                n_batch = len(batch_indices)
                batch_indices.append(n_vtx)
                for i, batch_idx1 in enumerate(batch_indices[:-1]):
                    time2 = time.time()
                    batch_idx2 = batch_indices[i + 1]
                    batch = data[batch_idx1:batch_idx2]
                    n_vtx_tmp = batch.shape[0]
                    rs = 1 - cdist(batch, data, 'correlation')
                    np.testing.assert_almost_equal(
                        rs[range(n_vtx_tmp), range(batch_idx1, batch_idx2)], 1)
                    np.testing.assert_almost_equal(
                        np.sum(rs[range(n_vtx_tmp), range(batch_idx1, batch_idx2)]), n_vtx_tmp)
                    rs[range(n_vtx_tmp), range(batch_idx1, batch_idx2)] = np.nan
                    fc_strength_run[0, batch_idx1:batch_idx2] =\
                        np.nanmean(np.abs(np.arctanh(rs)), 1)
                    print(f'Finish subj-{subj_idx}/{n_subj}_run-{run_idx}'
                          f'/{n_run}_batch-{i+1}/{n_batch}: cost '
                          f'{time.time() - time2} seconds.')
            else:
                raise ValueError('not supported batch size:', batch_size)
            fc_strength_sub += fc_strength_run
            print(f'Finish subj-{subj_idx}/{n_subj}_run-{run_idx}/{n_run}')
        fc_strength_sub /= n_run

        # save out
        save2cifti(out_file, fc_strength_sub, brain_models, volume=volume)
        print(f'Finish subj-{subj_idx}/{n_subj}, '
              f'cost {time.time()-time1} seconds')


def fc_strength_mine(s, e):
    """
    改造自fc_strength，只是为了我自己的项目，只计算视觉皮层的FC strength (cortex内)。
    只选用1096名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试
    其它都一样。
    """
    # prepare
    hemi = 'rh'
    Hemi = hemi2Hemi[hemi]
    hemi2offset_count = {
        'lh': (L_offset_32k, L_count_32k),
        'rh': (R_offset_32k, R_count_32k)}
    vis_name = f'MMP-vis3-{Hemi}'
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    check_file = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    run_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/{1}/'\
        '{1}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    tmp_dir = pjoin(work_dir, 'FC_strength_mine')
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    # loading
    offset, count = hemi2offset_count[hemi]
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    mask = mask[offset:offset+count]
    mask_indices = np.where(mask)[0]
    mask_indices_LR = mask_indices + offset
    n_vtx = len(mask_indices)
    df = pd.read_csv(info_file)
    subj_ids = df.loc[s:e, 'subID'].to_list()
    n_subj = len(subj_ids)
    df_ck = pd.read_csv(check_file, sep='\t')
    subj_ids_1206 = df_ck['subID'].to_list()
    ok_idx_vec = np.all(df_ck[runs] == 'ok=(1200, 91282)', 1)
    n_run = len(runs)

    # start
    first_flag = True
    bm = None
    for subj_idx, subj_id in enumerate(subj_ids, 1):
        time1 = time.time()

        # check valid runs
        subj_idx_1206 = subj_ids_1206.index(subj_id)
        if not ok_idx_vec[subj_idx_1206]:
            continue

        # loop all runs
        fc_strength_sub = np.ones((1, count), dtype=np.float64) * np.nan
        fc_strength_mask = np.zeros((n_run, n_vtx), dtype=np.float64)
        for run_idx, run in enumerate(runs):
            time2 = time.time()
            # get data
            run_file = run_files.format(subj_id, run)
            if first_flag:
                reader = CiftiReader(run_file)
                bm = reader.brain_models([hemi2stru[hemi]])[0]
                bm.index_offset = 0
                data = reader.get_data()
                first_flag = False
            else:
                data = nib.load(run_file).get_fdata()
            data = data[:, :LR_count_32k].T
            X1 = data[mask_indices_LR, :]
            rs = 1 - cdist(X1, data, 'correlation')
            np.testing.assert_almost_equal(
                rs[range(n_vtx), mask_indices_LR], 1)
            np.testing.assert_almost_equal(
                np.sum(rs[range(n_vtx), mask_indices_LR]), n_vtx)
            rs[range(n_vtx), mask_indices_LR] = np.nan
            fc_strength_mask[run_idx] =\
                np.nanmean(np.abs(np.arctanh(rs)), 1)
            print(f'Finish subj-{subj_id}_{subj_idx}/{n_subj}_run-{run_idx+1}/{n_run}, '
                  f'cost {time.time() - time2} seconds.')
        fc_strength_sub[0, mask_indices] = np.mean(fc_strength_mask, 0)

        # save out
        out_file = pjoin(tmp_dir, f'{subj_id}_{hemi}.dscalar.nii')
        save2cifti(out_file, fc_strength_sub, [bm])
        print(f'Finish subj-{subj_id}_{subj_idx}/{n_subj}, '
              f'cost {time.time()-time1} seconds')


def fc_strength_mine_merge():
    """
    把fc_strength_mine产生的结果合并到单个文件里
    """
    hemi = 'rh'
    Hemi = hemi2Hemi[hemi]
    hemi2count = {'lh': L_count_32k, 'rh': R_count_32k}
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    out_file = pjoin(work_dir, f'HCPY-FC-strength_{Hemi}.dscalar.nii')

    df = pd.read_csv(info_file)
    subj_ids = [str(i) for i in df['subID']]
    data = np.ones((len(subj_ids), hemi2count[hemi]), np.float64) * np.nan

    for subj_idx, subj_id in enumerate(subj_ids):
        fpath = pjoin(
            work_dir, f'FC_strength_mine/{subj_id}_{hemi}.dscalar.nii')
        if not os.path.exists(fpath):
            continue
        reader = CiftiReader(fpath)
        data[subj_idx] = reader.get_data()[0]
    save2cifti(out_file, data, reader.brain_models(), subj_ids)


def get_HCPY_morph(src_file, out_file):
    """
    从S1200 GroupAvg发布的1096个被试的morphology数据中
    摘出出我们需要的那些被试的数据。
    """
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    reader = CiftiReader(src_file)

    info_df = pd.read_csv(info_file)
    indices = info_df['1096_idx'].values
    data = reader.get_data()[indices]
    map_names = [str(i) for i in info_df['subID']]

    save2cifti(out_file, data, reader.brain_models(), map_names, reader.volume)


def get_HCPY_rsfc_mat_old(hemi='rh'):
    """
    这个函数只运行了一次，就是做了前两个被试，后面不再动了。
    只选用1096名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试。
    每个被试的4个run都先沿时间轴做zscore然后拼接在一起，来做
    视觉皮层内的顶点与所有cortex顶点的连接矩阵。
    最后得到跨被试平均的连接矩阵。
    """
    # prepare
    Hemi = hemi2Hemi[hemi]
    hemi2offset_count = {
        'lh': (L_offset_32k, L_count_32k),
        'rh': (R_offset_32k, R_count_32k)}
    vis_name = f'MMP-vis3-{Hemi}'
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    check_file = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    run_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/{1}/'\
        '{1}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    out_file = pjoin(work_dir, f'RSFC_{vis_name}2cortex.pkl')

    # loading
    offset, count = hemi2offset_count[hemi]
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name))[0]
    mask = mask[offset:offset+count]
    mask_indices = np.where(mask)[0]
    mask_indices_LR = mask_indices + offset
    n_vtx = len(mask_indices)
    df = pd.read_csv(info_file)
    subj_ids = df['subID'].to_list()[:2]  # 目前只做了前2个被试
    n_subj = len(subj_ids)
    df_ck = pd.read_csv(check_file, sep='\t')
    subj_ids_1206 = df_ck['subID'].to_list()
    ok_idx_vec = np.all(df_ck[runs] == 'ok=(1200, 91282)', 1)
    n_tp = 1200
    n_run = len(runs)

    # start
    rs = np.zeros((n_vtx, LR_count_32k))
    n_subj_valid = 0
    for subj_idx, subj_id in enumerate(subj_ids, 1):
        time1 = time.time()

        # check valid runs
        subj_idx_1206 = subj_ids_1206.index(subj_id)
        if not ok_idx_vec[subj_idx_1206]:
            continue

        # loop all runs
        t_series_sub = np.zeros((LR_count_32k, n_run*n_tp))
        for run_idx, run in enumerate(runs):
            run_file = run_files.format(subj_id, run)
            t_series_run = nib.load(run_file).get_fdata()
            t_series_run = t_series_run[:, :LR_count_32k].T
            t_series_run = zscore(t_series_run, 1)
            s_idx = run_idx * n_tp
            e_idx = (run_idx + 1) * n_tp
            t_series_sub[:, s_idx:e_idx] = t_series_run
        X1 = t_series_sub[mask_indices_LR, :]
        rs_sub = 1 - cdist(X1, t_series_sub, 'correlation')
        np.testing.assert_almost_equal(rs_sub[range(n_vtx), mask_indices_LR], 1)
        np.testing.assert_almost_equal(np.sum(rs_sub[range(n_vtx), mask_indices_LR]), n_vtx)

        rs = rs + rs_sub
        n_subj_valid += 1
        print(f'Finish subj-{subj_id}_{subj_idx}/{n_subj}, '
              f'cost {time.time()-time1} seconds')

    data = {'matrix': rs / n_subj_valid, 'row-idx_to_32k-fs-LR-idx': mask_indices_LR}
    pkl.dump(data, open(out_file, 'wb'))


def get_HCPY_alff():
    """
    只选用1096名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试
    """
    info_df = pd.read_csv(pjoin(work_dir, 'HCPY_SubjInfo.csv'))
    check_df = pd.read_csv(pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv'), sep='\t')
    reader = CiftiReader('/nfs/m1/hcp/falff.dscalar.nii')
    reader_mmp = CiftiReader(mmp_map_file)
    out_file = pjoin(work_dir, 'HCPY-falff.dscalar.nii')

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
    GBC计算的是一个视觉皮层内的顶点和外面的所有parcel的连接的均值
    """
    vis_name = 'MMP-vis3'
    src_file = '/nfs/m1/hcp/{subj_id}/MNINonLinear/Results/'\
        'rsfc_ColeParcel2Vertex.dscalar.nii'
    info_file = pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv')
    check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    cap_LabelKey_file = '/nfs/z1/atlas/ColeAnticevicNetPartition/' \
                        'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
    out_file = pjoin(proj_dir, f'data/HCP/HCPY-GBC_{vis_name}.dscalar.nii')

    vis_rois = get_rois(f'{vis_name}-L') + get_rois(f'{vis_name}-R')
    mask_map = Atlas('HCP-MMP').get_mask(vis_rois)[0]

    info_df = pd.read_csv(info_file)
    n_subj = info_df.shape[0]

    check_df = pd.read_csv(check_file, sep='\t')
    subj_ids_1206 = check_df['subID'].to_list()
    ok_idx_vec = np.all(check_df[
        ['rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR', 'rfMRI_REST2_LR']
    ] == 'ok=(1200, 91282)', 1)

    cap_df = pd.read_csv(cap_LabelKey_file, sep='\t')
    vis_labels = [cap_df.loc[cap_df['GLASSERLABELNAME'] == f'{i}_ROI', 'LABEL'].item()
                  for i in vis_rois]

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

    bms = CiftiReader(mmp_map_file).brain_models()
    save2cifti(out_file, data, bms, [str(i) for i in info_df['subID']])


def get_HCPY_GBC1(metric):
    """
    只选用1096名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试
    GBC1计算的是一个顶点和所有parcel的连接的均值
    FC-strength1计算的是一个顶点和所有parcel的连接经FisherZ转换取绝对值后的平均。
    """
    src_file = '/nfs/m1/hcp/{0}/MNINonLinear/Results/'\
        'rsfc_ColeParcel2Vertex.dscalar.nii'
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    check_file = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
    out_file = pjoin(work_dir, f'HCPY-{metric}.dscalar.nii')

    info_df = pd.read_csv(info_file)
    n_subj = info_df.shape[0]

    check_df = pd.read_csv(check_file, sep='\t')
    subj_ids_1206 = check_df['subID'].to_list()
    ok_idx_vec = np.all(check_df[
        ['rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR', 'rfMRI_REST2_LR']
    ] == 'ok=(1200, 91282)', 1)

    data = np.ones((n_subj, All_count_32k), np.float64) * np.nan
    bms = None
    vol = None
    first_flag = True
    for idx in info_df.index:
        time1 = time.time()
        subj_id = info_df.loc[idx, 'subID']
        idx_1206 = subj_ids_1206.index(subj_id)
        if ok_idx_vec[idx_1206]:
            if first_flag:
                reader = CiftiReader(src_file.format(subj_id))
                bms = reader.brain_models()
                vol = reader.volume
                src_data = reader.get_data()
                first_flag = False
            else:
                src_data = nib.load(src_file.format(subj_id)).get_fdata()

            if metric == 'GBC1':
                pass
            elif metric == 'FC-strength1':
                src_data = np.abs(np.arctanh(src_data))
            else:
                raise ValueError('not supported metric:', metric)
            data[idx] = np.mean(src_data, 0)
        print(f'Finished {idx+1}/{n_subj}, cost: {time.time()-time1} seconds.')

    save2cifti(out_file, data, bms, [str(i) for i in info_df['subID']], vol)


def get_HCPY_GBC_cortex_subcortex(metric='GBC', part='cortex'):
    """
    只选用1096名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试
    GBC计算的是一个点和(sub)cortex所有parcel的连接的均值
    FC-strength计算的是一个点和(sub)cortex所有parcel的连接经FisherZ转换取绝对值后的平均。
    """
    src_file = '/nfs/m1/hcp/{0}/MNINonLinear/Results/'\
        'rsfc_ColeParcel2Vertex.dscalar.nii'
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    check_file = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
    out_file = pjoin(work_dir, f'HCPY-{metric}_{part}.dscalar.nii')

    info_df = pd.read_csv(info_file)
    n_subj = info_df.shape[0]

    check_df = pd.read_csv(check_file, sep='\t')
    subj_ids_1206 = check_df['subID'].to_list()
    ok_idx_vec = np.all(check_df[
        ['rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR', 'rfMRI_REST2_LR']
    ] == 'ok=(1200, 91282)', 1)

    data = np.ones((n_subj, All_count_32k), np.float64) * np.nan
    for idx in info_df.index:
        time1 = time.time()
        subj_id = info_df.loc[idx, 'subID']
        idx_1206 = subj_ids_1206.index(subj_id)
        if ok_idx_vec[idx_1206]:
            reader = CiftiReader(src_file.format(subj_id))
            map_idx_vec = [i.endswith('-Ctx') for i in reader.map_names()]
            if part == 'cortex':
                assert np.sum(map_idx_vec) == 360
            elif part == 'subcortex':
                map_idx_vec = ~np.array(map_idx_vec)
                assert np.sum(map_idx_vec) == 358
            else:
                raise ValueError('not supported part:', part)
            src_data = reader.get_data()[map_idx_vec]
            print(src_data.shape)

            if metric == 'GBC':
                pass
            elif metric == 'FC-strength':
                src_data = np.abs(np.arctanh(src_data))
            else:
                raise ValueError('not supported metric:', metric)
            data[idx] = np.mean(src_data, 0)
        print(f'Finished {idx+1}/{n_subj}, cost: {time.time()-time1} seconds.')

    save2cifti(out_file, data, reader.brain_models(),
               [str(i) for i in info_df['subID']], reader.volume)


def get_HCPDA_rsfc_mat(dataset_name='HCPD', Hemi='Right'):
    """
    用HCP官方拼接好的时间序列来计算功能连接，例如：
    /nfs/e1/HCPD/fmriresults01/HCD1363546_V1_MR/MNINonLinear/Results/rfMRI_REST/rfMRI_REST_Atlas_MSMAll_hp0_clean.dtseries.nii
    /nfs/e1/HCPA/fmriresults01/HCA9090779_V1_MR/MNINonLinear/Results/rfMRI_REST/rfMRI_REST_Atlas_MSMAll_hp0_clean.dtseries.nii
    当Hemi='R'时，计算右脑29716个顶点和所有grayordinate顶点的相关矩阵
    并保存在/nfs/m1/HCPD/RSFC_matrix/RightCortex2Grayordinate或/nfs/m1/HCPA/RSFC_matrix/RightCortex2Grayordinate
    当Hemi='L'时，计算右脑29696个顶点和所有grayordinate顶点的相关矩阵
    并保存在/nfs/m1/HCPD/RSFC_matrix/LeftCortex2Grayordinate或/nfs/m1/HCPA/RSFC_matrix/LeftCortex2Grayordinate

    References:
    https://stackoverflow.com/questions/22400652/compress-numpy-arrays-efficiently
    """
    Hemi2offset_count = {
        'Left': (L_offset_32k, L_count_32k),
        'Right': (R_offset_32k, R_count_32k)
    }
    info_file = pjoin(work_dir, f'{dataset_name}_SubjInfo.csv')
    data_file = '/nfs/e1/{dataset}/fmriresults01/{sid}_V1_MR/'\
        'MNINonLinear/Results/rfMRI_REST/rfMRI_REST_Atlas_MSMAll_hp0_clean.dtseries.nii'
    # out_file = '/nfs/m1/{dataset}/RSFC_matrix/{Hemi}Cortex2Grayordinate/{sid}.npy.gz'
    out_file = '/nfs/m1/{dataset}/RSFC_matrix/{Hemi}Cortex2Grayordinate/{sid}.npy'

    offset, count = Hemi2offset_count[Hemi]
    info_df = pd.read_csv(info_file)
    subj_ids = info_df['subID'][2:]
    n_subj = subj_ids.shape[0]
    for subj_idx, subj_id in enumerate(subj_ids, 1):
        time1 = time.time()
        data = nib.load(
            data_file.format(dataset=dataset_name, sid=subj_id)).get_fdata()
        data_hemi = data[:, offset:(offset+count)]
        out_data = 1 - pairwise_distances(data_hemi.T, data.T, 'correlation', n_jobs=-4)
        out_data = out_data.astype(np.float32, copy=False)
        # wf = gzip.GzipFile(out_file.format(
        #     dataset=dataset_name, Hemi=Hemi, sid=subj_id), "w")
        # np.save(file=wf, arr=out_data)
        # wf.close()
        np.save(out_file.format(
            dataset=dataset_name, Hemi=Hemi, sid=subj_id), out_data)
        print(f'Finished {subj_id}-{subj_idx}/{n_subj}: '
              f'cost {time.time() - time1} seconds.')


def get_HCPY_rsfc_mat_bak(Hemi='R'):
    """
    只选用1096名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试。
    每个被试的4个run都先沿时间轴做去均值中心化然后拼接在一起，来做
    视觉皮层内的顶点与所有grayordinate顶点的连接矩阵。
    最后得到跨被试平均的连接矩阵。
    """
    # prepare
    hemi2offset_count = {
        'L': (L_offset_32k, L_count_32k),
        'R': (R_offset_32k, R_count_32k)}
    vis_name = f'MMP-vis3-{Hemi}'
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    check_file = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    run_files = '/nfs/m1/hcp/{0}/MNINonLinear/Results/{1}/'\
        '{1}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    out_file = pjoin(work_dir, f'HCPY-avg_RSFC-{vis_name}2grayordinate.pkl')

    # loading
    offset, count = hemi2offset_count[Hemi]
    mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name), 'grayordinate')[0]
    mask = mask[offset:offset+count]
    mask_indices = np.where(mask)[0]
    mask_indices_LR = mask_indices + offset
    n_vtx = len(mask_indices)
    df = pd.read_csv(info_file)
    subj_ids = df['subID'].to_list()
    n_subj = len(subj_ids)
    df_ck = pd.read_csv(check_file, sep='\t')
    subj_ids_1206 = df_ck['subID'].to_list()
    ok_idx_vec = np.all(df_ck[runs] == 'ok=(1200, 91282)', 1)
    n_tp = 1200
    n_run = len(runs)

    # start
    rs = np.zeros((n_vtx, All_count_32k))
    n_subj_valid = 0
    for subj_idx, subj_id in enumerate(subj_ids, 1):
        time1 = time.time()

        # check valid runs
        subj_idx_1206 = subj_ids_1206.index(subj_id)
        if not ok_idx_vec[subj_idx_1206]:
            continue

        # loop all runs
        t_series_sub = np.zeros((All_count_32k, n_run*n_tp))
        for run_idx, run in enumerate(runs):
            run_file = run_files.format(subj_id, run)
            t_series_run = nib.load(run_file).get_fdata()
            t_series_run = t_series_run - np.mean(t_series_run, 0, keepdims=True)
            s_idx = run_idx * n_tp
            e_idx = (run_idx + 1) * n_tp
            t_series_sub[:, s_idx:e_idx] = t_series_run.T
        X1 = t_series_sub[mask_indices_LR, :]
        rs_sub = 1 - pairwise_distances(X1, t_series_sub, 'correlation', n_jobs=-4)
        np.testing.assert_almost_equal(rs_sub[range(n_vtx), mask_indices_LR], 1)
        np.testing.assert_almost_equal(np.sum(rs_sub[range(n_vtx), mask_indices_LR]), n_vtx)

        rs = rs + rs_sub
        n_subj_valid += 1
        print(f'Finish subj-{subj_id}_{subj_idx}/{n_subj}, '
              f'cost {time.time()-time1} seconds')

    data = {'matrix': rs / n_subj_valid, 'row-idx_to_32k-fs-LR-idx': mask_indices_LR}
    pkl.dump(data, open(out_file, 'wb'))


def get_HCPY_rsfc_mat(Hemi='R'):
    """
    对1070名中的每一位被试，选用状态是ok=(1200, 91282)的run。
    每个run都会计算视觉皮层内两两顶点之间的RSFC, 然后为每个被试
    计算跨ta的所有run的平均RSFC，最终得到跨被试平均的RSFC。
    """
    # prepare
    vis_name = f'MMP-vis3-{Hemi}'
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    check_file = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    run_files = '/nfs/z1/HCP/HCPYA/{0}/MNINonLinear/Results/{1}/'\
        '{1}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    out_file = pjoin(work_dir, f'HCPY-avg_RSFC-{vis_name}.pkl')

    # loading
    vis_mask = Atlas('HCP-MMP').get_mask(get_rois(vis_name), 'grayordinate')[0]
    vtx_indices = np.where(vis_mask)[0]
    n_vtx = len(vtx_indices)
    df = pd.read_csv(info_file, index_col='subID')
    n_sid = df.shape[0]
    df_ck = pd.read_csv(check_file, sep='\t', index_col='subID')

    # start
    rs = np.zeros((n_vtx, n_vtx))
    n_sid_valid = 0
    for sidx, sid in enumerate(df.index, 1):
        time1 = time.time()

        rs_sub = np.zeros((n_vtx, n_vtx))
        n_run_valid = 0
        for run in runs:
            if df_ck.loc[sid, run] != 'ok=(1200, 91282)':
                continue
            run_file = run_files.format(sid, run)
            X = nib.load(run_file).get_fdata()[:, vtx_indices].T
            rs_run = 1 - pairwise_distances(X, metric='correlation', n_jobs=8)
            rs_sub = rs_sub + rs_run
            n_run_valid += 1
        if n_run_valid == 0:
            continue

        rs_sub = rs_sub / n_run_valid
        rs = rs + rs_sub
        n_sid_valid += 1
        print(f'Finish subj-{sid}_{sidx}/{n_sid}, '
              f'cost {time.time()-time1} seconds')
    print('n_sid_valid:', n_sid_valid)

    # save out
    data = {'matrix': rs / n_sid_valid,
            'row-idx_to_32k-fs-LR-idx': vtx_indices}
    pkl.dump(data, open(out_file, 'wb'))


def get_HCPY_rsfc_mat_roi():
    """
    只选用1070名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试。
    每个被试的4个run都先沿时间轴做去均值中心化然后拼接在一起，来做
    HCP-MMP1所有脑区之间的连接矩阵。
    最后得到跨被试平均的连接矩阵。
    """
    # prepare
    atlas = Atlas('HCP-MMP')
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    check_file = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    run_files = '/nfs/z1/HCP/HCPYA/{0}/MNINonLinear/Results/{1}/'\
        '{1}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    out_file = pjoin(work_dir, 'HCPY-avg_RSFC_HCP-MMP.pkl')

    # loading
    mask_map = atlas.maps[0]
    rois = list(atlas.roi2label.keys())
    n_roi = len(rois)
    df = pd.read_csv(info_file)
    subj_ids = df['subID'].to_list()
    n_subj = len(subj_ids)
    df_ck = pd.read_csv(check_file, sep='\t')
    subj_ids_1206 = df_ck['subID'].to_list()
    ok_idx_vec = np.all(df_ck[runs] == 'ok=(1200, 91282)', 1)
    n_tp = 1200
    n_run = len(runs)

    # start
    rs = np.zeros((n_roi, n_roi))
    roi2indices = {}
    for roi in rois:
        mask = mask_map == atlas.roi2label[roi]
        roi2indices[roi] = np.where(mask)[0]
    n_subj_valid = 0
    for subj_idx, subj_id in enumerate(subj_ids, 1):
        time1 = time.time()

        # check valid runs
        subj_idx_1206 = subj_ids_1206.index(subj_id)
        if not ok_idx_vec[subj_idx_1206]:
            continue

        # loop all runs
        t_series_sub = np.zeros((LR_count_32k, n_run*n_tp))
        t_series_roi = np.zeros((n_roi, n_run*n_tp))
        for run_idx, run in enumerate(runs):
            run_file = run_files.format(subj_id, run)
            t_series_run = nib.load(run_file).get_fdata()[:, :LR_count_32k]
            t_series_run = t_series_run - np.mean(t_series_run, 0, keepdims=True)
            s_idx = run_idx * n_tp
            e_idx = (run_idx + 1) * n_tp
            t_series_sub[:, s_idx:e_idx] = t_series_run.T
        for roi_idx, roi in enumerate(rois):
            t_series_roi[roi_idx] = np.mean(t_series_sub[roi2indices[roi]], 0)

        rs_sub = 1 - cdist(t_series_roi, t_series_roi, 'correlation')
        rs = rs + rs_sub
        n_subj_valid += 1
        print(f'Finish subj-{subj_id}_{subj_idx}/{n_subj}, '
              f'cost {time.time()-time1} seconds')

    data = {'matrix': rs / n_subj_valid, 'roi_name': rois}
    pkl.dump(data, open(out_file, 'wb'))


def get_HCPY_rsfc_mat_roi1():
    """
    每个被试（一共1070个）的有效run（状态是ok=(1200, 91282)）都先
    沿时间轴做zscore，然后拼接在一起来做HCP-MMP所有脑区之间的连接矩阵。
    为了得到被试间平均的连接矩阵，先对个体被试的连接矩阵进行Fisher Z变换，
    接着做被试间平均，最终转换回r值。同时再保存一份相关距离矩阵，该矩阵根据
    cosine theorem从r值转换而来(d=sqrt(2*(1-r)))。
    """
    # prepare
    atlas = Atlas('HCP-MMP')
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    check_file = pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv')
    runs = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL',
            'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
    run_files = '/nfs/z1/HCP/HCPYA/{0}/MNINonLinear/Results/{1}/'\
        '{1}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    out_file = pjoin(work_dir, 'HCPY-avg_RSFC_HCP-MMP1.pkl')

    # loading
    mask_map = atlas.maps[0]
    rois = list(atlas.roi2label.keys())
    n_roi = len(rois)
    df = pd.read_csv(info_file, index_col=0)
    n_subj = df.shape[0]
    df_ck = pd.read_csv(check_file, sep='\t', index_col=0)
    n_tp = 1200

    # start
    zs = np.zeros((n_roi, n_roi), dtype=np.float64)
    roi2indices = {}
    for roi in rois:
        mask = mask_map == atlas.roi2label[roi]
        roi2indices[roi] = np.where(mask)[0]
    n_subj_valid = 0
    for subj_idx, subj_id in enumerate(df.index, 1):
        time1 = time.time()

        # find valid runs
        runs_valid = []
        for run in runs:
            if df_ck.loc[subj_id, run] == 'ok=(1200, 91282)':
                runs_valid.append(run)
        n_run = len(runs_valid)
        if n_run == 0:
            continue

        # loop all runs
        t_series_sub = np.zeros((LR_count_32k, n_run*n_tp), dtype=np.float64)
        t_series_roi = np.zeros((n_roi, n_run*n_tp), dtype=np.float64)
        for run_idx, run in enumerate(runs_valid):
            run_file = run_files.format(subj_id, run)
            t_series_run = nib.load(run_file).get_fdata()[:, :LR_count_32k]
            t_series_run = zscore(t_series_run, 0)
            s_idx = run_idx * n_tp
            e_idx = (run_idx + 1) * n_tp
            t_series_sub[:, s_idx:e_idx] = t_series_run.T
        for roi_idx, roi in enumerate(rois):
            t_series_roi[roi_idx] = np.mean(t_series_sub[roi2indices[roi]], 0)

        rs_sub = 1 - cdist(t_series_roi, t_series_roi, 'correlation')
        zs_sub = np.arctanh(rs_sub)
        zs = zs + zs_sub
        n_subj_valid += 1
        print(f'Finish subj-{subj_id}_{subj_idx}/{n_subj}, '
              f'cost {time.time()-time1} seconds')
    print('n_subj_valid:', n_subj_valid)
    zs = zs / n_subj_valid
    rs = np.tanh(zs)
    ds = np.sqrt(2 * (1 - rs))

    data = {'roi_name': rois, 'r-matrix': rs, 'd-matrix': ds}
    pkl.dump(data, open(out_file, 'wb'))


def get_HCPY_Destrieux_atlas():
    """
    基于1070名被试的{sid}.aparc.a2009s.32k_fs_LR.dlabel.nii
    制作Destrieux Atlas各ROI的概率图以及MPM
    """
    info_file = pjoin(work_dir, 'HCPY_SubjInfo.csv')
    src_files = '/nfs/z1/HCP/HCPYA/{sid}/MNINonLinear/fsaverage_LR32k/'\
        '{sid}.aparc.a2009s.32k_fs_LR.dlabel.nii'
    out_file1 = pjoin(work_dir, 'HCPY_aparc-a2009s_prob_32k_fs_LR.dscalar.nii')
    out_file2 = pjoin(work_dir, 'HCPY_aparc-a2009s_MPM_32k_fs_LR.dlabel.nii')

    df = pd.read_csv(info_file, index_col='subID')
    n_sid = df.shape[0]

    lbl_tab = None
    lbl2key = None
    prob_maps = None
    map_names = None
    bms = None
    vol = None
    n_sid_valid = 0
    for sidx, sid in enumerate(df.index, 1):
        src_file = src_files.format(sid=sid)
        if not os.path.exists(src_file):
            continue
        reader = CiftiReader(src_file)
        ind_map = reader.get_data()[0]

        lbl_tab_tmp = reader.label_tables()[0]
        lbl2key_tmp = {}
        for k, v in lbl_tab_tmp.items():
            if k == 0:
                continue
            lbl2key_tmp[v.label] = k
        if lbl2key is None:
            lbl_tab = lbl_tab_tmp
            lbl2key = lbl2key_tmp
            prob_maps = np.zeros((len(lbl2key), ind_map.shape[0]))
            map_names = list(lbl2key.keys())
            bms = reader.brain_models()
            vol = reader.volume
        else:
            assert lbl2key == lbl2key_tmp

        for map_idx, map_name in enumerate(map_names):
            prob_maps[map_idx] = prob_maps[map_idx] +\
                (ind_map == lbl2key[map_name]).astype(int)

        n_sid_valid += 1
        print(f'Finished {sid}-{sidx}/{n_sid}')
    print('n_sid_valid:', n_sid_valid)

    prob_maps = prob_maps / n_sid_valid
    mpm_map = np.argmax(prob_maps, 0)
    mpm_map_tmp = mpm_map + 1
    for vtx_idx in range(mpm_map.shape[0]):
        lbl = map_names[mpm_map[vtx_idx]]
        mpm_map[vtx_idx] = lbl2key[lbl]
    assert np.all(mpm_map == mpm_map_tmp)
    mpm_map = np.expand_dims(mpm_map, 0)

    save2cifti(out_file1, prob_maps, bms, map_names, vol)
    save2cifti(out_file2, mpm_map, bms, None, vol,
               label_tables=[lbl_tab])


if __name__ == '__main__':
    # merge_data(dataset_name='HCPD', meas_name='thickness')
    # merge_data(dataset_name='HCPD', meas_name='myelin')
    # merge_data(dataset_name='HCPD', meas_name='corrThickness')
    # merge_data(dataset_name='HCPA', meas_name='thickness')
    # merge_data(dataset_name='HCPA', meas_name='myelin')
    # merge_data(dataset_name='HCPA', meas_name='corrThickness')

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

    # subj_par = '/nfs/z1/HCP/HCPYA'
    # ColeParcel_fc_vtx(
    #     subj_par=subj_par,
    #     check_file=pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv'),
    #     stem_path='MNINonLinear/Results',
    #     base_path='{run}/{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    # )

    # df_ck = pd.read_csv(pjoin(work_dir, 'HCPY_rfMRI_file_check.tsv'), sep='\t')
    # df_ck = df_ck.loc[[2], :]
    # fc_strength(
    #     subj_par='/nfs/m1/hcp', mask='cortex',
    #     df_ck=df_ck, batch_size=0,
    #     stem_path='MNINonLinear/Results',
    #     base_path='{run}/{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    # )

    # fc_strength_mine(825, 1095)
    # fc_strength_mine_merge()

    # get_HCPY_morph(
    #     src_file=s1200_1096_myelin,
    #     out_file=pjoin(work_dir, 'HCPY_myelin.dscalar.nii')
    # )
    # get_HCPY_morph(
    #     src_file=s1200_1096_corrThickness,
    #     out_file=pjoin(work_dir, 'HCPY_corrThickness.dscalar.nii')
    # )

    # get_HCPY_alff()
    # get_HCPY_GBC()
    # get_HCPY_GBC1('FC-strength1')
    # get_HCPY_GBC_cortex_subcortex(metric='GBC', part='cortex')
    # get_HCPY_GBC_cortex_subcortex(metric='FC-strength', part='cortex')
    # get_HCPY_GBC_cortex_subcortex(metric='GBC', part='subcortex')
    # get_HCPY_GBC_cortex_subcortex(metric='FC-strength', part='subcortex')
    # get_HCPY_face()
    # get_HCPY_rsfc_mat_old(hemi='rh')
    # get_HCPDA_rsfc_mat(dataset_name='HCPD', Hemi='Right')
    # get_HCPDA_rsfc_mat(dataset_name='HCPA', Hemi='Right')
    # get_HCPY_rsfc_mat_bak(Hemi='R')
    # get_HCPY_rsfc_mat(Hemi='R')
    # get_HCPY_rsfc_mat(Hemi='L')
    # get_HCPY_rsfc_mat_roi()
    get_HCPY_rsfc_mat_roi1()
    # get_HCPY_Destrieux_atlas()
