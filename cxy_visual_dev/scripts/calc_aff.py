import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from scipy.signal import detrend
from scipy.fft import fft, fftfreq
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import proj_dir, All_count_32k

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'AFF')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)

freq_name2band = {
    'slow5': (0.010, 0.027),
    'slow4': (0.027, 0.073),
    'slow3': (0.073, 0.198),
    'slow2': (0.198, 0.538),
    'slow1': (0.538, 0.625),
    'LFF': (0.010, 0.100),
    'LFF-old': (0.008, 0.100)
}
# slow5/slow4的上/下界是np.exp(np.log(0.12) - 1.5) = 0.0268 < 0.027
# slow4/slow3的上/下界是np.exp(np.log(0.12) - 0.5) = 0.0728 < 0.073
# slow3/slow2的上/下界是np.exp(np.log(0.12) + 0.5) = 0.1978 < 0.198
# slow2/slow1的上/下界是np.exp(np.log(0.12) + 1.5) = 0.5378 < 0.538
# 因此设置边界包含情况如下：
# slow5: [0.010, 0.027)
# slow4: [0.027, 0.073)
# slow3: [0.073, 0.198)
# slow2: [0.198, 0.538)
# slow1: [0.538, 0.625]
# LFF: [0.010, 0.100]
# LFF-old: [0.008, 0.100]


def calc_aff(x, tr, freq_names, axis=0, linear_detrend=True):
    """
    Calculate amplitude of frequency fluctuation (AFF) and
    Fractional AFF (fAFF)

    Parameters:
    ----------
        x (array-like): Array to Fourier transform.
        tr (float): Repetition time (second)
        freq_names (strings): frequency band names
        axis (int): Default is the first axis (i.e., axis=0).
            Axis along which the fft is computed.
        linear_detrend (bool): do linear detrend or not

    Returns:
    -------
        aff_list (list of ndarray):
        faff_list (list of ndarray):
    """
    x = np.asarray(x)
    if axis != 0:
        x = np.swapaxes(x, 0, axis)
    if linear_detrend:
        x = detrend(x, axis=0, type='linear')

    # fast fourier transform
    fft_array = fft(x, axis=0)
    # get fourier transform sample frequencies
    freq_scale = fftfreq(x.shape[0], tr)
    # calculate power of half frequency bands
    half_band_idx = (0.0 <= freq_scale) & (freq_scale <= (1 / tr) / 2)
    half_band_array = fft_array[half_band_idx]
    half_band_amps = np.sqrt(np.absolute(half_band_array))
    total_amp = np.sum(half_band_amps, axis=0)

    # calculate AFF or fAFF
    # AFF: sum of band amplitude
    # fAFF: ratio of AFF to total amplitude
    aff_list = []
    faff_list = []
    half_band_scale = freq_scale[half_band_idx]
    for freq_name in freq_names:
        freq_band = freq_name2band[freq_name]
        if freq_name in ('slow1', 'LFF', 'LFF-old'):
            freq_band_idx = (freq_band[0] <= half_band_scale) & \
                            (half_band_scale <= freq_band[1])
        else:
            freq_band_idx = (freq_band[0] <= half_band_scale) & \
                            (half_band_scale < freq_band[1])

        aff = np.sum(half_band_amps[freq_band_idx], axis=0)
        faff = aff / total_amp
        aff_list.append(aff)
        faff_list.append(faff)

    return aff_list, faff_list


def get_HCPY_aff(freq_names, linear_detrend=True):
    """
    计算指定频段的能量幅度(amplitude of frequency fluctuation, AFF)
    只选用1096名中'rfMRI_REST1_RL', 'rfMRI_REST2_RL', 'rfMRI_REST1_LR',
    'rfMRI_REST2_LR'的状态都是ok=(1200, 91282)的被试
    每个被试的AFF是这四个run的AFF的平均
    """
    # prepare
    tr = 0.8
    sid_file = pjoin(proj_dir, 'data/HCP/subject_id_1096')
    check_file = pjoin(proj_dir, 'data/HCP/HCPY_rfMRI_file_check.tsv')
    src_file = '/nfs/z1/HCP/HCPYA/{sid}/MNINonLinear/Results/'\
        '{run}/{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    aff_file = pjoin(work_dir, 'HCPY-aff-{}.dscalar.nii')
    faff_file = pjoin(work_dir, 'HCPY-faff-{}.dscalar.nii')

    subj_ids = open(sid_file).read().splitlines()
    n_subj = len(subj_ids)
    check_df = pd.read_csv(check_file, sep='\t')
    subj_ids_1206 = check_df['subID'].to_list()
    runs = ['rfMRI_REST1_RL', 'rfMRI_REST2_RL',
            'rfMRI_REST1_LR', 'rfMRI_REST2_LR']
    ok_idx_vec = np.all(check_df[runs] == 'ok=(1200, 91282)', 1)
    n_run = len(runs)

    # start
    first_flag = True
    brain_models = None
    volume = None
    freq_name2aff_all = {}
    freq_name2faff_all = {}
    for freq_name in freq_names:
        freq_name2aff_all[freq_name] = \
            np.ones((n_subj, All_count_32k), dtype=np.float64) * np.nan
        freq_name2faff_all[freq_name] = \
            np.ones((n_subj, All_count_32k), dtype=np.float64) * np.nan

    for subj_idx, subj_id in enumerate(subj_ids):
        time1 = time.time()

        idx_1206 = subj_ids_1206.index(int(subj_id))
        if not ok_idx_vec[idx_1206]:
            continue

        # loop all runs
        freq_name2aff_sub = {}
        freq_name2faff_sub = {}
        for freq_name in freq_names:
            freq_name2aff_sub[freq_name] = \
                np.zeros(All_count_32k, dtype=np.float64)
            freq_name2faff_sub[freq_name] = \
                np.zeros(All_count_32k, dtype=np.float64)
        for run_idx, run in enumerate(runs, 1):
            fpath = pjoin(src_file.format(sid=subj_id, run=run))

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
            aff_list, faff_list = calc_aff(data, tr, freq_names, 0, linear_detrend)
            for freq_idx, freq_name in enumerate(freq_names):
                freq_name2aff_sub[freq_name] += aff_list[freq_idx]
                freq_name2faff_sub[freq_name] += faff_list[freq_idx]

            print(f'Finish subj-{subj_idx+1}/{n_subj}_run-{run_idx}/{n_run}')

        # average across runs and assign it to final array
        for freq_name in freq_names:
            freq_name2aff_all[freq_name][subj_idx] = \
                freq_name2aff_sub[freq_name] / n_run
            freq_name2faff_all[freq_name][subj_idx] = \
                freq_name2faff_sub[freq_name] / n_run

        print(f'Finish subj-{subj_idx+1}/{n_subj}, '
              f'cost {time.time()-time1} seconds')

    # save out
    for freq_name in freq_names:
        save2cifti(
            aff_file.format(freq_name), freq_name2aff_all[freq_name],
            brain_models, subj_ids, volume)
        save2cifti(
            faff_file.format(freq_name), freq_name2faff_all[freq_name],
            brain_models, subj_ids, volume)


def merge_mean_map(a_type='aff'):
    """
    计算各频段的平均map (限制在1070名中有4个run的被试内)，放到同一个文件中
    """
    freq_names = ['LFF', 'slow5', 'slow4', 'slow3', 'slow2', 'slow1']
    src_files = pjoin(work_dir, 'HCPY-{0}-{1}.dscalar.nii')
    info_file = pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv')
    out_file = pjoin(work_dir, f'HCPY-{a_type}.dscalar.nii')

    bms = None
    vol = None
    mns = None
    out = None
    indices = None
    info_df = pd.read_csv(info_file)
    for idx, freq_name in enumerate(freq_names):
        src_file = src_files.format(a_type, freq_name)
        reader = CiftiReader(src_file)
        data = reader.get_data()
        if idx == 0:
            bms = reader.brain_models()
            vol = reader.volume
            mns = reader.map_names()
            sids_1096 = [int(i) for i in mns]
            indices = [sids_1096.index(i) for i in info_df['subID']]
            out = np.zeros((len(freq_names), data.shape[1]))
        else:
            assert mns == reader.map_names()
        out[idx] = np.nanmean(data[indices], 0)

    save2cifti(out_file, out, bms, freq_names, vol)


if __name__ == '__main__':
    # get_HCPY_aff(
    #     freq_names=['slow5', 'slow4', 'slow3', 'slow2', 'slow1', 'LFF'],
    #     linear_detrend=True
    # )
    merge_mean_map(a_type='aff')
    merge_mean_map(a_type='faff')
