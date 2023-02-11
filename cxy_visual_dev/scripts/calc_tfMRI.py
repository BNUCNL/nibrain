import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from matplotlib import pyplot as plt
from magicbox.io.io import CiftiReader, save2cifti
from cxy_visual_dev.lib.predefine import All_count_32k, proj_dir

anal_dir = pjoin(proj_dir, 'analysis')
work_dir = pjoin(anal_dir, 'tfMRI')
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)


def get_category_prob_map(thr=2.3):
    """
    只选用1096名中存在“MNINonLinear/Results/tfMRI_WM/
    tfMRI_WM_hp200_s2_level2_MSMAll.feat/
    {sid}_tfMRI_WM_level2_hp200_s2_MSMAll.dscalar.nii”的被试
    对其中的BODY-AVG, FACE-AVG, PLACE-AVG, TOOL-AVG map以z>thr
    为阈限进行二值化，然后跨被试求平均
    """
    categories = ['BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG']
    category_indices = [18, 19, 20, 21]
    src_file = '/nfs/m1/hcp/{sid}/MNINonLinear/Results/'\
        'tfMRI_WM/tfMRI_WM_hp200_s2_level2_MSMAll.feat/'\
        '{sid}_tfMRI_WM_level2_hp200_s2_MSMAll.dscalar.nii'
    info_file = pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv')
    out_file = pjoin(work_dir, f'HCPY-category_prob-map_thr{thr}.dscalar.nii')
    out_log = pjoin(work_dir, 'HCPY-category_prob-map_log')

    info_df = pd.read_csv(info_file)
    n_subj_total = info_df.shape[0]
    n_category = len(categories)

    data = np.zeros((n_category, All_count_32k))
    wf = open(out_log, 'w')
    first_flag = True
    bms = None
    vol = None
    n_subj = 0
    for idx in info_df.index:
        time1 = time.time()
        sid = info_df.loc[idx, 'subID']
        
        try:
            reader = CiftiReader(src_file.format(sid=sid))
        except Exception as err:
            wf.write(f'{err}\n')
            continue
        
        map_names = [i.split('_')[4] for i in reader.map_names()]
        tmp_indices = [map_names.index(i) for i in categories]
        if category_indices != tmp_indices:
            wf.write(f'{sid} has different category indices: {category_indices}')

        if first_flag:
            bms = reader.brain_models()
            vol = reader.volume
            first_flag = False
        cate_maps = reader.get_data()[tmp_indices] > thr
        data = data + cate_maps
        n_subj += 1
        print(f'Finished {sid}-{idx+1}/{n_subj_total}, '
              f'cost: {time.time()-time1} seconds.')
    print('#subject:', n_subj)
    data = data / n_subj

    save2cifti(out_file, data, bms, categories, vol)
    wf.close()


def summary_category_prob_map(fpath, methods, thr=0.2):
    """
    对于每个category probability map，将拥有0.2以上概率值的点
    视为属于该category。然后进行下述总结：
    1. method='MPM', 将每个点分配给所属概率最高的那个category，
        body, face, place, tool分别标记为1, 2, 3, 4
    2. method='count', 跨map求和，得到每个点所属category的数量
    3. method='animate', 将每个点分配给所属概率最高的那个category,
        如果是body或face就标记为1，如果是place或tool就标记为2

    Args:
        thr (float, optional): Defaults to 0.2.
    """
    reader = CiftiReader(fpath)
    map_names = reader.map_names()
    assert ['BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG'] == map_names
    data = reader.get_data()
    data_thr = data > thr
    cmap = plt.cm.jet
    fname = os.path.basename(fpath)
    out_files = pjoin(work_dir, '{0}_summary-{1}-thr{2}.dlabel.nii')

    for method in methods:
        out_file = out_files.format(fname[:-12], method, thr)
        lbl_tab = nib.cifti2.Cifti2LabelTable()
        lbl_tab[0] = nib.cifti2.Cifti2Label(0, '???', 1, 1, 1, 0)
    
        if method == 'MPM':
            out_data = np.argmax(data, 0)
            vec_thr = np.any(data_thr, 0)
            out_data[vec_thr] = out_data[vec_thr] + 1
            out_data = np.expand_dims(out_data, 0)
            n_label = len(map_names)
            color_indices = np.linspace(0, 1, n_label)
            for key, label in enumerate(map_names, 1):
                lbl_tab[key] = nib.cifti2.Cifti2Label(
                    key, label, *cmap(color_indices[key - 1]))

        elif method == 'count':
            out_data = np.sum(data_thr, 0, keepdims=True)
            out_keys = np.unique(out_data).tolist()
            if 0 in out_keys:
                out_keys.remove(0)
            n_key = len(out_keys)
            color_indices = np.linspace(0, 1, n_key)
            for key_idx, key in enumerate(out_keys):
                lbl_tab[key] = nib.cifti2.Cifti2Label(
                    key, str(key), *cmap(color_indices[key_idx]))

        elif method == 'animate':
            out_data = np.argmax(data, 0)
            vec_thr = np.any(data_thr, 0)
            out_data[vec_thr] = out_data[vec_thr] + 1
            out_data[out_data == 2] = 1
            out_data[out_data == 3] = 2
            out_data[out_data == 4] = 2
            out_data = np.expand_dims(out_data, 0)
            color_indices = np.linspace(0, 1, 2)
            lbl_tab[1] = nib.cifti2.Cifti2Label(1, 'animate', *cmap(color_indices[0]))
            lbl_tab[2] = nib.cifti2.Cifti2Label(2, 'inanimate', *cmap(color_indices[1]))

        else:
            raise ValueError('not supported method:', method)
    
        save2cifti(out_file, out_data, reader.brain_models(),
                   volume=reader.volume, label_tables=[lbl_tab])


def get_WM_cope_map():
    """
    提取1070名被试WM任务中'BODY', 'FACE', 'PLACE', 'TOOL',
    'BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG'的平均beta map
    """
    subj_file = pjoin(proj_dir, 'data/HCP/HCPY_SubjInfo.csv')
    copes = ['BODY', 'FACE', 'PLACE', 'TOOL',
             'BODY-AVG', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG']
    feat_dir = '/nfs/z1/HCP/HCPYA/{sid}/MNINonLinear/Results/'\
        'tfMRI_WM/tfMRI_WM_hp200_s2_level2_MSMAll.feat'
    contrast_file = pjoin(feat_dir.format(sid='100307'),
                          'Contrasts.txt')
    cope_files = pjoin(
        feat_dir,
        'GrayordinatesStats/cope{c_num}.feat/cope1.dtseries.nii')
    out_file = pjoin(work_dir, 'tfMRI-WM-cope.dscalar.nii')
    log_file = pjoin(work_dir, 'tfMRI-WM-cope_file-status.csv')

    sids = pd.read_csv(subj_file)['subID'].values
    n_sid = len(sids)
    contrasts = open(contrast_file).read().splitlines()
    c_nums = [contrasts.index(i) + 1 for i in copes]

    bms = None
    vol = None
    out_maps = []
    out_dict = {}
    for c_idx, c_num in enumerate(c_nums):
        c_name = copes[c_idx]
        c_map = 0
        n_sid_valid = 0
        out_dict[c_name] = []
        for sidx, sid in enumerate(sids, 1):
            time1 = time.time()
            cope_file = cope_files.format(sid=sid, c_num=c_num)
            try:
                cope_map = nib.load(cope_file).get_fdata()[0]
            except Exception:
                out_dict[c_name].append('err')
                continue
            out_dict[c_name].append('ok')
            n_sid_valid += 1
            if bms is None:
                reader = CiftiReader(cope_file)
                bms = reader.brain_models()
                vol = reader.volume
            c_map = c_map + cope_map
            print(f'Finished {c_name}-{sidx}/{n_sid}, cost: '
                  f'{time.time()-time1} seconds.')
        c_map = c_map / n_sid_valid
        out_maps.append(c_map)
    out_maps = np.array(out_maps)
    out_df = pd.DataFrame(out_dict)

    # save out
    save2cifti(out_file, out_maps, bms, copes, vol)
    out_df.to_csv(log_file, index=False)


def add_avg_for_WM_cope_map():
    """
    这里的AVG是直接基于BODY, FACE, PLACE, 和TOOL
    四个被试间平均map做平均。由于拥有这四个条件的被试是一致的。
    所以这里直接基于被试间平均map做平均和
    先基于单个被试做平均，然后跨被试平均是一样的。
    """
    copes = ['BODY', 'FACE', 'PLACE', 'TOOL']
    cope_file = pjoin(work_dir, 'tfMRI-WM-cope.dscalar.nii')
    reader = CiftiReader(cope_file)
    bms = reader.brain_models()
    vol = reader.volume
    cope_maps = reader.get_data()
    map_names = reader.map_names()
    cope_indices = [map_names.index(i) for i in copes]
    avg_map = np.mean(cope_maps[cope_indices], 0, keepdims=True)
    cope_maps = np.r_[cope_maps, avg_map]
    map_names.append('AVG')
    save2cifti(cope_file, cope_maps, bms, map_names, vol)


if __name__ == '__main__':
    # get_category_prob_map(thr=2.3)
    # summary_category_prob_map(
    #     fpath=pjoin(work_dir, 'HCPY-category_prob-map_thr2.3.dscalar.nii'),
    #     methods=['MPM', 'count', 'animate'], thr=0.2
    # )
    # get_WM_cope_map()
    add_avg_for_WM_cope_map()
