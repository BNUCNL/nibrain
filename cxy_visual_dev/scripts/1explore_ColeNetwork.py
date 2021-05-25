from os.path import join as pjoin

cole_dir = '/nfs/p1/atlases/ColeAnticevicNetPartition'
proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'data/ColeNetwork')


def separate_networks():
    """
    把ColeNetwork的12个网络分到单独的map里。
    每个map的MMP parcel的label保留原来的样子。
    需要注意的是multimodal_glasser的MMP_mpmLR32k.dlabel.nii中，
    ROI label的编号1~180是右脑，181~360是左脑。0对应的是???
    而cortex_parcel_network_assignments.mat中0~359的index是左脑在先
    """
    import numpy as np
    import nibabel as nib
    from scipy.io import loadmat
    from cxy_visual_dev.lib.ColeNet import get_name_label_of_ColeNetwork
    from magicbox.io.io import CiftiReader, save2cifti

    # inputs
    mmp_file = '/nfs/p1/atlases/multimodal_glasser/surface/'\
               'MMP_mpmLR32k.dlabel.nii'
    roi2net_file = pjoin(cole_dir, 'cortex_parcel_network_assignments.mat')

    # outputs
    out_file = pjoin(work_dir, 'networks.dlabel.nii')

    # load
    mmp_reader = CiftiReader(mmp_file)
    mmp_map = mmp_reader.get_data()[0]
    lbl_tab_raw = mmp_reader.label_tables()[0]

    roi2net = loadmat(roi2net_file)['netassignments'][:, 0]
    roi2net = np.r_[roi2net[180:], roi2net[:180]]
    net_labels = np.unique(roi2net)

    # prepare
    data = np.zeros((len(net_labels), len(mmp_map)), dtype=np.uint16)
    map_names = []
    label_tables = []
    net_lbl2name = {}
    for name, lbl in zip(*get_name_label_of_ColeNetwork()):
        net_lbl2name[lbl] = name

    # calculate
    for net_idx, net_lbl in enumerate(net_labels):
        roi_labels = np.where(roi2net == net_lbl)[0] + 1
        lbl_tab = nib.cifti2.cifti2.Cifti2LabelTable()
        lbl_tab[0] = lbl_tab_raw[0]
        for roi_lbl in roi_labels:
            data[net_idx, mmp_map == roi_lbl] = roi_lbl
            lbl_tab[roi_lbl] = lbl_tab_raw[roi_lbl]
        map_names.append(net_lbl2name[net_lbl])
        label_tables.append(lbl_tab)

    # save
    save2cifti(out_file, data, mmp_reader.brain_models(),
               map_names, label_tables=label_tables)


def extract_net_parcel_info():
    """
    把每个网络的编号，名字以及所包含的MMP parcel的编号和名字列出来
    """
    from magicbox.io.io import CiftiReader

    # inputs
    net_file = pjoin(work_dir, 'networks.dlabel.nii')

    # outputs
    out_file = pjoin(work_dir, 'net_parcel_info.txt')

    # prepare
    reader = CiftiReader(net_file)
    net_names = reader.map_names()
    lbl_tables = reader.label_tables()

    # calculate & save
    wf = open(out_file, 'w')
    for net_idx, net_name in enumerate(net_names):
        wf.write(f'>>>{net_idx+1}-{net_name}\n')
        lbl_tab = lbl_tables[net_idx]
        for lbl_k, lbl_v in lbl_tab.items():
            if lbl_k == 0:
                continue
            wf.write(f'{lbl_k}-{lbl_v.label[:-4]}\n')
        wf.write('<<<\n')
    wf.close()


def check_parcel_num():
    """
    统计每个网络包含的parcel数量，以及左右半脑成对情况
    """
    info_file = pjoin(work_dir, 'net_parcel_info.txt')

    lines = open(info_file).read().splitlines()
    parcels = []
    for line in lines:
        if line.startswith('>>>'):
            print(line.lstrip('>>>'))
        elif line == '<<<':
            n_parcel = len(parcels)
            parcels = [i.split('_')[-1] for i in parcels]
            n_parcel_uniq = len(set(parcels))
            n_paired = n_parcel - n_parcel_uniq
            print(f'#parcel: {n_parcel}\t#LR-paired: {n_paired}\n')
            parcels = []
        else:
            parcels.append(line.split('-')[-1])


if __name__ == '__main__':
    # separate_networks()
    # extract_net_parcel_info()
    check_parcel_num()
