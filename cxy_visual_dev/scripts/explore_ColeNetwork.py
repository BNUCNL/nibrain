from os.path import join as pjoin

cole_dir = '/nfs/p1/atlases/ColeAnticevicNetPartition'
proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev'
work_dir = pjoin(proj_dir, 'data/ColeNetwork')


def get_name_label_of_ColeNetwork():
    import numpy as np

    rf = open(pjoin(cole_dir, 'network_labelfile.txt'))
    names = []
    labels = []
    while True:
        name = rf.readline()
        if name == '':
            break
        names.append(name.rstrip('\n'))
        labels.append(int(rf.readline().split(' ')[0]))
    indices_sorted = np.argsort(labels)
    names = [names[i] for i in indices_sorted]
    labels = [labels[i] for i in indices_sorted]

    return names, labels


def separate_networks():
    """
    把ColeNetwork的12个网络分到单独的map里。
    每个map的MMP parcel的label保留原来的样子。
    """
    import numpy as np
    import nibabel as nib
    from scipy.io import loadmat
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


if __name__ == '__main__':
    separate_networks()
