import numpy as np


def get_name_label_of_ColeNetwork():
    rf = open('/nfs/p1/atlases/ColeAnticevicNetPartition/'
              'network_labelfile.txt')
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


cole_name2label = {}
for name, lbl in zip(*get_name_label_of_ColeNetwork()):
    cole_name2label[name] = lbl


def get_parcel2label_by_ColeName(net_names):
    """
    根据Cole Network的名字提取所有包含的parcel及其label

    Args:
        net_names (str|list): ColeNet names
            If is str, one ColeNet name.
            If is list, a list of ColeNet names.
            12 valid names: Primary Visual, Secondary Visual,
            Somatomotor, Cingulo-Opercular, Dorsal-attention,
            Language, Frontoparietal, Auditory, Default,
            Posterior Multimodal, Ventral Multimodal, Orbito-Affective
    """
    if isinstance(net_names, str):
        net_names = [net_names]
    elif isinstance(net_names, list):
        pass
    else:
        raise TypeError("Please input str or list!")

    info_file = '/nfs/s2/userhome/chenxiayu/workingdir/study/visual_dev/'\
        'data/ColeNetwork/net_parcel_info.txt'
    rf = open(info_file)
    parcel2label = {}
    work_flag = False
    while True:
        line = rf.readline().rstrip('\n')
        if line.startswith('>>>'):
            name = '-'.join(line.split('-')[1:])
            if name in net_names:
                work_flag = True
                net_names.remove(name)
        elif line == '<<<':
            work_flag = False
        elif line == '':
            break
        else:
            if work_flag:
                parcel_lbl, parcel_name = line.split('-')
                parcel2label[parcel_name] = int(parcel_lbl)
    if net_names:
        print('Find all ColeNames except for:', net_names)
    else:
        print('Find all ColeNames!')

    return parcel2label
