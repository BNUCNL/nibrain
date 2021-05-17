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
