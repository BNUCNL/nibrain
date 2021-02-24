import numpy as np, scipy.stats as stats
import os, re, random, nibabel, shutil
import pandas as pd
import matplotlib.cm as cm
from matplotlib import pyplot as plt

#################################
# Diretory settings and methods #
#################################

class PATH(object):
    SUBJECTS = '/nfs/m1/HCPD/fmriresults01/'
    INFO = os.path.join(os.path.dirname(__file__), 'data', 'HCPD.csv')
    WORKING_DIR = os.path.join(os.path.dirname(__file__), 'data/')

def getSubjectIDs():
    filnames = os.listdir(PATH.SUBJECTS)
    subjects = []
    for filename in filnames:
        match = re.match(r'(HCD[0-9]+)_V1_MR$', filename)
        if match: subjects.append(match.groups()[0])
    return subjects

def getMNIDir(sub_id):
    return os.path.join(PATH.SUBJECTS, f'{sub_id}_V1_MR', 'MNINonLinear')

def get32kDir(sub_id):
    return os.path.join(getMNIDir(sub_id), 'fsaverage_LR32k')

SUBJECTS = getSubjectIDs()
def pickSubjectID():
    return random.choice(SUBJECTS)

###################################
# ROI Selection in two approaches #
###################################

def getROI(roi_id):
    GROUPDIR = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1'
    MMP = nibabel.load(os.path.join(GROUPDIR,
        'Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'))
    return np.where(MMP.get_fdata()[0] == roi_id)[0]

class _ROI_INDICES:
    '''ROI indices in .2 resolution.
    '''
    ROI_LABELS = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'ROIs.csv'))
    ROI_TO_INDEX = {key: value for key, value in zip(list(ROI_LABELS['roi']), list(ROI_LABELS['index']))}
    CEREBELLUM = (8, 47)
    BRAIN_STEM = 16

    def __init__(self):
        self.LABELS = set(map(
            lambda s: re.match(r'([A-Z]+(_[A-Z]+)?)_(RIGHT|LEFT)', s).groups()[0],list(self.ROI_TO_INDEX.keys())
        ))
        self.ROI_TO_INDEX_LR = {key: (self.ROI_TO_INDEX[f'{key}_LEFT'], self.ROI_TO_INDEX[f'{key}_RIGHT']) for key in self.LABELS}

ROI_INDICES = _ROI_INDICES()

#######################
# Subject information #
#######################

INFO = pd.read_csv(PATH.INFO)
INFO_DICT = {key: list(INFO.to_dict()[key]) for key in INFO.to_dict()}
INFO_ARR = INFO.to_numpy()
AGES = sorted(list(set(INFO['Age in years'])))

##############################
# Myelination map generating #
##############################

def getMyelinationMap(sub_id):
    sub_dir = getMNIDir(sub_id)
    t1 = nibabel.load(os.path.join(sub_dir, 'T1w_restore.2.nii.gz'))
    t2 = nibabel.load(os.path.join(sub_dir, 'T2w_restore.2.nii.gz'))
    t1_arr = t1.get_fdata()
    t2_arr = t2.get_fdata()
    return t1_arr / t2_arr

def getCerebellumMap(sub_id, invert = False):
    myelinMap = getMyelinationMap(sub_id)
    sub_dir = getMNIDir(sub_id)
    ROIs = nibabel.load(os.path.join(sub_dir, 'ROIs', 'Atlas_ROIs.2.nii.gz'))
    ROIs_arr = ROIs.get_fdata()
    LCEREBELLUM, RCEREBELLUM = ROI_INDICES.CEREBELLUM
    
    indices = np.where(((ROIs_arr - LCEREBELLUM) * (ROIs_arr - RCEREBELLUM)) == 0)
    if invert:
        return myelinMap, indices
    else:
        return myelinMap[indices]

def getCerebrumMap(sub_id):
    """Drwan from original myelination map generated during preprocessing.
    """
    sub_32kdir = get32kDir(sub_id)
    myelinMap = nibabel.load(os.path.join(sub_32kdir, f'{sub_id}_V1_MR.MyelinMap.32k_fs_LR.dscalar.nii'))
    return myelinMap.get_fdata()[0]

def getSubcorticalMap(sub_id, invert = False):
    myelinMap = getMyelinationMap(sub_id)
    sub_dir = getMNIDir(sub_id)
    ROIs = nibabel.load(os.path.join(sub_dir, 'ROIs', 'Atlas_ROIs.2.nii.gz'))
    ROIs_arr = ROIs.get_fdata()
    LCEREBELLUM, RCEREBELLUM = ROI_INDICES.CEREBELLUM
    OTHERS = 0

    indices = np.where(((ROIs_arr - LCEREBELLUM) * (ROIs_arr - RCEREBELLUM) * (ROIs_arr - OTHERS)) != 0)
    if invert:
        return myelinMap, indices
    else:
        return myelinMap[indices]

def getAllMaps(_method, *args):
    """Get maps of specific ROI of all ages with given method.

    Args:
        _method ([function]): Methods to get maps of ROI defined above.
        args: Arguments for selected "_method".

    Returns:
        [dict]: Key for each age, and value for a 2-D numpy array, with axis-0 being subjects.
    """
    MAPS = {}
    for age in AGES:
        print(f'Age: {age}')
        subs = list(INFO[INFO['Age in years'] == age]['Sub'])
        print(f'Subjects: {len(subs)}')
        err = 0
        maps = []
        for sub in subs:
            try:
                maps.append(_method(sub, *(args)))
            except:
                print(f'Error when processing Subject {sub}, ignored.')
                err += 1
                continue
        
        print(f'Unprocessed: {err} in {len(subs)}')
        if not maps:
            print('Since no subject data is available, output is skipped.')
            continue
        MAPS[age] = np.row_stack(maps)

    return MAPS

def getMeanMaps(_method, *args):
    MAPS = getAllMaps(_method, *(args))
    meanMaps = {key: MAPS[key].mean(axis = 0) for key in MAPS}
    return meanMaps

def getNibabelObj(data):
    """With given data, generate a corresponding nibabel object to enable file I/O.
    """
    sub_id = pickSubjectID()
    sub_dir = getMNIDir(sub_id)
    t1 = nibabel.load(os.path.join(sub_dir, 'T1w_restore.2.nii.gz'))
    return nibabel.nifti1.Nifti1Image(data, t1.affine)

def invertMyelinationMap(flatMap, flatIndices = None, _method = getCerebellumMap):
    """Map a flat myelination map to original space according.

    Args:
        flatMap (np.array): 1-D myelination map.
        flatIndices (list, optional): Original indices in the whole map. Should be provided when extreme values are screened out.

    Returns:
        np.ndarray: 3-D Myelination map in original space.
    """
    sub_id = pickSubjectID()
    template, indices = _method(sub_id, True)
    template[:, :, :] = 0
    if not flatIndices: flatIndices = [i for i in range(flatMap.shape[0])]
    for i, idx in enumerate(flatIndices):
        template[indices[0][idx], indices[1][idx], indices[2][idx]] = flatMap[i]

    return template

#################
# Data cleaning #
#################

def getRowOutliersByIQR(x, n = 2):
    """For a 1-D array, find out indices of extreme values beyond n * IQR.
    """
    Q1, Q3 = stats.scoreatpercentile(x, 25), stats.scoreatpercentile(x, 75)
    IQR = Q3 - Q1
    l, u = Q1 - n * IQR, Q3 + n * IQR
    return np.where((x - l) * (x - u) > 0)[0]

def getRowOutliersByThres(x, l, u):
    """For a 1-D array, find out indices of extreme values beyond given thresholds.
    """
    return np.where((x - l) * (x - u) > 0)[0]

def getRowOutliersByNan(x):
    return np.where(np.isnan(x))[0]

def getNormalIndices(array, _method = getRowOutliersByIQR, **kargs):
    """For a 2-D array, find out and concatenate outlier indices in each row.
    """
    outliers = [_method(x, **(kargs)) for x in array]
    combinedOutliers = set()
    for outlier in outliers:
        combinedOutliers |= set(outlier)
    noramls = list(set(np.arange(array.shape[1])) - combinedOutliers)
    print(f'{len(noramls)} / {len(array[0])}')
    return noramls

def setOutliersAsNan(maps: dict, _method = getRowOutliersByIQR, *args):
    """For a dict containing maps of all subjects for each age, set outliers as nan within subject data.

    Args:
        maps (dict): Expected to be {age: np.ndarray([map, map, ..., map])}.

    Returns:
        dict: The same data structure as input.
    """
    new_maps = {}
    for age in maps:
        new_maps[age] = maps[age].copy()
        for i, sub in enumerate(maps[age]):
            indices = _method(sub, *(args))
            new_maps[age][i, indices] = np.nan

    return new_maps

def normalize(x, l, u):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler((l, u))
    return scaler.fit_transform(x)

###################################################
# Generating correlation matrix and visualization #
###################################################

def generateMatrix(_array, rmv_outliers = True):
    """For a dict containing one map for each age, generate the correlation matrix.

    Args:
        _array (dict | np.ndarray | list): Expected to be {age: map} or [map, map, ..., map].
        rmv_outliers (bool, optional): Whether remove outliers by default 2IQR strategy. Defaults to True.

    Returns:
        tuple: (corrMatrx, corrSig)
    """
    array = np.array(_array) if not isinstance(_array, dict) else np.array(list(_array.values()))
    length = len(array)
    
    normals = getNormalIndices(array) if rmv_outliers else np.arange(array.shape[1])
    matrix = np.zeros((length, length), dtype = np.float64)
    sig = np.zeros((length, length), dtype = np.float64)
    for i in range(length):
        for j in range(length):
            if j < i: matrix[i, j], sig[i, j] = matrix[j, i], sig[j, i]
            else: matrix[i, j], sig[i, j] = stats.pearsonr(array[i, normals], array[j, normals])
    return matrix, sig

def showMatrix(maps, vmin = 0):
    if isinstance(maps, dict):
        matrix, _ = generateMatrix(list(maps.values()))
    else:
        matrix = maps
    
    cmap = cm.viridis
    plt.figure()
    ax = plt.axes()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.imshow(matrix - np.eye(len(maps)), cmap = cmap, vmin = vmin)
    plt.colorbar()
    plt.xticks([i for i in range(0, 15, 2)], [i + 8 for i in range(0, 15, 2)])
    plt.yticks([i for i in range(0, 15, 2)], [i + 8 for i in range(0, 15, 2)])
    plt.show()