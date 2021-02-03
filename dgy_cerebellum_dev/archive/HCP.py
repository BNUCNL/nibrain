import numpy as np
import os, re, random, nibabel, shutil
import pandas as pd
# import scipy.stats as stats
# from matplotlib import pyplot as plt

class PATH(object):
    ADULTS = '/nfs/m1/hcp/'
    INFO = os.path.join(os.getcwd(), 'Data', 'HCP.csv')
    WORKING_DIR = './Data/'

def getSubjectIDs():
    filnames = os.listdir(PATH.ADULTS)
    subjects = []
    for filename in filnames:
        if re.match(r'^[0-9]{6}$', filename):
            subjects.append(filename)
    return subjects

def getMNIDir(sub_id):
    return os.path.join(PATH.ADULTS, str(sub_id), 'MNINonLinear')

def get32kDir(sub_id):
    return os.path.join(getMNIDir(sub_id), 'fsaverage_LR32k')

SUBJECTS = getSubjectIDs()
def pickSubjectID():
    return random.choice(SUBJECTS)

def getROI(roi_id):
    GROUPDIR = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1'
    MMP = nibabel.load(os.path.join(GROUPDIR,
        'Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'))
    return np.where(MMP.get_fdata()[0] == roi_id)[0]

# Subject information
INFO = pd.read_csv(PATH.INFO)
INFO_DICT = {key: list(INFO.to_dict()[key]) for key in INFO.to_dict()}
INFO_ARR = INFO.to_numpy()

def getMyelinationMap(sub_id):
    sub_dir = getMNIDir(sub_id)
    t1 = nibabel.load(os.path.join(sub_dir, 'T1w_restore.2.nii.gz'))
    t2 = nibabel.load(os.path.join(sub_dir, 'T2w_restore.2.nii.gz'))
    t1_arr = t1.get_fdata()
    t2_arr = t2.get_fdata()
    return t1_arr / t2_arr

def getCerebellumMap(sub_id, flattened = True):
    myelinMap = getMyelinationMap(sub_id)
    sub_dir = getMNIDir(sub_id)
    ROIs = nibabel.load(os.path.join(sub_dir, 'ROIs', 'Atlas_ROIs.2.nii.gz'))
    ROIs_arr = ROIs.get_fdata()
    LCEREBELLUM = 8
    RCEREBELLUM = 47
    if not flattened:
        myelinMap[np.where((ROIs_arr - LCEREBELLUM) * (ROIs_arr - RCEREBELLUM) != 0)] = 0
        return myelinMap
    else:
        return myelinMap[((ROIs_arr - LCEREBELLUM) * (ROIs_arr - RCEREBELLUM)) == 0]