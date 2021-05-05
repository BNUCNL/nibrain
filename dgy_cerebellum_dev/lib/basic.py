import os
import re
import pandas as pd

#--------------
# Path settings

NI_DATADIR = '/nfs/e1/HCPD/fmriresults01/'
HCP_AVERAGE_DATADIR = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/HCP_S1200_GroupAvg_v1'

BASE = os.path.dirname(os.path.dirname(__file__))
_DATA_DIRNAME = 'data'
_CACHE_DIRNAME = 'cache'
def DATA_DIR():
    return os.path.join(BASE, _DATA_DIRNAME)

def CACHE_DIR():
    return os.path.join(BASE, _CACHE_DIRNAME)

#--------------------
# Subject information

def SUB_INFO():
    return os.path.join(DATA_DIR(), 'HCPD-Full.csv')

def _SUBJECTS_AVAILABLE():
    filnames = os.listdir(NI_DATADIR)
    subjects = []
    for filename in filnames:
        match = re.match(r'(HCD[0-9]+)_V1_MR$', filename)
        if match: subjects.append(match.groups()[0])
    
    return subjects

# Mutiple types of objects are provided for flexible usage
SUB_INFO_DF = pd.read_csv(SUB_INFO())
SUB_INFO_DICT = {key: list(SUB_INFO_DF.to_dict()[key]) for key in SUB_INFO_DF.to_dict()}
SUB_INFO_ARR = SUB_INFO_DF.to_numpy()
SUB_AGES = sorted(list(set(SUB_INFO_DF['Age in years'])))

SUB_AVAILABLE = _SUBJECTS_AVAILABLE()

#-------------------------
# Subject sub-directories

def get_mni_dir(sub_id):
    return os.path.join(NI_DATADIR, f'{sub_id}_V1_MR', 'MNINonLinear')

def get_32k_dir(sub_id):
    return os.path.join(get_mni_dir(sub_id), 'fsaverage_LR32k')

#-------
# Others

DEBUG = False
def rand_pick_sub():
    import random
    return random.choice(SUB_AVAILABLE)