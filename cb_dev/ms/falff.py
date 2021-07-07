"""
    caculate fALFF of HCPD resting state data
"""

import os
import numpy as np
import nibabel as nib
from scipy.fftpack import fft
from docopt import docopt
import tempfile
import shutil
import subprocess
import sys



def run_cmd(cmd, subject_id):
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        raise Exception('CB_MASK: Error happened in subject {}'.format(subject_id))