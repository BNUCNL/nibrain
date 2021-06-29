"""
    denoise suggest
"""

import os, re
import numpy as np
import pandas as pd

def read_matrix(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df








if __name__ == '__main__':
    subject_list = ['sub-']