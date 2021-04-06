"""
    pipeline
"""

import argparse
import cnls_validation, data2bids, run_fmriprep, melodic_denoise

# initialize argparse
parser = argparse.ArgumentParser()

"""
    required parameters 
"""
parser.add_argument("projectdir", help="path of project dir")
parser.add_argument("taskname", help="name of task")
parser.add_argument("tr", help="repetition time")
parser.add_argument("scaninfo", help="filename of scaninfo, default is <scaninfo.xlsx>", default='scaninfo.xlsx')
parser.add_argument("workdir", help="name of directory stores temp files, should be in <derivatives>, default is <workdir>", default='workdir')



"""
    optinal parameters 
"""
# general parameters
parser.add_argument("--subject", type=str, nargs="+", help="subjects")
parser.add_argument("--preview", action="store_true", help="if choose, user can preview the whole pipeline and inspect critical information without runing any process command")
# CNLS validation
parser.add_argument("--initialize", action="store_true", help="if choose, CNLS validator will initialize a new projectdir.")
parser.add_argument("--create", action="store_true", help="if choose, CNLS validator will create required folder if it is not exist.")
parser.add_argument("--origdir", help="dir contains original data, if not None, CNLS validator will create a soft link to original data dir. It should an be absolute path.")
# data2bids
parser.add_argument("--quality-filter", type=str, help="quality filter on scaninfo.xlsx", choices=['ok', 'all', 'discard'], default='ok')
parser.add_argument("--session", type=str, nargs="+", help="sessions")
parser.add_argument("--skip-feature-validation", action="store_true", help="if choose, pipeline will not compare scan features between scaninfo.xlsx and dicom.tsv")
parser.add_argument("--overwrite", action="store_true", help="if choose, heudiconv will overwrite the existed files")
parser.add_argument("--skip-unpack", action="store_true", help="if choose, pipeline will skip upack")
# fMRIPrep

# MELODIC
parser.add_argument("--run", type=str, nargs="+", help="runs")

args = parser.parse_args()

# CNLS validation
cnls_validation.cnls_validation(args)
# data2bids
data2bids.data2bids(args)
# fMRIPrep
run_fmriprep.run_fmriprep(args)
# melodic denoise
melodic_denoise.melodic_decompose(args)
