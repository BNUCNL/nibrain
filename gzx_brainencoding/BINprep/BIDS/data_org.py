#! /usr/bin/env python
import os, glob, argparse, string
import subprocess as  sp
import pandas as pd
import numpy as np

from os.path import join as pjoin
from tqdm import tqdm

# ====================================
# class
# ====================================
# print color https://blog.csdn.net/wls666/article/details/100867234
class bcolors:
    WARNING = '\033[33m'
    FAIL = '\033[41m'
    BOLD_NODE = '\033[1;32m'
    BOLD = '\033[1;34m'
    ENDC = '\033[0m'

class heucreation:
    """
    this class will automatically create the heuritics.py
    """
    def __init__(self, file):
        self.file = open(file, 'a+')
        self.HEADER = ['import os\ndef create_key(template, outtype=(\'nii.gz\',), annotation_classes=None):\n    if template is None or not template:\n        raise ValueError(\'Template must be a valid format string\')\n    return template, outtype, annotation_classes\ndef infotodict(seqinfo):\n    """Heuristic evaluator for determining which runs belong where\n    allowed template fields - follow python string module:\n    item: index within category\n    subject: participant id\n    seqitem: run number during scanning\n    subindex: sub index within group\n    """\n\n']
        
    def write_catalog(self,task_list):
        """
        write catalog rules part
        parameters:
        -----------
        task_list : list, value of session_task dict, like ['func_rest', 'fmap/magnitude']
        """
        content = []
        for _ in task_list:
            mod, label = _.split('/')[0], _.split('/')[1]
            if mod in ['anat', 'dwi', 'fmap']:
                content.append("    {0}_{1} = create_key('sub-{{subject}}/{{session}}/{0}/sub-{{subject}}_{{session}}_run-{{item:02d}}_{1}')\n"\
                    .format(mod, label))
            if mod in ['func']:
                content.append("    {0}_{1} = create_key('sub-{{subject}}/{{session}}/{0}/sub-{{subject}}_{{session}}_task-{1}_run-{{item:02d}}_bold')\n"\
                    .format(mod, label))
        self.file.writelines(content)

    def write_info(self, task_list):
        """
        write the info dict part
        parameters:
        -----------
        task_list: list, value of session_task dict, like ['func_rest', 'fmap/magnitude']
        """
        
        content = ["\n    info = {"] + ["{0[0]}_{0[1]}:[],".format(_.split('/')) for _ in task_list[:-1]] \
            + ["{0[0]}_{0[1]}:[]}}\n".format(_.split('/')) for _ in [task_list[-1]]]
        
        self.file.writelines(content)

    def write_condition(self, task_list, feature_dict):
        """
        write the condition part
        parameters:
        ----------
        task_list: list
        feaure_dict: dict
        """
        openning = ["\n    for idx, s in enumerate(seqinfo):\n"]
        ending = ["    return info\n"]
        middle = []
        for _ in task_list:
            mod, label = _.split('/')[0], _.split('/')[1]
            if mod == 'anat':
                middle.append("        if ('{}' in s.protocol_name):\n".format(feature_dict[_][0]))
                middle.append("            info[{0}_{1}].append(s.series_id)\n".format(mod, label))
            if mod == 'fmap':
                middle.append("        if ('{0[0]}' in s.protocol_name) and (s.dim3 == {0[1]}):\n".format(feature_dict[_]))
                middle.append("            info[{0}_{1}].append(s.series_id)\n".format(mod, label))
            if mod == 'func':
                middle.append("        if ('{0[0]}' in s.protocol_name) and (s.dim4 == {0[1]}):\n".format(feature_dict[_]))
                middle.append("            info[{0}_{1}].append(s.series_id)\n".format(mod, label))
        content = openning + middle + ending
        self.file.writelines(content)

    def create_heuristic(self, task_list, feature_dict):
        """
        create the heuristic.py according to task_list & feature_dict
        parameters:
        -----------
        task_list: list
        feature_dict: dict
        """
        self.file.writelines(self.HEADER)
        self.write_catalog(task_list)
        self.write_info(task_list)
        self.write_condition(task_list, feature_dict)
        self.file.close()

# ====================================
# functions
# ====================================
def runcmd(command, verbose=0, timeout=1200):
    """
    run command line
    """
    ret = subprocess.run(command,shell=True,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            encoding="utf-8",timeout=timeout)
    if ret.returncode == 0 and verbose:
        print("success:",ret)
    elif ret.returncode != 0:
        print("error:",ret)

def check_path(path, mk=1):
    if type(path) == str:
        if not os.path.exists(path):
            if mk:
                os.mkdir(path)
                print("[news] Inspected: {} created.".format(path))
            else:
                print("[news] Inspected: {} not existed.".format(path))
            return 0
        else:
            print("[news] Inspected: {} have exsited".format(path))
            return 1
    else: 
        raise AssertionError("Input must be str")

def session_input_dict(scaninfo):
    """
    Generate a dict show where the data of each session from.
    parameter:
    ----------
    scaninfo： pd.DataFrame

    return:
    ---------
    session_input: dict
    """ 
    session_input = {}
    print("[news] Generating session-input mapping...")
    for _ in tqdm(range(len(scaninfo))):
        cur_run = scaninfo.iloc[_, :]
        for __ in cur_run['ses'].split(','):
            _key = "sub-{:02d}/ses-{:s}".format(cur_run['sub'], __)
            if _key not in session_input.keys():
                session_input[_key] = "{0}*{1}.tar.gz".format(cur_run['date'].strftime("%Y%m%d"), cur_run['name'])

    return session_input

def session_task_dict(scaninfo):
    """
    Generate a dict show where the data of each session from.
    Won't contain fmap/ cause every session should have one
    parameter:
    ----------
    scaninfo： pd.DataFrame

    return:
    ---------
    session_task: dict
    """ 
    session_task = {}
    print("[news] Generating session-task mapping...")
    for _ in tqdm(range(len(scaninfo))):
        cur_run = scaninfo.iloc[_, :]
        for __ in cur_run['ses'].split(','):
            _key = "sub-{:02d}/ses-{:s}".format(cur_run['sub'], __)
            if _key not in session_task.keys():
                session_task[_key] = ["{:s}/{:s}".format(cur_run['modality'], cur_run['task'])]
            else:
                value = "{:s}/{:s}".format(cur_run['modality'], cur_run['task'])
                if not value in session_task[_key]:
                    session_task[_key].append("{:s}/{:s}".format(cur_run['modality'], cur_run['task']))
    return session_task

def task_feature_dict(scaninfo):
    """
    Generate a dict ponit out feature of each task
    parameter:
    ----------
    scaninfo: pd.DataFrame

    return
    -------------
    task_feature: dict
    """
    task_feature = {}
    print("[news] Generating task-feature mapping...")
    for _ in tqdm(range(len(scaninfo))):
            cur_run = scaninfo.iloc[_, :]
            _key = "{:s}/{:s}".format(cur_run['modality'], cur_run['task'])
            if _key not in task_feature.keys():
                if not np.isnan(cur_run['dim4']):
                    task_feature[_key] = [cur_run['protocol_name'], int(cur_run['dim4'])]
                else:
                    task_feature[_key] = [cur_run['protocol_name'], None]
    return task_feature
      
# argparse
parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="path fetch to scaninfo.xlsx")
parser.add_argument("-b", "--base", type=str, \
    help="base dir contains all relevant bold data, usually is /../bold", default=None)
parser.add_argument("-i", "--input-dir", type=str, help="input", default='orig')
parser.add_argument("-t", "--temp-dir", type=str, help="temp folds", default='orig/dicom')
parser.add_argument("-o", "--output-dir", type=str, \
    help="name of directory stores nifti files with BIDS specifications", default='nifti')

parser.add_argument("-q", "--quality-filter", type=str, \
     help="quality filter on scaninfo.xlsx", choices=['ok', 'all','discard'], default='ok')
parser.add_argument("-s", "--subject", type=str, nargs="+", help="subjects")
parser.add_argument("-ss", "--session", type=str, nargs="+", help="sessions")
parser.add_argument("--overwrite", action="store_true", help="whether overwrite")
args = parser.parse_args()
# nargs:https://blog.csdn.net/kinggang2017/article/details/94036386

# step 1 Inspect directory
print(bcolors.BOLD_NODE + "[Node] Inspecting..." + bcolors.ENDC)
# inpect scaninfo.xlsx
# expect using relative path, put in base -- /../bold
# check input output & temp 
# inspect whether base match with BNL data orgniaztion specification
# how about nifiti/code/

if args.base:
    os.chdir(args.base)
    #
    if not os.path.exists(args.file):
        raise AssertionError(bcolors.FAIL +  "[Error] NOT FOUND {}".format(args.file) + bcolors.ENDC)
    #
    check_results = [check_path(_) for _ in [args.input_dir, args.temp_dir, args.output_dir]]
    if check_results[0] == 0:
        print("[Warning] {} is just created, it should be prepared before.".format(args.input_dir))
        if os.listdir(args.input_dir):
            raise AssertionError("No files found in {}".format(args.input_dir))
    if check_results[2] == 0:
        print("[Warning] {} is just created, it should be prepared before.".format(args.output_dir))
        check_path(pjoin(args.input_dir, 'code'))
    # 
    expect_child = np.array(['info', 'nifti', 'orig'])
    absent_child = expect_child[np.array([ _ not in os.listdir(args.base) for _ in expect_child ])]
    if absent_child:
        print(bcolors.WARNING + "[Warning] data catalog structure mismatches with BNL specifications" \
                + bcolors.ENDC)
        print(bcolors.WARNING + "  NOTFOUND： {0} absent in base path {1}".format(absent_child, args.base) \
                + bcolors.ENDC)
        print(bcolors.WARNING + "  It is encouraged to take use of BNL specification, though not coerced..." \
                + bcolors.ENDC)
else:
    if not os.path.exists(args.file):
        raise AssertionError(bcolors.FAIL +  "[Error] NOT FOUND {}".format(args.file) + bcolors.ENDC)
    expect_folds = np.array([args.input_dir, args.temp_dir, args.output_dir])
    absent_folds = expect_folds[np.array([check_path(_, 0) for _ in expect_folds])]
    if absent_folds:
        raise AssertionError(bcolors.FAIL +  "[Error] NOT FOUND {}".format(absent_folds) + bcolors.ENDC)

# step 2 Information Reorganization
print(bcolors.BOLD_NODE + "[Node] Re-organizing..." + bcolors.ENDC)
# filter at first
# traverse the scaninfo 
# generate session:input - determine 
# generate session:task
# generate task:feature
scaninfo_raw = pd.read_excel(args.file)
# pandas:https://www.cnblogs.com/ech2o/p/11831488.html

# 
if args.subject or args.session:
    scaninfo = scaninfo_raw
    if args.subject:
        scaninfo = scaninfo[scaninfo['sub'].isin(args.subject)]
        scaninfo.reset_index(drop=True)
    if args.session:
        scaninfo = scaninfo[scaninfo['ses'].isin(args.session)]
        scaninfo.reset_index(drop=True)
    if args.quality_filter != 'all':
        scaninfo = scaninfo[scaninfo['quality'] == args.quality_filter]
        scaninfo.reset_index(drop=True)
else:
    if args.quality_filter != 'all':
        scaninfo = scaninfo_raw[scaninfo_raw['quality'] == args.quality_filter]
        scaninfo.reset_index(drop=True)
    else:
        scaninfo = scaninfo_raw

# determine input of each session -- 
session_input = session_input_dict(scaninfo)
print("[news] Find {:d} session(s) waiting for processing..".format(len(session_input)))
for key, value in session_input.items():
    print(bcolors.BOLD+ "    {:s} ".format(key) + bcolors.ENDC+ "from: {:s}".format(value) )

# detemine session-contained tasks --
session_task = session_task_dict(scaninfo)
print("[news] Tasks in each sub-session collected")
heu_session_task = {}
for key, value in session_task.items():
    print(bcolors.BOLD+ "    {:s} ".format(key) + bcolors.ENDC+ "contains: {}".format(value) )
    s_key = (key.split('-')[-1]).strip(string.digits)
    if s_key not in heu_session_task.keys():
        heu_session_task[s_key] = value
    else:
        if set(value) - set(heu_session_task[s_key]):
            heu_session_task[s_key].extend(list(set(value) - set(heu_session_task[s_key])))
print("[news] Found {} kinds of session:".format(len(heu_session_task)))
for key, value in heu_session_task.items():
    print(bcolors.BOLD+ "    {:s} ".format(key) + bcolors.ENDC+ "contains: {}".format(value) )

# determine task feature -- task : [protocolname dim]
task_feature = task_feature_dict(scaninfo)
print("[news] Task feature information collected..")
for key, value in task_feature.items():
    print(bcolors.BOLD+ "    {:s} : ".format(key) + bcolors.ENDC + "protocolname = " + \
            bcolors.BOLD + "{0[0]},".format(value) + bcolors.ENDC +" dim = " + \
            bcolors.BOLD + "{0[1]} ".format(value) + bcolors.ENDC)

# step 3 Unpack 
print(bcolors.BOLD_NODE + "[Node] Unpacking..." + bcolors.ENDC)
for _key, _value in tqdm(session_input.items()):
    # upack
    if not glob.glob(pjoin(args.temp_dir,_value)):
        cmd = "tar -xzvf {:s} -C {:s}".format(pjoin(args.input_dir, _value), args.temp_dir)
        print("[news] Running command: {:s}".format(cmd))
        # runcmd(cmd)

        # 考虑要不要，if 要 放到后面？
        # # prepare for Inspceting
        # dcom_files = pjoin(args.input_dir,_value).replace('.tar.gz', '/*.IMA')
        # subID, sesID = _key.split('/')[0].replace('sub-'), _key.split('/')[1].replace('ses-')
        # cmd = "heudiconv --files {:s} -o {:s} -f convertall -s {:s} -ss {:s} -c none --overwrite"\
        #     .format(dcom_files, args.output_dir, subID, sesID)
        # print("[news] dicominfo")
        # runcmd(cmd)
        # dicominfo = pd.read_csv("{:s}/.heudiconv/{:s}/info/dicominfo_ses-{:s}.tsv"\
        #     .format(args.output_dir, subID, sesID), sep='\t')
        # # what squence (kinds of tasks) this scan contains
        # for _ in tqdm(range(len(dicominfo))):
        #     if dicominfo.iloc[_, :]:
    
# step 4 Heuristic.py generation
print(bcolors.BOLD_NODE + "[Node] Heuristic.py Generating..." + bcolors.ENDC)
# task_feature & heu_session_task will be used
for key, value in heu_session_task.items():
    check_path(pjoin(args.output_dir, 'code', key))
    file = pjoin(args.output_dir, 'code', key, 'heuristic.py')
    if not os.path.exists(file):
        heu_creation = heucreation(file)
        heu_creation.create_heuristic(value, task_feature)
print("[news] Heuristic.py completion!")

# step 5 heudiconv
print(bcolors.BOLD_NODE + "[Node] BIDS converting..." + bcolors.ENDC)
# session_input will be used
for _key, _value in tqdm(session_input.items()):
    dcom_files = pjoin(args.input_dir,_value).replace('.tar.gz', '/*.IMA')
    subID, sesID = _key.split('/')[0].replace('sub-', ''), _key.split('/')[1].replace('ses-', '')
    heuristicpy = pjoin(args.output_dir, 'code', sesID.strip(string.digits), 'heuristic.py')
    if args.overwrite:
        cmd = "heudiconv -files {:s} -o {:s} -f {:s} -s {:s} -ss {:s} -c dcm2niix -b --overwrite" \
            .format(dcom_files, args.output_dir, heuristicpy, subID, sesID)
    else:
        cmd = "heudiconv -files {:s} -o {:s} -f {:s} -s {:s} -ss {:s} -c dcm2niix -b" \
            .format(dcom_files, args.output_dir, heuristicpy, subID, sesID)
    print("[news] Processing sub-{:s}/ses-{:s}".format(subID, sesID))
    print("    command: " + bcolors.BOLD + "{:s}".format(cmd) + bcolors.ENDC)

# step 6 fmap filling


