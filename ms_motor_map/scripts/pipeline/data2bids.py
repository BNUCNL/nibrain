"""
    transform dicom data to bids style
"""

import os, glob, string, logging, argparse, subprocess, json
from tqdm import tqdm
import pandas as pd
import numpy as np
from os.path import join as pjoin

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
        self.file = open(file, 'w')
        self.HEADER = [
            'import os\ndef create_key(template, outtype=(\'nii.gz\',), annotation_classes=None):\n    if template is None or not template:\n        raise ValueError(\'Template must be a valid format string\')\n    return template, outtype, annotation_classes\ndef infotodict(seqinfo):\n    """Heuristic evaluator for determining which runs belong where\n    allowed template fields - follow python string module:\n    item: index within category\n    subject: participant id\n    seqitem: run number during scanning\n    subindex: sub index within group\n    """\n\n']

    def write_catalog(self, task_list):
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
                content.append(
                    "    {0}_{1} = create_key('sub-{{subject}}/{{session}}/{0}/sub-{{subject}}_{{session}}_run-{{item:02d}}_{1}')\n" \
                    .format(mod, label))
            if mod in ['func']:
                content.append(
                    "    {0}_{1} = create_key('sub-{{subject}}/{{session}}/{0}/sub-{{subject}}_{{session}}_task-{1}_run-{{item:02d}}_bold')\n" \
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
                middle.append(
                    "        if ('{0[0]}' in s.protocol_name) and (s.dim3 == {0[1]}):\n".format(feature_dict[_]))
                middle.append("            info[{0}_{1}].append(s.series_id)\n".format(mod, label))
            if mod == 'func':
                middle.append(
                    "        if ('{0[0]}' in s.protocol_name) and (s.dim4 == {0[1]}):\n".format(feature_dict[_]))
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
    ret = subprocess.run(command, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         encoding="utf-8", timeout=timeout)
    if ret.returncode == 0 and verbose:
        print("success:", ret)
    elif ret.returncode != 0:
        print("error:", ret)


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
    scaninfo:pd.DataFrame

    return:
    ---------
    session_input: dict
    """
    session_input = {}
    print("[news] Generating session-input mapping...")
    for _ in tqdm(range(len(scaninfo))):
        cur_run = scaninfo.iloc[_, :]
        _key = "sub-{:02d}/ses-{:s}".format(cur_run['sub'], str(cur_run['ses']))
        if _key not in session_input.keys():
            session_input[_key] = "{0}*{1}.tar.gz".format(cur_run['date'].strftime("%Y%m%d"), cur_run['name'])

    return session_input


def session_task_dict(scaninfo):
    """
    Generate a dict show where the data of each session from.
    Won't contain fmap/ cause every session should have one
    parameter:
    ----------
    scaninfo: pd.DataFrame

    return:
    ---------
    session_task: dict
    """
    session_task = {}
    print("[news] Generating session-task mapping...")
    for _ in tqdm(range(len(scaninfo))):
        cur_run = scaninfo.iloc[_, :]
        for __ in str(cur_run['ses']).split(','):
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
            if not np.isnan(cur_run['dim']):
                task_feature[_key] = [cur_run['protocol_name'], int(cur_run['dim'])]
            else:
                task_feature[_key] = [cur_run['protocol_name'], None]
    return task_feature

def log_config(log_name):
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO, filename=log_name, filemode='w')

def data2bids(args):

    # prepare parameter and path
    orig_dir = pjoin(args.projectdir, 'data', 'bold', 'orig')
    dicom_dir = pjoin(args.projectdir, 'data', 'bold', 'dicom')
    nifti_dir = pjoin(args.projectdir, 'data', 'bold', 'nifti')
    info_dir = pjoin(args.projectdir, 'data', 'bold', 'info')

    # prepare logging
    log_config(pjoin(nifti_dir, 'data2bids.log'))

    # step 2 Information Reorganization
    print(bcolors.BOLD_NODE + "[Node] Re-organizing..." + bcolors.ENDC)
    # filter at first
    # traverse the scaninfo
    # generate session:input - determine
    # generate session:task
    # generate task:feature
    scaninfo_raw = pd.read_excel(pjoin(info_dir, args.scaninfo))
    # pandas:https://www.cnblogs.com/ech2o/p/11831488.html

    if args.subject or args.session:
        scaninfo = scaninfo_raw
        if args.subject:
            logging.info('DATA2BIDS: Selected subject is {}'.format(args.subject))
            scaninfo = scaninfo[scaninfo['sub'].isin(args.subject)]
            scaninfo.reset_index(drop=True)
        if args.session:
            logging.info('DATA2BIDS: selected session is {}'.format(args.session))
            scaninfo = scaninfo[scaninfo['ses'].isin(args.session)]
            scaninfo.reset_index(drop=True)
        if args.quality_filter != 'all':
            logging.info('DATA2BIDS: quality filter is {}'.format(args.quality_filter))
            logging.info('DATA2BIDS: filtered scaninfo is in {}'.format(pjoin(nifti_dir, 'scaninfo_filtered.xlsx')))
            scaninfo = scaninfo[scaninfo['quality'] == args.quality_filter]
            scaninfo.reset_index(drop=True)
            scaninfo.to_excel(pjoin(nifti_dir, 'scaninfo_filtered.xlsx'))
    else:
        if args.quality_filter != 'all':
            logging.info('DATA2BIDS: quality filter is {}'.format(args.quality_filter))
            logging.info('DATA2BIDS: filtered scaninfo is stored in {}'.format(pjoin(nifti_dir, 'scaninfo_filtered.xlsx')))
            scaninfo = scaninfo_raw[scaninfo_raw['quality'] == args.quality_filter]
            scaninfo.reset_index(drop=True)
            scaninfo.to_excel(pjoin(nifti_dir, 'scaninfo_filtered.xlsx'))
        else:
            logging.info('DATA2BIDS: process all parts in {}'.format(args.scaninfo))
            scaninfo = scaninfo_raw

    # determine input of each session -- {sub-*/ses-* : *.tar.gz}
    session_input = session_input_dict(scaninfo)
    print("[news] Find {:d} session(s) waiting for processing..".format(len(session_input)))
    for key, value in session_input.items():
        print(bcolors.BOLD + "    {:s} ".format(key) + bcolors.ENDC + "from: {:s}".format(value))
    logging.info('DATA2BIDS: session-input mapping is stored in {}'.format(pjoin(nifti_dir, 'session-input.json')))


    # detemine session-contained tasks --
    session_task = session_task_dict(scaninfo)
    print("[news] Tasks in each sub-session collected")
    heu_session_task = {}
    for key, value in session_task.items():
        print(bcolors.BOLD + "    {:s} ".format(key) + bcolors.ENDC + "contains: {}".format(value))
        s_key = (key.split('-')[-1]).strip(string.digits)
        if s_key not in heu_session_task.keys():
            heu_session_task[s_key] = value
        else:
            if set(value) - set(heu_session_task[s_key]):
                heu_session_task[s_key].extend(list(set(value) - set(heu_session_task[s_key])))
    print("[news] Found {} kinds of session:".format(len(heu_session_task)))
    for key, value in heu_session_task.items():
        print(bcolors.BOLD + "    {:s} ".format(key) + bcolors.ENDC + "contains: {}".format(value))
    logging.info('DATA2BIDS: session-task mapping is stored in {}'.format(pjoin(nifti_dir, 'session-task.json')))

    # determine task feature -- task : [protocolname dim]
    task_feature = task_feature_dict(scaninfo)
    print("[news] Task feature information collected..")
    for key, value in task_feature.items():
        print(bcolors.BOLD + "    {:s} : ".format(key) + bcolors.ENDC + "protocolname = " + \
              bcolors.BOLD + "{0[0]},".format(value) + bcolors.ENDC + " dim = " + \
              bcolors.BOLD + "{0[1]} ".format(value) + bcolors.ENDC)
    logging.info('DATA2BIDS: task-feature mapping is stored in {}'.format(pjoin(nifti_dir, 'task-feature.json')))

    # step 3 Unpack
    if not args.skip_unpack:
        print(bcolors.BOLD_NODE + "[Node] Unpacking..." + bcolors.ENDC)

        for _value in tqdm([__ for __ in session_input.values()]):
            # upack
            if not glob.glob(pjoin(dicom_dir, _value.replace('.tar.gz', ''))):
                cmd = "tar -xzvf {:s} -C {:s}".format(pjoin(orig_dir, _value), dicom_dir)
                logging.info('Unpack command: {:s}'.format(cmd))
                print("[news] Running command: {:s}".format(cmd))
                if not args.preview:
                    runcmd(cmd)

    # step 4 Heuristic.py generation
    print(bcolors.BOLD_NODE + "[Node] Heuristic.py Generating..." + bcolors.ENDC)
    # task_feature & heu_session_task will be used
    for key, value in heu_session_task.items():
        check_path(pjoin(nifti_dir, 'code', key))
        file = pjoin(nifti_dir, 'code', key, 'heuristic.py')

        heu_creation = heucreation(file)
        heu_creation.create_heuristic(value, task_feature)
    print("[news] Heuristic.py completion!")

    # step 5 heudiconv
    print(bcolors.BOLD_NODE + "[Node] BIDS converting..." + bcolors.ENDC)
    # session_input will be used
    for _key, _value in tqdm(session_input.items()):
        dicom_files = pjoin(dicom_dir, _value).replace('.tar.gz', '/*.IMA')
        subID, sesID = _key.split('/')[0].replace('sub-', ''), _key.split('/')[1].replace('ses-', '')

        if not args.skip_feature_validation:
            # feature validation
            if args.overwrite:
                cmd = "heudiconv --files {:s} -o {:s} -f convertall -s {:s} -ss {:s} -c none --overwrite" \
                    .format(dicom_files, nifti_dir, subID, sesID)
            else:
                cmd = "heudiconv --files {:s} -o {:s} -f convertall -s {:s} -ss {:s} -c none" \
                    .format(dicom_files, nifti_dir, subID, sesID)
            print("[news] inspecting task feature in dicominfo.tsv")
            logging.info('Heudiconv command: {:s}'.format(cmd))
            print("[news] command:" + bcolors.BOLD + " {}".format(cmd) + bcolors.ENDC)
            if not args.preview:
                runcmd(cmd)
                dicominfo = pd.read_csv("{:s}/.heudiconv/{:s}/info/dicominfo_ses-{:s}.tsv" \
                                        .format(nifti_dir, subID, sesID), sep='\t')
                dicominfo_scan_feature = list(
                    set([(dicominfo.iloc[_run, :]['protocol_name'], dicominfo.iloc[_run, :]['dim4']) \
                             if dicominfo.iloc[_run, :]['dim4'] != 1 else (
                        dicominfo.iloc[_run, :]['protocol_name'], dicominfo.iloc[_run, :]['dim3']) \
                         for _run in range(len(dicominfo))]))

                _check = []
                for _task in session_task[_key]:
                    _feature = (task_feature[_task][0], task_feature[_task][1])
                    if 'anat' in _task:
                        if not any([_feature[0] == __[0] for __ in dicominfo_scan_feature]):
                            _check.append(any([_feature[0] == __[0] for __ in dicominfo_scan_feature]))
                            logging.critical("'{:s}' protocol name mismtach! Found no {:s} in {:s}/.heudiconv/{:s}/info/dicominfo_ses-{:s}.tsv" \
                                             .format(_task, _feature[0], nifti_dir, subID, sesID))
                            print(bcolors.FAIL + \
                                  "[ERROR] '{:s}' protocol name mismtach! Found no {:s} in {:s}/.heudiconv/{:s}/info/dicominfo_ses-{:s}.tsv" \
                                  .format(_task, _feature[0], nifti_dir, subID, sesID) + bcolors.ENDC)
                    else:
                        if not _feature in dicominfo_scan_feature:
                            _check.append(_feature in dicominfo_scan_feature)
                            logging.critical('"'+_task+'" protocol name mismtach! Found no '+str(_feature)+' in '+nifti_dir+'/.heudiconv/'+subID+'/info/dicominfo_ses-'+sesID+'.tsv')
                            print('[ERROR] "'+_task+'" protocol name mismtach! Found no '+str(_feature)+' in '+nifti_dir+'/.heudiconv/'+subID+'/info/dicominfo_ses-'+sesID+'.tsv')
                if not all(_check):
                    logging.critical('Feature validation failure!')
                    raise AssertionError(
                        '[ERROR] Feature validation failure! Please read [ERROR] message above or log for more details!')
                print(bcolors.BOLD + "[news] Feature validation seuccess!" + bcolors.ENDC)
                del _task, _feature

        heuristicpy = pjoin(nifti_dir, 'code', sesID.strip(string.digits), 'heuristic.py')
        if args.overwrite:
            cmd = "heudiconv --files {:s} -o {:s} -f {:s} -s {:s} -ss {:s} -c dcm2niix -b --overwrite" \
                .format(dicom_files, nifti_dir, heuristicpy, subID, sesID)
        else:
            cmd = "heudiconv --files {:s} -o {:s} -f {:s} -s {:s} -ss {:s} -c dcm2niix -b" \
                .format(dicom_files, nifti_dir, heuristicpy, subID, sesID)
        print("[news] Processing sub-{:s}/ses-{:s}".format(subID, sesID))
        logging.info("Heudiconv command (overwrite): {:s}".format(cmd))
        print("command: " + bcolors.BOLD + "{:s}".format(cmd) + bcolors.ENDC)
        if not args.preview:
            runcmd(cmd, timeout=3600)

    # fill fmap json files
    print(bcolors.BOLD_NODE + "[Node] .json Filling up..." + bcolors.ENDC)

    if args.subject:
        subjects = ["{:02d}".format(int(_)) for _ in args.subject]
    else:
        subjects = [_.replace("sub-", "") for _ in os.listdir(nifti_dir) if "sub-" in _]

    sessions = {name: [] for name in (subjects)}  # {sub : [session]}
    jsonfiles = {name: {sesname: [] for sesname in sessions[name]} for name in
                 (subjects)}  # {sub:{session:[files]}}
    intendedfornii = {name: {sesname: [] for sesname in sessions[name]} for name in
                      (subjects)}  # {sub:{session:[files]}}

    # collect .json files waiting to fill & .nii.gz filenames
    if not args.preview:
        print("[news] collect .json files waiting to fill & .nii.gz filenames")
        for subname in sessions.keys():
            # get all the sessions under a subject
            subpth = pjoin(nifti_dir, 'sub-%s' % (subname))
            sessions[subname] = os.listdir(subpth)
            # collect jsonfiles & values
            for fold in sessions[subname]:
                # path preparation
                sesspth = pjoin(nifti_dir, subpth, fold)
                fmappth = pjoin(sesspth, 'fmap')
                funcpth = pjoin(sesspth, 'func')
                # if fmap exist then clollect
                if os.path.exists(fmappth):
                    jsonfiles[subname][fold] = [file for file in os.listdir(fmappth) if '.json' in file]
                    # the file path must be the relative path to sub- folder
                    intendedfornii[subname][fold] = ['%s/func/%s' % (fold, file) for file in os.listdir(funcpth) if
                                                     '.nii.gz' in file]

    # write key:value for each json
        print("[news] write key:value for each json")
        for sub, ses_fold in jsonfiles.items():
            for ses, files in ses_fold.items():
                for file in files:
                    # file path
                    file_path = os.path.join(nifti_dir, 'sub-%s/%s' % (sub, ses), 'fmap', file)
                    
                    # change mode
                    chmod_cmd = ' '.join(['chmod', '755', file_path])
                    runcmd(chmod_cmd, timeout=3600)

                    # load in & add IntendedFor
                    with open(file_path, 'r') as datafile:
                        data = json.load(datafile)
                    data['IntendedFor'] = intendedfornii[sub][ses]

                    # save out
                    with open(file_path, 'w') as datafile:
                        json.dump(data, datafile)
            print('[news] fill up json for %s ... done!' % sub)

    print('Log is saved in {}.'.format(pjoin(nifti_dir, 'data2bids.log')))

if __name__ == '__main__':

    # initialize argparse
    parser = argparse.ArgumentParser()

    """
       required parameters 
    """
    parser.add_argument("projectdir", help="base dir contains all project files")
    parser.add_argument("scaninfo", help="filename of scaninfo, default is <scaninfo.xlsx>", default='scaninfo.xlsx')
    
    """
        optinal parameters 
    """
    parser.add_argument("-q","--quality-filter", type=str, help="quality filter on scaninfo.xlsx", choices=['ok', 'all', 'discard'], default='ok')
    parser.add_argument("-s","--subject", type=str, nargs="+", help="subjects")
    parser.add_argument("-ss","--session", type=str, nargs="+", help="sessions")
    parser.add_argument("-p","--preview", action="store_true", help="if choose, user can preview the whole pipeline and inspect critical information without runing any process command")
    parser.add_argument("--skip-feature-validation", action="store_true", help="if choose, pipeline will not compare scan features between scaninfo.xlsx and dicom.tsv")
    parser.add_argument("--overwrite", action="store_true", help="if choose, heudiconv will overwrite the existed files")
    parser.add_argument("--skip-unpack", action="store_true", help="if choose, pipeline will skip upack")

    args = parser.parse_args()

    # data2bids
    data2bids(args)
