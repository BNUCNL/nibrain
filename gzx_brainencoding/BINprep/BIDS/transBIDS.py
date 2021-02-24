#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 12:45:09 2020

!!!!!! Please read before:
This code uses docker nipy/heudeconv to transfer *dcom/IMA* files into BIDS type. 
Frist, the machine should have aready installed docker, 
  see:http://c.biancheng.net/view/3118.html and fllowing the tutorial,
  linux os is highly recommended.
Second, BIDS infomation see: https://bids.neuroimaging.io
More tutorial infomation about using heudeconv see: https://reproducibility.stanford.edu/bids-tutorial-series-part-2a/
@author: gongzhengxin
"""
import os
from os.path import join as pjoin
import subprocess
from tqdm import tqdm
# 
def check_path(path, verbose=0):
    if type(path) == str:
        if not os.path.exists(path):
            os.mkdir(path)
        elif verbose==1: 
            print('Already existed {}'.format(path))
    else: print('path should be str!')
# 
def runcmd(command, verbose=0):
    ret = subprocess.run(command,shell=True,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            encoding="utf-8",timeout=1200)
    if ret.returncode == 0 and verbose:
        print("success:",ret)
    else:
        print("error:",ret)
# %%
# get subject & session information 
# check your sessinfo.csv for addapting
with open('new_session_info.tsv','r') as info:
    information = [line.split('\t') for line in info.readlines()]
header = information[0]
content = information[1:]

# rearange into a dict
# {'name': {'date': 'sessname'; 'folder': 'foldname'}}
session_dict = {}
for line in content:
    # get useful var
    name, ID, sess, status, date = line[0], line[1], line[2], line[-2], line[-1][:8]
    if not name in session_dict.keys():
        session_dict[name] = {'folder':'sub-core{:s}'.format(ID[-2:])}
    if status == 'OK':
        if 'session' in sess: 
            sessnum = eval(sess[sess.index('session')+7])
            session_dict[name][date] = 'sess-ImageNet{:02d}'.format(sessnum)
        elif 'Test' in sess:
            session_dict[name][date] = 'sess-COCO'
        elif 'LOC' in sess:
            session_dict[name][date] = 'sess-LOC'
        else:
            session_dict[name][date] = 'sess-'+sess
    else:
        session_dict[name][date] = 'sess-others'
del line, information

# prepare folders, check if not create
# !!! assignments should be adpated according to current machine
compressed_data_folder = '.targzFiles' # targzFile is where all .tar.gz data file stored
dicom_folder, nifiti_folder = 'sourcedata', 'rawdata' # where store the Dcom date & Nifiti data
# check whether compressed data files in the current working path
if compressed_data_folder in os.listdir(os.getcwd()):
    check_path(dicom_folder, 1)
    check_path(nifiti_folder, 1)
    for key,value in session_dict.items():
        sub_folder = pjoin(dicom_folder, value['folder'])
        check_path(sub_folder)
else:
    raise AssertionError('current path is not appropriate!\n check path to change to where there is data folder ')
# %%
# heuristic.py should be placed in the folder of nifiti_fold/code
# then excute the next section
check_path(pjoin(nifiti_folder, "code"))
if 'heuristic.py' in  os.listdir(pjoin(nifiti_folder, "code")):
    print('Yes')
else:
    raise AssertionError("Please check the heuristic.py! It should be palced in 'nifti_fold/code'!")
# %%
# get all the .tar.gz files
targzFiles = [line for line in os.listdir(compressed_data_folder) if '.tar.gz' in line] 
for file in tqdm(targzFiles[:]):
    # get the date & name information
    date, name = file.split('_')[0], file.split('_')[-1].replace('.tar.gz','') 
    # prepare all the foldnames
    try:
        target_folder = pjoin(dicom_folder,session_dict[name]['folder']) # sub folder ./sub-core0x 
        # after decompressing
        decompressed_foldname = file.replace('.tar.gz','') # where store IMA files
        session_foldername = session_dict[name][date] # standard fildname: sess-label0x
        trgname = pjoin(target_folder, session_foldername)
        final_fold = pjoin(nifiti_folder,session_dict[name]['folder'],session_foldername)
    except KeyError:
        print('%s-%s NOT IN sessinfo.tsv, pass processing' % (name, date) )
        continue
    if not os.path.exists(final_fold): # if the final fold is empty then run next
        if not os.path.exists(trgname): # if decompressed files are not existed
            # First, decompress the .tar.gz to target sub folder
            decompress_cmd = "tar -xzvf ./{:s} -C ./{:s}".format(pjoin(compressed_data_folder,file), target_folder)
            runcmd(decompress_cmd)
            
            # Second, rename the fold
            os.rename(pjoin(target_folder, decompressed_foldname), trgname)
        else: 
            print('have deteced sourcedata: {:s}'.format(trgname))
        # Third, generate .heudeconv fold, 
        # !!! this should compitable to sub folder & sess folder name
        # $$$$ docker command line
#        base_fold = "/nfs/m1/BrainImageNet/fMRIData"
#        dcom_files = "/base/"+dicom_folder+"/sub-{subject}/sess-{session}/*.IMA"
#        nifi_fold = "/base/"+ nifiti_folder
#        subID, sessID = session_dict[name]['folder'].replace("sub-", ''), session_foldername.replace("sess-", '')
#        gen_heu_cmd = "docker run --rm -i -v {:s}:/base nipy/heudiconv:latest -d {:s} -o {:s} -f convertall -s {:s} -ss {:s} -c none --overwrite".format(base_fold,
#                dcom_files, nifi_fold, subID, sessID)
        
        # !!! $$$$ local command line
        base_fold = "/nfs/m1/BrainImageNet/fMRIData"
        dcom_files = "$base/"+dicom_folder+"/sub-{subject}/sess-{session}/*.IMA"
        nifi_fold = "$base/"+ nifiti_folder
        subID, sessID = session_dict[name]['folder'].replace("sub-", ''), session_foldername.replace("sess-", '')
        gen_heu_cmd = "base=\"{:s}\"; heudiconv -d {:s} -o {:s} -f convertall -s {:s} -ss {:s} -c none --overwrite".format(base_fold, dcom_files, nifi_fold, subID, sessID)
        
        #runcmd(gen_heu_cmd)
        
        # Last, heuristic.py should be stored at the Nifitifolder/code
        # !!! $$$$ docker command line
        # heuristicpy = "/"+nifiti_folder+"/code/heuristic.py"
        # decom2bids_cmd1 = "docker run --rm -i -v {:s}:/base nipy/heudiconv:latest -d {:s} -o {:s}".format(base_fold, dcom_files, nifi_fold)
        # decom2bids_cmd2 = " -f /base{:s} -s {:s} -ss {:s} -c dcm2niix -b --overwrite".format(heuristicpy, subID, sessID)
        # decom2bids_cmd = decom2bids_cmd1 + decom2bids_cmd2
      
        # !!! $$$$ local command lin
        heuristicpy = "/"+nifiti_folder+"/code/heuristic.py"
        decom2bids_cmd1 = "base=\"{:s}\"; heudiconv -d {:s} -o {:s}".format(base_fold, dcom_files, nifi_fold)
        decom2bids_cmd2 = " -f $base{:s} -s {:s} -ss {:s} -c dcm2niix -b --overwrite".format(heuristicpy, subID, sessID)
        decom2bids_cmd = decom2bids_cmd1 + decom2bids_cmd2
        
#        runcmd(decom2bids_cmd)
#        
    else:
        print('{:s} already existed!'.format(trgname))

    








