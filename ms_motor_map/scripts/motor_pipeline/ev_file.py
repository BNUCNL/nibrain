"""
    prepare events files
"""

import os, re

old_bids_dir = '/nfs/e4/function_guided_resection/MotorMapping'
new_bids_dir = '/nfs/e4/function_guided_resection/MotorMap/data/bold/nifti'

subjid_list = [sub.replace("sub-", "") for sub in os.listdir(new_bids_dir) if "sub-" in sub]

for subjid in subjid_list:
    # /nfs/e4/function_guided_resection/MotorMapping/sub-M01/ses-01/func/
    old_func_dir = os.path.join(old_bids_dir, 'sub-M' + subjid, 'ses-01', 'func')
    # /nfs/e4/function_guided_resection/MotorMap/data/bold/nifti/sub-01/ses-1/func/
    new_func_dir = os.path.join(new_bids_dir, 'sub-' + subjid, 'ses-1', 'func')
    # reindex old run id
    old_runid_list = []
    for filename in os.listdir(old_func_dir):
        if '_events.tsv' in filename:
            old_runid_list.append(str(re.findall('run-(.+?)_events', filename)[0]))
    old_runid_list.sort(key=int)
    # print(old_runid_list)
    runid_dict = {v: str(k+1) for k, v in dict(enumerate(old_runid_list)).items()}
    # print(runid_dict)
    for old_runid, new_runid in runid_dict.items():
        print('*****************************************')
        print('subject: ' + subjid)
        print('old:' + old_runid + '--new:' + new_runid)
        # old_ev_file: sub-M01_ses-01_task-motor_run-10_events.tsv
        with open(os.path.join(old_func_dir, 'sub-M' + subjid + '_ses-01_task-motor_run-' + old_runid + '_events.tsv')) as old_ev:
            # sub-01_ses-1_task-motor_run-01_events.tsv
            with open(os.path.join(new_func_dir, 'sub-' + subjid + '_ses-1_task-motor_run-0' + new_runid + '_events.tsv'), 'w') as new_ev:
                new_ev.write(old_ev.read())
                print(subjid + ': ' + old_runid + ' --> ' + new_runid)