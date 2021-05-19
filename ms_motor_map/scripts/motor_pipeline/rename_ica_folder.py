"""
    rename ica folder
"""

import os

melodic_dir = '/nfs/e4/function_guided_resection/MotorMap/data/bold/derivatives/melodic'
subject_list = os.listdir(melodic_dir)

for subject in subject_list:
    session_dir = os.path.join(melodic_dir, subject, 'ses-1')
    run_list = os.listdir(session_dir)

    for run in run_list:
        run_dir_old = os.path.join(session_dir, run)
        run_dir_new = os.path.join(session_dir, subject + '_' + 'ses-1' + '_' + 'task-motor' + '_' + run)
        print(run_dir_old + ' >>> ' + run_dir_new)
        os.rename(run_dir_old, run_dir_new)