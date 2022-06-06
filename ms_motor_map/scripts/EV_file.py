import os, subprocess
import pandas as pd

# subject_list = ['sub-04', 'sub-23', 'sub-27', 'sub-46']
# run_list = ['run-1', 'run-2', 'run-3', 'run-4', 'run-5', 'run-6']
#
# # EV
# for subject in subject_list:
#     subject_id = subject.replace('sub-', '')
#     old_subject = 'sub-M' + subject_id
#     new_subject = 'sub-' + subject_id
#
#     old_run_list = []
#     for fold in os.listdir(os.path.join('/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/ciftify/', old_subject, 'MNINonLinear/Results')):
#         if 'ses-01_task-motor_run-' in fold:
#             old_run_list.append(fold.replace('ses-01_task-motor_run-', ''))
#     old_run_list.sort(key=int)
#     run_dict = {v: str(k + 1) for k, v in dict(enumerate(old_run_list)).items()}
#
#     for old_runid, new_runid in run_dict.items():
#         old_dir = os.path.join('/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/ciftify',
#                                old_subject,
#                                'MNINonLinear/Results',
#                                'ses-01_task-motor_run-'+old_runid,
#                                'EVs')
#         new_dir = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/ciftify/',
#                                new_subject,
#                                'MNINonLinear/Results',
#                                'ses-1_task-motor_run-'+new_runid,
#                                'EVs')
#         cp_cmd = ' '.join(['cp', '-r', old_dir, new_dir])
#         print(cp_cmd)
#         subprocess.check_call(cp_cmd, shell=True)


subject_list = pd.read_csv('/nfs/z1/userhome/MaSai/workingdir/Motor_project/code/task_analysis_group/subject_list.csv',
                           header=None)[0].to_list()
run_list = ['run-1', 'run-2', 'run-3', 'run-4', 'run-5', 'run-6']
for subject in subject_list:
    print(subject)
    level2_dir = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/ciftify_orig',
                              subject, 'MNINonLinear/Results/ses-1_task-motor', )
    if not os.path.exists(level2_dir):
        os.makedirs(level2_dir)
    for run in run_list:
        print(run)
        source_ev = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/ciftify',
                                 subject, 'MNINonLinear/Results', 'ses-1_task-motor_'+run, 'EVs')
        target_ev = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/ciftify_orig',
                                 subject, 'MNINonLinear/Results', 'ses-1_task-motor_'+run, 'EVs')
        cp_cmd = ' '.join(['cp', '-r', source_ev, target_ev])
        if not os.path.exists(target_ev):
            subprocess.check_call(cp_cmd, shell=True)

        source_file = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/ciftify_orig',
                                   subject, 'MNINonLinear/Results', 'ses-1_task-motor_'+run,
                                   'ses-1_task-motor_'+run+'_Atlas_s0.dtseries.nii')
        target_file = os.path.join('/nfs/z1/zhenlab/MotorMap/data/bold/derivatives/ciftify_orig',
                                   subject, 'MNINonLinear/Results', 'ses-1_task-motor_'+run,
                                   'ses-1_task-motor_'+run+'_Atlas.dtseries.nii')
        mv_cmd = ' '.join(['mv', source_file, target_file])
        if not os.path.exists(target_file):
            subprocess.check_call(mv_cmd, shell=True)