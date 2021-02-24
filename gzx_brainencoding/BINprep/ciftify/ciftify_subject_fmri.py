# %%
import os
import subprocess
from os.path import join as pjoin
# %%

# input path includes all subject folds
input_path = '/nfs/m1/BrainImageNet/fMRIData/derivatives/fmriprep'

# flags for selection
sub_flag = ['sub-core']
ses_flag = ['ses-']
mod_flag = ['func']
file_flag = ['space-T1w_desc-preproc_bold.nii.gz', 'retinotopy']

# initialize func_files
func_files = []

# generate all run files
subject_folds = [ _ for _ in os.listdir(input_path) if any([ __ in _ for __ in \
                 sub_flag]) and ('.' not in _)]
for subject in subject_folds:
    session_folds = [ _ for _ in os.listdir(pjoin(input_path, subject)) \
                     if any([ __ in _ for __ in ses_flag]) and ('.' not in _)]
    for session in session_folds:
        modality_folds = [ _ for _ in os.listdir(pjoin(input_path, subject, session)) \
                     if any([ __ in _ for __ in mod_flag]) and ('.' not in _)]
        for modality in modality_folds:
            current_files = [ _ for _ in os.listdir(pjoin(input_path, subject, session, modality)) \
                     if all([ __ in _ for __ in file_flag])]
            func_files.extend([pjoin(input_path,subject,session,modality,file) for file in \
                  current_files])
# %%
# run cml
ciftify_work_dir = '/nfs/m1/BrainImageNet/fMRIData/derivatives/ciftify'
ev_cmd = 'export CIFTI_WORKDIR=' + ciftify_work_dir
subprocess.call(ev_cmd, shell=True)
cmd_pre = 'ciftify_subject_fmri --surf-reg MSMSulc --ciftify-work-dir ' + ciftify_work_dir + ' '
for func_file in func_files[1:2]:
    out_name = os.path.basename(func_file).split('_')
    subject = [i for i in out_name if i.startswith('sub-')][0]
    out_name = '_'.join([i for i in out_name
                         if i.startswith('ses-') or i.startswith('task-') or i.startswith('run-')])
    cmd = cmd_pre + ' '.join([func_file, subject, out_name])
    print(cmd)
    subprocess.call(cmd, shell=True)


