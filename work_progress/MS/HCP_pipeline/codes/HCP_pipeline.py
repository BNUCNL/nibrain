"""
    hcp_pipeline.py
"""



import os
import pandas as pd
import numpy as np
import subprocess

# Three steps: fmri preprocess, ciftify and task analysis

class hcp_pipeline(object):

    def __init__(self, raw_data_dir, fmriprep_output_dir, fmriprep_workdir, ciftify_workdir, fsf_template_dir, subject_list, task):
        self.raw_data_dir = raw_data_dir
        self.fmriprep_output_dir = fmriprep_output_dir
        self.fmriprep_workdir = fmriprep_workdir
        self.ciftify_workdir = ciftify_workdir
        self.fsf_template_dir = fsf_template_dir
        self.subject_list = subject_list
        self.task = task

    # First step: fmri preprocess (fmriprep and fsl_regfilt)

    def fmriprep(self):
        for subject in self.subject_list:

            # fmriprep command options:
            # Usage:
            #     --participant_label: A space delimited list of participant identifiers or a single identifier
            #     --use-aroma: Add ICA_AROMA to your preprocessing stream
            #     --use-syn-sdc: Use fieldmap-free distortion correction
            #     --output-space: Standard and non-standard spaces to resample anatomical and functional images to.

            fmriprep_command = ' '.join(['fmriprep', self.raw_data_dir, self.fmriprep_output_dir, 'participant',
                                         '-w', self.fmriprep_workdir,
                                         '--participant_label', subject,
                                         '--use-aroma', '--use-syn-sdc', '--output-space', 'T1w'])
            try:
                subprocess.check_call(fmriprep_command, shell=True)
            except subprocess.CalledProcessError:
                raise Exception('FMRIPREP: Error happened in subject {}'.format(subject))

            # Non-aggressive denoising can be manually performed in the T1w space with fsl_regfilt.
            # fsl_regfilt command options:
            #     -i: sub-<subject_label>_task-<task_id>_space-T1w_desc-preproc_bold.nii.gz
            #     -f $(cat sub-<subject_label>_task-<task_id>_AROMAnoiseICs.csv)
            #     -d sub-<subject_label>_task-<task_id>_desc-MELODIC_mixing.tsv
            #     -o sub-<subject_label>_task-<task_id>_space-T1w_desc-AROMAnonaggr_bold.nii.gz

            session_list = os.listdir(os.path.join(self.raw_data_dir, subject))
            for session in session_list:
                with open(os.path.join(self.raw_data_dir, subject, session, 'tmp', 'run_info', self.task + '.rlf'), 'r') as f:
                    runs_list = f.read().splitlines()
                for run in runs_list:
                    i = os.path.join(self.fmriprep_output_dir, 'fmriprep', subject, session, 'func',
                                     subject + '_' + session + '_' + 'task-' + self.task + '_' + run + '_space-T1w_desc-preproc_bold.nii.gz')
                    f = os.path.join(self.fmriprep_output_dir, 'fmriprep', subject, session, 'func',
                                     subject + '_' + session + '_' + 'task-' + self.task + '_' + run + '_AROMAnoiseICs.csv')
                    d = os.path.join(self.fmriprep_output_dir, 'fmriprep', subject, session, 'func',
                                     subject + '_' + session + '_' + 'task-' + self.task + '_' + run + '_desc-MELODIC_mixing.tsv')
                    o = os.path.join(self.fmriprep_output_dir, 'fmriprep', subject, session, 'func',
                                     subject + '_' + session + '_' + 'task-' + self.task + '_' + run + '_space-T1w_desc-AROMAnonaggr_bold.nii.gz')
                    fsl_regfilt_command = ' '.join(['fsl_regfilt',
                                                    '-i', i,
                                                    '-f', '$(cat {})'.format(f),
                                                    '-d', d,
                                                    '-o', o])
                    try:
                        subprocess.check_call(fsl_regfilt_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('FSLREG: Error happened in subject {}'.format(subject))

    # Second step: ciftify workflow (ciftify_recon_all, ciftify_subject_fmri)

    def ciftify(self):
        for subject in self.subject_list:

            # ciftify_recon_all: Will convert any freeserfer output directory into an HCP (cifti space) output directory
            # Usage: ciftify_recon_all [options] <Subject>
            #     --resample-to-T1w32k: Resample the Meshes to 32k Native (T1w) Space
            #     --surf-reg: Registration sphere prefix [default: MSMSulc]
            #     --ciftify-work-dir: The directory for HCP subjects
            #     --fs-subjects-dir: Path to the freesurfer SUBJECTS_DIR directory

            freesurfer_dir = os.path.join(self.fmriprep_output_dir, 'freesurfer')
            ciftify_recon_all_command = ' '.join(['ciftify_recon_all',
                                         '--resample-to-T1w32k',
                                         '--surf-reg', 'MSMSulc',
                                         '--ciftify-work-dir', self.ciftify_workdir,
                                         '--fs-subjects-dir', freesurfer_dir,
                                         subject])
            try:
                subprocess.check_call(ciftify_recon_all_command, shell=True)
            except subprocess.CalledProcessError:
                raise Exception('CIFTIFY_RECON_ALL: Error happened in {}'.format(subject))

            session_list = os.listdir(os.path.join(self.raw_data_dir, subject))
            for session in session_list:
                with open(os.path.join(self.raw_data_dir, subject, session, 'tmp', 'run_info', self.task + '.rlf'), 'r') as f:
                    runs_id = f.read().splitlines()
                for run in runs_id:

                    # ciftify_subject_fmri: Will project a nifti functional scan to a cifti .dtseries.nii in that subjects hcp analysis directory
                    # Usage: ciftify_subject_fmri [options] <func.nii.gz> <subject> <task_label>
                    #     --ciftify-work-dir: The ciftify working directory
                    #     --surf-reg: Registration sphere prefix [default: MSMSulc]

                    input_bold = os.path.join(self.fmriprep_output_dir, 'fmriprep', subject, session, 'func',
                                              subject + '_' + session + '_' + 'task-' + self.task + '_' + run + '_' + 'space-T1w_desc-AROMAnonaggr_bold.nii.gz')
                    ciftify_subject_fmri_command = ' '.join(['ciftify_subject_fmri',
                                                   '--ciftify-work-dir', self.ciftify_workdir,
                                                   '--surf-reg', 'MSMSulc',
                                                   input_bold, subject, session + '_' + 'task-' + self.task + '_' + run])
                    try:
                        subprocess.check_call(ciftify_subject_fmri_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('CIFTIFY_SUBJECT_FMRI: Error happened in {}'.format(subject))

                    # fslmaths: Image calculator
                    # Usage: fslmaths [-dt <datatype>] <first_input> [operations and inputs] <output> [-odt <datatype>]
                    #     -mas: Use (following image>0) to mask current image

                    input_volume = os.path.join(self.ciftify_workdir, subject, 'MNINonLinear', 'Results',
                                               session + '_' + 'task-' + self.task + '_' + run,
                                               session + '_' + 'task-' + self.task + '_' + run + '.nii.gz')
                    brainmask = os.path.join(self.ciftify_workdir, subject, 'MNINonLinear', 'brainmask_fs.nii.gz')
                    output_volume = input_volume
                    maskbold_command = ' '.join(['fslmaths', input_volume,
                                                 '-mas', brainmask,
                                                 output_volume])
                    try:
                        subprocess.check_call(maskbold_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('MASKBOLD: Error happened in {}'.format(subject))

                    # Rename cifti files

                    first_level_dir = session + '_' + 'task-' + self.task + '_' + 'run-' + run
                    cifti_file_name = os.path.join(self.ciftify_workdir, subject, 'MNINonLinear', 'Results', first_level_dir,
                                                   first_level_dir + '_Atlas_s0.dtseries.nii')
                    new_cifti_file_name = os.path.join(self.ciftify_workdir, subject, 'MNINonLinear', 'Results', first_level_dir,
                                                     first_level_dir + '_Atlas.dtseries.nii')
                    os.rename(cifti_file_name, new_cifti_file_name)

    # Third step: task analysis (prepare EV files, prepare fsf files, first and second level analysis)

    def task_analysis(self):

        # Prepare EV files

        for subject in self.subject_list:
            session_list = os.listdir(os.path.join(self.raw_data_dir, subject))
            for session in session_list:
                with open(os.path.join(self.raw_data_dir, subject, session, 'tmp', 'run_info', self.task + '.rlf'), 'r') as f:
                    runs_id = f.read().splitlines()
                for run_id in runs_id:
                    ev_file = os.path.join(self.raw_data_dir, subject, session, 'func', subject + '_' + session + '_' + 'task-' + self.task + '_' + 'run-' + run_id + '_events.tsv')
                    ev_cond = pd.read_csv(ev_file, sep='\t')

                    labeldict = {1: 'toe', 2: 'ankle', 3: 'leftleg', 4: 'rightleg', 5: 'forearm', 6: 'upperarm', 7: 'wrist', 8: 'finger', 9: 'eye', 10: 'jaw', 11: 'lip', 12: 'tongue'}
                    assert (np.all(np.unique(ev_cond['trial_type']) == np.arange(len(labeldict) + 1))), "Conditions are not complete."
                    for lbl in labeldict.keys():
                        ev_cond_tmp = ev_cond[ev_cond['trial_type'] == lbl]
                        ev_cond_decomp = np.zeros((3, len(ev_cond_tmp)))
                        ev_cond_decomp[0, :] = np.array(ev_cond_tmp['onset'])
                        ev_cond_decomp[1, :] = np.array(ev_cond_tmp['duration'])
                        ev_cond_decomp[2, :] = np.ones(len(ev_cond_tmp))
                        ev_cond_decomp = ev_cond_decomp.T
                        outpath = os.path.join(self.ciftify_workdir, subject, 'MNINonLinear', 'Results', session + '_' + 'task-' + self.task + '_' + 'run-' + run_id, 'EVs')
                        if not os.path.isdir(outpath):
                            subprocess.call('mkdir ' + outpath, shell=True)
                        np.savetxt(os.path.join(outpath, labeldict[lbl] + '.txt'), ev_cond_decomp, fmt='%-6.1f', delimiter='\t', newline='\n')

        # Prepare fsf files of first and second level

        fsf1_template = os.path.join(self.fsf_template_dir, 'level1.fsf')
        fsf2_template = os.path.join(self.fsf_template_dir, 'level2.fsf')
        for subject in self.subject_list:
            results_dir = os.path.join(self.ciftify_workdir, subject, 'MNINonLinear', 'Results')
            session_list = os.listdir(os.path.join(self.raw_data_dir, subject))
            for session in session_list:
                with open(os.path.join(self.raw_data_dir, subject, session, 'tmp', 'run_info', self.task + '.rlf'), 'r') as f:
                    runs_id = f.read().splitlines()
                run_list = ['run-' + run for run in runs_id]
                for run in run_list:

                    # Copy the first level fsf files to results directory

                    fsf1_outdir = os.path.join(results_dir, session + '_' + 'task-' + self.task + '_' + run)
                    fsf1_filepath = os.path.join(fsf1_outdir,
                                                 session + '_' + 'task-' + self.task + '_' + run + '_hp200_s4_level1.fsf')
                    if not os.path.exists(fsf1_outdir):
                        os.makedirs(fsf1_outdir)
                    cp_fsf1_command = ' '.join(['cp', fsf1_template, fsf1_filepath])
                    try:
                        subprocess.call(cp_fsf1_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('CPFSF1: Error happened in {}'.format(subject))

                    # Modify first level fsf files

                    sedfsf1_command = " ".join(['sed', '-i', '\'s#{0}#{1}#g\''.format('run-a', run), fsf1_filepath])
                    try:
                        subprocess.call(sedfsf1_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('SEDFSF1: Error happened in {}'.format(subject))


                # Copy the second level fsf file to results directory

                fsf2_outdir = os.path.join(results_dir, session + '_' + 'task-' + self.task)
                fsf2_filepath = os.path.join(fsf2_outdir, session + '_' + 'task-' + self.task + '_hp200_s4_level2.fsf')
                if not os.path.exists(fsf2_outdir):
                    os.makedirs(fsf2_outdir)
                cp_fsf2_command = ' '.join(['cp', fsf2_template, fsf2_filepath])
                try:
                    subprocess.call(cp_fsf2_command, shell=True)
                except subprocess.CalledProcessError:
                    raise Exception('CPFSF2: Error happened in {}'.format(subject))

                # Modify the second level fsf file

                letter_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
                for run in run_list:
                    letter_id = 0
                    sedfsf2_command = " ".join(['sed', '-i', '\'s#{0}#{1}#g\''.format('run-' + letter_list[letter_id], run), fsf2_filepath])
                    try:
                        subprocess.call(sedfsf2_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('SEDFSF2: Error happened in {}'.format(subject))
                    letter_id += 1

        # Analysis based on general linear model

        lowres = '32'
        grayres = '2'
        origFWHM = '2'
        confound = 'NONE'
        finalFWHM = '4'
        tempfilter = '200'
        vba = 'NO'
        regname = 'NONE'
        parcellation = 'NONE'
        parcefile = 'NONE'
        for subject in self.subject_list:
            session_list = os.listdir(os.path.join(self.raw_data_dir, subject))
            level2_task_list = []
            for session in session_list:
                level2_task_list.append(session + '_' + 'task-' + self.task)
                with open(os.path.join(self.raw_data_dir, subject, session, 'tmp', 'run_info', self.task + '.rlf'), 'r') as f:
                    runs_list = f.read().splitlines()
                level1_tasks_list = []
                for run in runs_list:
                    level1_tasks_list.append(session + '_' + 'task-' + self.task + '_' + 'run-' + run)
                level1_tasks = '@'.join(level1_tasks_list)
                level1_fsfs = level1_tasks
            level2_task = '@'.join(level2_task_list)
            level2_fsf = level2_task

            analysis_command = ' '.join(['${HCPPIPEDIR}/TaskfMRIAnalysis/TaskfMRIAnalysis.sh',
                                        '--path=' + self.ciftify_workdir,
                                        '--subject=' + subject,
                                        '--lvl1tasks=' + level1_tasks,
                                        '--lvl1fsfs=' + level1_fsfs,
                                        '--lvl2task=' + level2_task,
                                        '--lvl2fsf=' + level2_fsf,
                                        '--lowresmesh=' + lowres,
                                        '--grayordinatesres=' + grayres,
                                        '--origsmoothingFWHM=' + origFWHM,
                                        '--confound=' + confound,
                                        '--finalsmoothingFWHM=' + finalFWHM,
                                        '--temporalfilter=' + tempfilter,
                                        '--vba=' + vba,
                                        '--regname=' + regname,
                                        '--parcellation=' + parcellation,
                                        '--parcellationfile=' + parcefile])
            try:
                subprocess.check_call(analysis_command, shell=True)
            except subprocess.CalledProcessError:
                raise Exception('TASKANALYSIS: Error happened in {}'.format(subject))









