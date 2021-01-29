"""
    motormapping pipeline
"""

# Data processing pipeline

import subprocess
import os
import numpy as np
import pandas as pd


# Three part, fmriprep, ciftify, and HCP Pipeline
class Pipeline(object):
    def __init__(self, data_inpath, data_outpath, prep_workdir, ciftify_workdir, fsf_dir, subject_id, task='motor'):
        self.data_inpath = data_inpath
        self.data_outpath = data_outpath
        self.prep_workdir = prep_workdir
        self.subject_id = subject_id
        self.task = task
        self.ciftify_workdir = ciftify_workdir
        self.fsf_dir = fsf_dir

    def _fmriprep(self, subj_id, workdir):
        fmriprep_command = ' '.join(['fmriprep', self.data_inpath,
                                     self.data_outpath, 'participant',
                                     '-w', workdir,
                                     '--participant_label', subj_id,
                                     '--use-aroma', '--use-syn-sdc',
                                     '--output-space', 'T1w'])
        return fmriprep_command

    def run_fmriprep(self):
        for subj_id in self.subject_id:
            fmriprep_command = self._fmriprep(subj_id, self.prep_workdir)
            try:
                subprocess.check_call(fmriprep_command, shell=True)
            except subprocess.CalledProcessError:
                raise Exception('FMRIPREP: Error happened in subject {}'.format(subj_id))

    def _fsl_regfilt(self, subj_id, ses_id, run_id):
        input_data = os.path.join(self.data_outpath, 'fmriprep', subj_id, ses_id, 'func',
                                  subj_id + '_' + ses_id + '_' + 'task-' + self.task + '_' + run_id + '_space-T1w_desc-preproc_bold.nii.gz')
        aromaIC = os.path.join(self.data_outpath, 'fmriprep', subj_id, ses_id, 'func',
                               subj_id + '_' + ses_id + '_' + 'task-' + self.task + '_' + run_id + '_AROMAnoiseICs.csv')
        aromaMix = os.path.join(self.data_outpath, 'fmriprep', subj_id, ses_id, 'func',
                                subj_id + '_' + ses_id + '_' + 'task-' + self.task + '_' + run_id + '_desc-MELODIC_mixing.tsv')
        output_data = os.path.join(self.data_outpath, 'fmriprep', subj_id, ses_id, 'func',
                                   subj_id + '_' + ses_id + '_' + 'task-' + self.task + '_' + run_id + '_space-T1w_desc-AROMAnonaggr_bold.nii.gz')

        fslreg_command = ' '.join(['fsl_regfilt', '-i', input_data,
                                   '-f', '$(cat {})'.format(aromaIC),
                                   '-d', aromaMix,
                                   '-o', output_data])
        return fslreg_command

    def run_fslreg(self):
        """
        """
        for subj_id in self.subject_id:
            # Load ses_id
            session_id = os.listdir(os.path.join(self.data_inpath, subj_id))
            for ses_id in session_id:
                # Load run_id
                with open(os.path.join(self.data_inpath, subj_id,
                                       ses_id, 'tmp', 'run_info',
                                       self.task + '.rlf'), 'r') as f:
                    runs_id = f.read().splitlines()
                for run_id in runs_id:
                    fslreg_command = self._fsl_regfilt(subj_id, ses_id, 'run-' + run_id)
                    try:
                        subprocess.check_call(fslreg_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('FSLREG: Error happened in subject {}'.format(subj_id))

    def _ciftify_reconall(self, subj_id):
        fs_subjects_dir = os.path.join(self.data_outpath, 'freesurfer')
        cifrecon_command = ' '.join(['ciftify_recon_all',
                                     '--resample-to-T1w32k',
                                     '--surf-reg', 'MSMSulc',
                                     '--ciftify-work-dir', self.ciftify_workdir,
                                     '--fs-subjects-dir', fs_subjects_dir,
                                     subj_id])
        return cifrecon_command

    def _ciftify_subjfmri(self, subj_id, ses_id, run_id):
        input_bold = os.path.join(self.data_outpath, 'fmriprep',
                                  subj_id, ses_id, 'func',
                                  subj_id + '_' + ses_id + '_' + 'task-' + self.task + '_' + run_id + '_' + 'space-T1w_desc-AROMAnonaggr_bold.nii.gz')
        cifsubfmri_command = ' '.join(['ciftify_subject_fmri',
                                       '--ciftify-work-dir', self.ciftify_workdir,
                                       '--surf-reg', 'MSMSulc',
                                       input_bold, subj_id,
                                       ses_id + '_' + 'task-' + self.task + '_' + run_id])
        return cifsubfmri_command

    def _mask_bold(self, subj_id, ses_id, run_id):
        """
        """
        inputvolume = os.path.join(self.ciftify_workdir,
                                   subj_id, 'MNINonLinear',
                                   'Results',
                                   ses_id + '_' + 'task-' + self.task + '_' + run_id,
                                   ses_id + '_' + 'task-' + self.task + '_' + run_id + '.nii.gz')
        maskbold_command = ' '.join(['fslmaths', inputvolume,
                                     '-mas',
                                     os.path.join(self.ciftify_workdir, subj_id, 'MNINonLinear', 'brainmask_fs.nii.gz'),
                                     inputvolume])
        return maskbold_command

    def run_ciftify(self):
        for subj_id in self.subject_id:
            # ciftify_recon_all
            cifrecon_command = self._ciftify_reconall(subj_id)
            try:
                subprocess.check_call(cifrecon_command, shell=True)
            except subprocess.CalledProcessError:
                raise Exception('CIFRECONALL: Error happened in subject {}'.format(subj_id))
            # subject_fmri
            # Load ses_id
            session_id = os.listdir(os.path.join(self.data_inpath, subj_id))
            for ses_id in session_id:
                # Load run_id
                with open(os.path.join(self.data_inpath, subj_id,
                                       ses_id, 'tmp', 'run_info',
                                       self.task + '.rlf'), 'r') as f:
                    runs_id = f.read().splitlines()
                for run_id in runs_id:
                    # ciftify subject fmri
                    cifsubfmri_command = self._ciftify_subjfmri(subj_id, ses_id, 'run-' + run_id)
                    try:
                        subprocess.check_call(cifsubfmri_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('CIFSUBJFMRI: Error happened in subject {}'.format(subj_id))
                    # Mask bold signal
                    maskbold_command = self._mask_bold(subj_id, ses_id, 'run-' + run_id)
                    try:
                        subprocess.check_call(maskbold_command, shell=True)
                    except subprocess.CalledProcessError:
                        raise Exception('MASKBOLD: Error happened in subject {}'.format(subj_id))
                    # Rename file
                    levelonefname = ses_id + '_' + 'task-' + self.task + '_' + 'run-' + run_id
                    cifti_file_name = os.path.join(self.ciftify_workdir, subj_id,
                                                   'MNINonLinear', 'Results',
                                                   levelonefname,
                                                   levelonefname + '_Atlas_s0.dtseries.nii')
                    cifti_file_rename = os.path.join(self.ciftify_workdir, subj_id,
                                                     'MNINonLinear', 'Results',
                                                     levelonefname,
                                                     levelonefname + '_Atlas.dtseries.nii')
                    rename_command = ' '.join(['mv', cifti_file_name, cifti_file_rename])
                    if os.path.isfile(cifti_file_name):
                        try:
                            subprocess.check_call(rename_command, shell=True)
                        except subprocess.CalledProcessError:
                            raise Exception('RENAME: Error happened in subject {}'.format(subj_id))

    def _decompose_ev(self, subj_id, ses_id, run_id, ev_cond):
        """
        -------------------
        Decompose paradigm into different conditions
        we promise:
        0: fixation
        1: toe
        2: ankle
        3: leftleg
        4: rightleg
        5: forearm
        6: upperarm
        7: wrist
        8: finger
        9: eye
        10: jaw
        11: lip
        12: tongue

        Parameters:
        -----------
        ev_cond[pd.DataFrame]: experimental variable paradigm
        """
        labeldict = {1: 'toe', 2: 'ankle', 3: 'leftleg', 4: 'rightleg', 5: 'forearm', 6: 'upperarm', 7: 'wrist',
                     8: 'finger', 9: 'eye', 10: 'jaw', 11: 'lip', 12: 'tongue'}
        assert (
            np.all(np.unique(ev_cond['trial_type']) == np.arange(len(labeldict) + 1))), "Conditions are not complete."
        for lbl in labeldict.keys():
            ev_cond_tmp = ev_cond[ev_cond['trial_type'] == lbl]
            ev_cond_decomp = np.zeros((3, len(ev_cond_tmp)))
            ev_cond_decomp[0, :] = np.array(ev_cond_tmp['onset'])
            ev_cond_decomp[1, :] = np.array(ev_cond_tmp['duration'])
            ev_cond_decomp[2, :] = np.ones(len(ev_cond_tmp))
            ev_cond_decomp = ev_cond_decomp.T
            outpath = os.path.join(self.ciftify_workdir, subj_id, 'MNINonLinear', 'Results',
                                   ses_id + '_' + 'task-' + self.task + '_' + 'run-' + run_id, 'EVs')
            if not os.path.isdir(outpath):
                subprocess.call('mkdir ' + outpath, shell=True)
            np.savetxt(os.path.join(outpath, labeldict[lbl] + '.txt'), ev_cond_decomp, fmt='%-6.1f', delimiter='\t',
                       newline='\n')

    def prepare_EVs(self):
        """
        """
        for subj_id in self.subject_id:
            # Load ses_id
            session_id = os.listdir(os.path.join(self.data_inpath, subj_id))
            for ses_id in session_id:
                # Load run_id
                with open(os.path.join(self.data_inpath, subj_id,
                                       ses_id, 'tmp', 'run_info',
                                       self.task + '.rlf'), 'r') as f:
                    runs_id = f.read().splitlines()
                for run_id in runs_id:
                    ev_cond = pd.read_csv(os.path.join(self.data_inpath, subj_id, ses_id, 'func',
                                                       subj_id + '_' + ses_id + '_' + 'task-' + self.task + '_' + 'run-' + run_id + '_events.tsv'),
                                          sep='\t')
                    self._decompose_ev(subj_id, ses_id, run_id, ev_cond)

    def _modify_fsf1(self, fsfpath, to_runid, from_runid='run-a'):
        """
        """
        sedfsf1_command = " ".join(['sed', '-i',
                                    '\'s#{0}#{1}#g\''.format(from_runid, to_runid), fsfpath])
        subprocess.call(sedfsf1_command, shell=True)

    def _modify_fsf2(self, fsfpath, runid_list):
        runid_list = ['run-' + rl for rl in runid_list]
        if len(runid_list) == 6:
            sedfsf2_command1 = " ".join(['sed', '-i',
                                         '\'s#{0}#{1}#g\''.format('run-a', runid_list[0]), fsfpath])
            subprocess.call(sedfsf2_command1, shell=True)
            sedfsf2_command2 = " ".join(['sed', '-i',
                                         '\'s#{0}#{1}#g\''.format('run-b', runid_list[1]), fsfpath])
            subprocess.call(sedfsf2_command2, shell=True)
            sedfsf2_command3 = " ".join(['sed', '-i',
                                         '\'s#{0}#{1}#g\''.format('run-c', runid_list[2]), fsfpath])
            subprocess.call(sedfsf2_command3, shell=True)
            sedfsf2_command4 = " ".join(['sed', '-i',
                                         '\'s#{0}#{1}#g\''.format('run-d', runid_list[3]), fsfpath])
            subprocess.call(sedfsf2_command4, shell=True)
            sedfsf2_command5 = " ".join(['sed', '-i',
                                         '\'s#{0}#{1}#g\''.format('run-e', runid_list[4]), fsfpath])
            subprocess.call(sedfsf2_command5, shell=True)
            sedfsf2_command6 = " ".join(['sed', '-i',
                                         '\'s#{0}#{1}#g\''.format('run-f', runid_list[5]), fsfpath])
            subprocess.call(sedfsf2_command6, shell=True)

    def prepare_fsf(self):
        """
        """
        fsflevel1_indir = os.path.join(self.fsf_dir, 'level1.fsf')
        fsflevel2_indir = os.path.join(self.fsf_dir, 'level2.fsf')
        for subj_id in self.subject_id:
            result_dir = os.path.join(self.ciftify_workdir, subj_id,
                                      'MNINonLinear', 'Results')
            # Load ses_id
            session_id = os.listdir(os.path.join(self.data_inpath, subj_id))
            for ses_id in session_id:
                with open(os.path.join(self.data_inpath, subj_id,
                                       ses_id, 'tmp', 'run_info',
                                       self.task + '.rlf'), 'r') as f:
                    runs_id = f.read().splitlines()
                if len(runs_id) != 6:
                    continue

                fsflevel2_outdir = os.path.join(result_dir,
                                                ses_id + '_' + 'task-' + self.task)
                if not os.path.isdir(fsflevel2_outdir):
                    os.makedirs(fsflevel2_outdir)
                cpfsf2_command = ' '.join(['cp', fsflevel2_indir, os.path.join(fsflevel2_outdir,
                                                                               ses_id + '_' + 'task-' + self.task + '_hp200_s4_level2.fsf')])
                self.cpfsf2_commmand = cpfsf2_command
                subprocess.call(cpfsf2_command, shell=True)
                self._modify_fsf2(
                    os.path.join(fsflevel2_outdir, ses_id + '_' + 'task-' + self.task + '_hp200_s4_level2.fsf'),
                    runs_id)
                for run_id in runs_id:
                    fsflevel1_outdir = os.path.join(result_dir,
                                                    ses_id + '_' + 'task-' + self.task + '_' + 'run-' + run_id)
                    if not os.path.isdir(fsflevel1_outdir):
                        os.makedirs(fsflevel1_outdir)
                    cpfsf1_command = ' '.join(['cp', fsflevel1_indir, os.path.join(fsflevel1_outdir,
                                                                                   ses_id + '_' + 'task-' + self.task + '_' + 'run-' + run_id + '_hp200_s4_level1.fsf')])
                    subprocess.call(cpfsf1_command, shell=True)
                    self._modify_fsf1(os.path.join(fsflevel1_outdir,
                                                   ses_id + '_' + 'task-' + self.task + '_' + 'run-' + run_id + '_hp200_s4_level1.fsf'),
                                      'run-' + run_id)

    def run_taskglm(self):
        """
        """
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
        for subj_id in self.subject_id:
            # Load ses_id
            session_id = os.listdir(os.path.join(self.data_inpath, subj_id))
            lvl2task_list = []
            for ses_id in session_id:
                lvl2task_list.append(ses_id + '_' + 'task-' + self.task)
                # Load run_id
                with open(os.path.join(self.data_inpath, subj_id,
                                       ses_id, 'tmp', 'run_info',
                                       self.task + '.rlf'), 'r') as f:
                    runs_id = f.read().splitlines()

                lvl1tasks_list = []
                for run_id in runs_id:
                    # prepare lvl1tasks
                    lvl1tasks_list.append(ses_id + '_' + 'task-' + self.task + '_' + 'run-' + run_id)
                lvl1tasks = '@'.join(lvl1tasks_list)
                lvl1fsfs = lvl1tasks
            lvl2task = '@'.join(lvl2task_list)
            lvl2fsf = lvl2task

            taskglm_command = ' '.join(['TaskfMRIAnalysis.sh',
                                        '--path=' + self.ciftify_workdir,
                                        '--subject=' + subj_id,
                                        '--lvl1tasks=' + lvl1tasks,
                                        '--lvl1fsfs=' + lvl1fsfs,
                                        '--lvl2task=' + lvl2task,
                                        '--lvl2fsf=' + lvl2fsf,
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
            # self.taskglm_command = taskglm_command
            try:
                subprocess.check_call(taskglm_command, shell=True)
            except subprocess.CalledProcessError:
                raise Exception('TASKGLM: Error happened in subject {}'.format(subj_id))

    def run_pipeline(self):
        self.run_fmriprep()
        self.run_fslreg()
        self.run_ciftify()
        self.prepare_EVs()
        self.prepare_fsf()
        self.run_taskglm()


if __name__ == '__main__':
    data_inpath = '/nfs/e4/function_guided_resection/MotorMapping'
    data_outpath = '/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/fmriprep'
    ciftify_workdir = '/nfs/e4/function_guided_resection/MotorMapping/derivatives/surface/ciftify'
    fsf_dir = os.path.join(data_inpath, 'fsf_template')

    prep_workdir = os.path.join('/nfs/e4/function_guided_resection', 'fmriprep_tmp', 'work-M01-M58-M68')

    participants_info = pd.read_csv(os.path.join(data_inpath, 'participants.tsv'), sep='\t')
    subject_id = participants_info['participant_id'].values
    # run the first 8 participants in this terminal
    # subject_id = subject_id[4+30:4+40]
    # subject_id = subject_id[49:54]
    subject_id = ['sub-M01', 'sub-M58', 'sub-M59', 'sub-M60', 'sub-M61', 'sub-M62', 'sub-M63', 'sub-M65', 'sub-M66', 'sub-M67', 'sub-M68']


    pip_cls = Pipeline(data_inpath, data_outpath, prep_workdir, ciftify_workdir, fsf_dir, subject_id)
    pip_cls.run_pipeline()
    # pip_cls.run_fmriprep()
    # pip_cls.prepare_fsf()
    # pip_cls.run_taskglm()





