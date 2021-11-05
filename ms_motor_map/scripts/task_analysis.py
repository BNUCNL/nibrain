#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 04:46:16 2021

@author: MaSai
"""

import os
import subprocess

class task_analysis(object):

    def __init__(self, project_dir, subject_list, fsf_template_dir):
        self.project_dir = project_dir
        self.subject_list = subject_list
        self.fsf_template_dir = fsf_template_dir

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


    def prepare_fsf(self):
        level1_fsf_file = os.path.join(self.fsf_template_dir, 'level1.fsf')
        level2_fsf_file = os.path.join(self.fsf_template_dir, 'level2.fsf')
        for subject_id in self.subject_list:
            results_dir = os.path.join(self.project_dir, 'data', 'bold', 'derivatives', 'ciftify', subject_id, 'MNINonLinear', 'Results')
            with open(os.path.join(self.raw_data_dir, subject_id, 'ses-01', 'tmp', 'run_info', 'motor.rlf'), 'r') as f:
                runs_id = f.read().splitlines()
            level2_fsf_file_outdir = os.path.join(results_dir, 'ses-01_task-motor')
            cpfsf2_command = ' '.join(['cp', level2_fsf_file, os.path.join(level2_fsf_file_outdir, 'ses-01_task-motor_hp200_s4_level2.fsf')])
            subprocess.check_call(cpfsf2_command, shell=True)
            self.customize_fsf2(os.path.join(level2_fsf_file_outdir, 'ses-01_task-motor_hp200_s4_level2.fsf'), runs_id)
            for run_id in runs_id:
                level1_fsf_file_outdir = os.path.join(results_dir, 'ses-01_task-motor_run-' + run_id)
                if not os.path.exists(level1_fsf_file_outdir):
                    os.mkdir(level1_fsf_file_outdir)
                cpfsf1_command = ' '.join(['cp', level1_fsf_file, os.path.join(level1_fsf_file_outdir, 'ses-01_task-motor_run-' + run_id + '_hp200_s4_level1.fsf')])
                subprocess.call(cpfsf1_command, shell=True)
                self.customize_fsf1(os.path.join(level1_fsf_file_outdir, 'ses-01_task-motor_run-' + run_id + '_hp200_s4_level1.fsf'), 'run-' + run_id)

    def customize_fsf1(self, fsf_file_path, runid, from_runid='run-a'):
        sed_level1_fsf_command = " ".join(['sed', '-i', '\'s#{0}#{1}#g\''.format(from_runid, runid), fsf_file_path])
        subprocess.check_call(sed_level1_fsf_command, shell=True)

    def customize_fsf2(self, fsf_file_path, runid_list):
        runid_list = ['run-' + rl for rl in runid_list]
        sedfsf2_command1 = " ".join(['sed', '-i', '\'s#{0}#{1}#g\''.format('run-a', runid_list[0]), fsf_file_path])
        subprocess.call(sedfsf2_command1, shell=True)
        sedfsf2_command2 = " ".join(['sed', '-i', '\'s#{0}#{1}#g\''.format('run-b', runid_list[1]), fsf_file_path])
        subprocess.call(sedfsf2_command2, shell=True)
        sedfsf2_command3 = " ".join(['sed', '-i', '\'s#{0}#{1}#g\''.format('run-c', runid_list[2]), fsf_file_path])
        subprocess.call(sedfsf2_command3, shell=True)
        sedfsf2_command4 = " ".join(['sed', '-i', '\'s#{0}#{1}#g\''.format('run-d', runid_list[3]), fsf_file_path])
        subprocess.call(sedfsf2_command4, shell=True)
        sedfsf2_command5 = " ".join(['sed', '-i', '\'s#{0}#{1}#g\''.format('run-e', runid_list[4]), fsf_file_path])
        subprocess.call(sedfsf2_command5, shell=True)
        sedfsf2_command6 = " ".join(['sed', '-i', '\'s#{0}#{1}#g\''.format('run-f', runid_list[5]), fsf_file_path])
        subprocess.call(sedfsf2_command6, shell=True)

    def analysis(self):
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
            with open(os.path.join(self.raw_data_dir, subject, 'ses-01', 'tmp', 'run_info', 'motor.rlf'), 'r') as f:
                runs_id = f.read().splitlines()
            lvl1tasks_list = []
            for run_id in runs_id:
                lvl1tasks_list.append('ses-01_task-motor_run-' + run_id)
            level1_tasks = '@'.join(lvl1tasks_list)
            level1_fsfs = level1_tasks
            level2_tasks = 'ses-01_task-motor'
            level2_fsf = level2_tasks
            analysis_command = ' '.join(['${HCPPIPEDIR}/TaskfMRIAnalysis/TaskfMRIAnalysis.sh',
                                        '--path=' + self.ciftify_dir,
                                        '--subject=' + subject,
                                        '--lvl1tasks=' + level1_tasks,
                                        '--lvl1fsfs=' + level1_fsfs,
                                        '--lvl2task=' + level2_tasks,
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
            subprocess.check_call(analysis_command, shell=True)



if __name__ == '__main__':
    project_dir = '/nfs/z1/zhenlab/MotorMap'
    subject_list = ['sub-01']
    fsf_template_dir = '/nfs/z1/userhome/MaSai/workingdir/Motor_project/data/fsf_template'

    task_analysis = task_analysis(project_dir, subject_list, fsf_template_dir)
    # task_analysis.prepare_fsf()
    # task_analysis.analysis()