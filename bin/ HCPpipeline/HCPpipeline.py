"""
    HCPpipeline.py
"""



# Three steps: fmriprep, ciftify, and HCP Pipeline

class pipeline(object):

    def __init__(self, raw_data_dir, fmriprep_output_dir, fmriprep_workdir, ciftify_workdir, fsf_template_dir, subject_list, task_name):
        self.raw_data_dir = raw_data_dir
        self.fmriprep_output_dir = fmriprep_output_dir
        self.fmriprep_workdir = fmriprep_workdir
        self.ciftify_workdir = ciftify_workdir
        self.fsf_template_dir = fsf_template_dir
        self.subject_list = subject_list
        self.task_name = task_name


    # First part: Pre-FreeSurfer processing

    def fmriprep_command(self):
        


