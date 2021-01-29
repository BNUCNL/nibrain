HCP pipeline

HCP_pipeline.py
    Intro: Integrating fmriprep, cifify and taskanalysis, all functions can be used together or individually.
    Requirements of input: BIDS dataset, standard file name(i.e., subjectId_sessionId_task-taskName_runId_fileType.nii.gz').
    Requirements of environment: HCPpipelines v4.2.0 (https://github.com/Washington-University/HCPpipelines),
                                 FSL v6.0.2 (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/),
                                 Connectome Workbench v1.4.2 (https://www.humanconnectome.org/software/connectome-workbench)
                                 fmriprep v20.0.2 (https://fmriprep.org/en/stable)
    Usage: Two examples are given in the scripts folder, including complete use of HCPpiplines and task_analysis function use alone
