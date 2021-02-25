export CIFTIFY_WORKDIR=/nfs/m1/BrainImageNet/fMRIData/ciftify
ciftify_subject_fmri --surf-reg MSMSulc /nfs/m1/BrainImageNet/fMRIData/derivatives/fmriprep/sub-core02/ses-COCO/func/*retinotopy*run-1*space-T1w*preproc_bold.nii.gz sub-core02 Retinotopy-COCO-1

export nii_files=/nfs/m1/BrainImageNet/fMRIData/derivatives/fmriprep/sub-core02/ses-COCO/func/*retinotopy*space-T1w*preproc_bold.nii.gz
