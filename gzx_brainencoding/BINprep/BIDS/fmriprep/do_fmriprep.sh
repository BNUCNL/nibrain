Home=/nfs/m1/BrainImageNet/fMRIData
bids_fold=$Home/rawdata
out_dir=$Home/derivatives
work_dir=$Home/workdir
license_file=/usr/local/neurosoft/freesurfer/license.txt

fmriprep-docker $bids_fold $out_dir participant \
    --participant-label core02 \
    --fs-license-file $license_file \
    --output-space anat MNI152NLin2009cAsym:res-2 fsnative fsLR \
    --use-aroma \
    --cifti-output 91k \
    -w $work_dir
fmriprep-docker $bids_fold $out_dir participant --participant-label core02 --fs-license-file $license_file --output-spaces anat MNI152NLin2009cAsym:res-2 fsnative fsLR --use-aroma --cifti-output 91k -w $work_dir
