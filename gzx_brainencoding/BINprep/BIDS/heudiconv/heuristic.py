import os
def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes
def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    allowed template fields - follow python string module:
    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    t1w = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_run-{item:02d}_T1w')
    
    func_rest = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-rest_run-{item:02d}_bold')
    func_object =  create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-object_run-{item:02d}_bold')
    func_fixcolor =  create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-fixcolor_run-{item:02d}_bold')
    func_retinotopy = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-retinotopy_run-{item:02d}_bold')
    func_category = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-category_run-{item:02d}_bold')
    func_category154 = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-category154_run-{item:02d}_bold')

    fmap_mag =  create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_run-{item:02d}_magnitude')
    fmap_phase = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_run-{item:02d}_phasediff')
    

    info = {t1w: [], func_rest: [], func_object: [], func_fixcolor: [], func_category: [], 
            func_retinotopy: [], func_category154: [], fmap_mag: [], fmap_phase: []}
    
    for idx, s in enumerate(seqinfo):
        if (s.dim4 == 256) and ('bold_100x100_s3_2x2x2_static_256' in s.protocol_name):
            info[func_object].append(s.series_id)
        if ('t1_mprage_sag_1x1x1_p2_64ch' in s.protocol_name):
            info[t1w].append(s.series_id)
        if (s.dim4 == 240) and ('bold_100x100_s3_2x2x2_RS_240' in s.protocol_name):
            info[func_rest].append(s.series_id)
        if (s.dim4 == 241) and ('bold_100x100_s3_2x2x2_static_test_241' in s.protocol_name):
            info[func_fixcolor].append(s.series_id)
        if (s.dim4 == 150) and ('retino_150' in s.protocol_name):
            info[func_retinotopy].append(s.series_id)
        if (s.dim4 == 154) and ('bold_100x100_s3_2x2x2_cat_154' in s.protocol_name):
            info[func_category154].append(s.series_id)
        if (s.dim4 == 150) and ('bold_100x100_s3_2x2x2_cat_150' in s.protocol_name):
            info[func_category].append(s.series_id)
        if (s.dim3 == 144) and ('field_mapping_2x2x2' in s.protocol_name):
            info[fmap_mag].append(s.series_id)
        if (s.dim3 == 72) and ('field_mapping_2x2x2' in s.protocol_name):
            info[fmap_phase].append(s.series_id)
    return info
