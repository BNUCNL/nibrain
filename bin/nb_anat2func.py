# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine

def pickfile(files, order):
    """
    Pick which tissue for processing, 0 for csf, and 2 for White Matter.
    """
    if isinstance(files, list):
        return files[order]
    else:
        return files

def anat2func(name='anat2func'):
    """
    Get anat2func transform matrix, csf mask and wm mask in functional space.
    """

    anat2func = pe.Workflow(name=name)

    # Set up a node to define all inputs and outputs required for the 
    # preprocessing workflow

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['example_func',
                                                                 'mask',
                                                                 'brain',
                                                                 'highres2standard']),
                        name='inputspec')
    outputnode = pe.Node(interface=util.IdentityInterface(fields=['anat2funcmtx',
                                                                  'csfmask',
                                                                  'wmmask',
                                                                  'gmmask']),
                         name='outputspec')
  
    # Mask the example_func with the extracted mask
    
    maskfunc = pe.MapNode(interface=fsl.ImageMaths(suffix='_bet',
                                                   op_string='-mas'),
                          iterfield=['in_file','in_file2'],
                          name = 'maskfunc')

    coregister = pe.MapNode(interface=fsl.FLIRT(dof=6,cost='corratio',interp='trilinear'),
                            iterfield=['reference'],
                            name = 'coregister')

    # Convert highres2standard matrix to standard2highres matrix

    xfmconvert = pe.Node(interface=fsl.ConvertXFM(invert_xfm=True),
                         name='xfmconvert')


    # Create mask for three tissue types.

    tissuemasks = pe.Node(interface=fsl.FAST(no_pve=True,segments=True,use_priors=True),
                          name = 'segment')

    # Transform CSF mask to func space.
    
    csf2func = pe.MapNode(interface=fsl.ApplyXfm(apply_xfm=True),
                          iterfield=['reference','in_matrix_file'],
                          name = 'csf2func')

    
    # Threshold CSF segmentation mask from  .90 to 1

    threshcsf = pe.MapNode(interface = fsl.ImageMaths(op_string = ' -thr .90 -uthr 1 -bin '),
                           iterfield=['in_file'],                           
                           name = 'threshcsf')

    
    # Transform WM mask to func
    
    wm2func = pe.MapNode(interface=fsl.ApplyXfm(apply_xfm=True),
                         iterfield=['reference','in_matrix_file'],
                         name = 'wm2func')

    
    # Threshold WM segmentation mask from  .90 to 1
    
    threshwm = pe.MapNode(interface = fsl.ImageMaths(op_string = ' -thr .90 -uthr 1 -bin '),
                          iterfield=['in_file'],                       
                          name = 'threshwm')

    # Transform GM mask to func
    
    gm2func = pe.MapNode(interface=fsl.ApplyXfm(apply_xfm=True),
                         iterfield=['reference','in_matrix_file'],
                         name = 'gm2func')

    
    # Threshold WM segmentation mask from  .50 to 1
    
    threshgm = pe.MapNode(interface = fsl.ImageMaths(op_string = ' -thr .50 -uthr 1 -bin '),
                          iterfield=['in_file'],                       
                          name = 'threshgm')		
    
    # To get CSF and WM mask in functional space
    
    anat2func.connect([(inputnode, coregister,[('brain','in_file')]),
                       (inputnode, maskfunc,[('example_func', 'in_file')]),
                       (inputnode, maskfunc, [('mask', 'in_file2')]),
                       (inputnode, xfmconvert, [('highres2standard', 'in_file')]),                       
                       (xfmconvert, tissuemasks, [('out_file', 'init_transform')]),                       
                       (maskfunc, coregister,[('out_file','reference')]),
                       (coregister, outputnode, [('out_matrix_file', 'anat2funcmtx')]),
                       (inputnode, tissuemasks,[('brain','in_files')]),
                       (tissuemasks, csf2func,[(('tissue_class_files',pickfile,0),'in_file')]),
                       (maskfunc, csf2func,[('out_file','reference')]),
                       (coregister,csf2func,[('out_matrix_file','in_matrix_file')]),
                       (csf2func,threshcsf,[('out_file','in_file')]),
                       (tissuemasks, wm2func,[(('tissue_class_files',pickfile,2),'in_file')]),
                       (maskfunc, wm2func,[('out_file','reference')]),
                       (coregister, wm2func,[('out_matrix_file','in_matrix_file')]),
                       (wm2func,threshwm,[('out_file','in_file')]),
                       (tissuemasks, gm2func,[(('tissue_class_files',pickfile,1),'in_file')]),
                       (maskfunc, gm2func,[('out_file','reference')]),
                       (coregister, gm2func,[('out_matrix_file','in_matrix_file')]),
                       (gm2func,threshgm,[('out_file','in_file')]),
                       (threshcsf,outputnode,[('out_file','csfmask')]),
                       (threshwm,outputnode,[('out_file','wmmask')]),
                       (threshgm,outputnode,[('out_file','gmmask')]),
                       ])
    return anat2func