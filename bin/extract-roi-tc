#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Used to do resting functional connectivity analysis.
Author: Wang Xu @ BNU, 2011-10-10
Last modified by Wang Xu, 2014-7

"""
import os
import re
import sys
import logging
import argparse
#import subprocess

def main():
    parser = argparse.ArgumentParser(prog='rsfc', 
                                     prefix_chars='-',
                                     description='Do resting functional' + \
                                                 ' connectivity analysis')
    parser.add_argument('-method',
                        help='Analysis method: must be sba or roi',
                        dest='method',
                        choices=['sba','roi'],
                        required=True)
    parser.add_argument('-mask', 
                        help='Input the seed mask(sba) or atlas mask file',
                        dest='mask',
                        required=True)
    grpsess = parser.add_mutually_exclusive_group(required=True)
    grpsess.add_argument('-sess',
                        help='Input the sessid',
                        metavar='sessid',
                        dest='sess')
    grpsess.add_argument('-sessf',
                        help='Input the sessid file',
                        metavar='sessid-file',
                        dest='sessf')
    #parser.add_argument('-sf', 
    #                    help='Input the sessid file',
    #                    dest='sf',
    #                    required=True)
    parser.add_argument('-outDir',
                        help='The output directory.',
                        dest='outDir',
                        required=True)
    parser.add_argument('-regMeth', 
                        help='The registeration method, for sba.',
                        choices=['lin','nonl'],
                        dest='regMeth')
    parser.add_argument('-dd', 
                        help='(Optional)The data directory.If you want \
                        use your own data, pls set this.',
                        dest='dataDir',
                        default='/nfs/t1/nsppara/resting')
    parser.add_argument('-v','--version',
                        action='version',
                        version='%(prog)s 0.1')

    args = parser.parse_args()
    method = args.method
    fsessid = args.sessf
    seed = args.mask
    outdir = args.outDir

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        filename='rsfc.log', filemode='w',
                        level=logging.DEBUG)
    logging.info(args)

    #fsessid = '/nfs/s3/workingshop/wangxu/fc_test/testid'
    #fsessid = open(fsessid)	
    if fsessid:
        fsessid = open(fsessid)
        subject_list  = [line.strip() for line in fsessid]
    else:
        subject_list = [args.sess]
    #subject_list  = [line.strip() for line in fsessid]
    #data_dir = '/nfs/s2/rfmricenter/20120618_result'
    data_dir = args.dataDir
    rawdata_dir = '/nfs/t1/nspnifti/nii'
    standardspace = '/usr/local/neurosoft/fsl5.0.1/data/standard/MNI152_T1_2mm_brain.nii.gz'

    if method == 'sba':
        for subj in subject_list:
            struct = os.path.join(rawdata_dir, subj, '3danat', 'reg_fs', 'brain.nii.gz')
            resdir = os.path.join(data_dir,subj,'res4d')
            resstddir = os.path.join(data_dir,subj,'res4dstandard')
            regdir = os.path.join(data_dir,subj,'reg')
            seed_ts_dir = os.path.join(outdir,subj,'seed_ts')
            rsfcdir = os.path.join(outdir,subj,'RSFC')
            os.system('mkdir -p ' + seed_ts_dir)
            os.system('mkdir -p ' + rsfcdir)
            seed_file = re.split('/',seed)[len(re.split('/',seed))-1]
            seed_name = re.split('\.',seed_file)[0]
            if os.path.exists(rsfcdir + '/' + seed_name + '_Z_2standard.nii.gz'):
                print 'The same file has already existed, please check!\n'

            else:
                print 'Do functional connectivity analysis for seed ' + seed_name + '\n'
                print 'Extracting timeseries for seed ' + seed + '\n'
                os.system('3dROIstats -quiet -mask_f2short -mask ' + seed + \
                          ' ' + resstddir + '/res4dmni.nii.gz'\
                          + ' > ' + seed_ts_dir + '/' + seed_name + '.1D')

                print 'Computing Correlation for seed ' + seed + '\n'
                os.system('3dfim+ -input ' + resdir + '/res4d_demean_add100.nii.gz' + \
                          ' -ideal_file ' + seed_ts_dir + '/' + seed_name + \
                          '.1D -out Correlation -bucket ' + rsfcdir + '/' + \
                          seed_name + '_corr.nii.gz')

                print 'Z-transforming correlations for seed ' + seed + '\n'
                os.system('3dcalc -a ' + rsfcdir + '/' + seed_name + \
                          '_corr.nii.gz -expr ' + '\'log((a+1)/(a-1))/2\' -prefix ' + \
                          rsfcdir + '/' + seed_name + '_Z.nii.gz')

                print 'Registering Z-transformed map to standard space\n'
                if args.regMeth == 'nonl':
                    os.system('flirt -in ' + rsfcdir + '/' + seed_name + \
                              '_Z.nii.gz -ref ' + struct + \
                              ' -applyxfm -init ' + regdir + \
                              '/func2anat.mat -out ' + \
                              rsfcdir + '/' + seed_name + '_Z_2anat.nii.gz')
                    os.system('applywarp --warp=' + regdir + '/' + 'T1_fieldwarp.nii.gz --in=' +\
                              rsfcdir + '/' + seed_name + '_Z_2anat.nii.gz --ref=' + standardspace +\
                              ' --out=' + rsfcdir + '/' + seed_name + '_Z_2standard.nii.gz')
                elif args.regMeth == 'lin':
                    os.system('flirt -in ' + rsfcdir + '/' + seed_name + \
                              '_Z.nii.gz -ref ' + standardspace + \
                              ' -applyxfm -init ' + regdir + \
                              '/func2anat_brain_flirt.mat -out ' + \
                              rsfcdir + '/' + seed_name + '_Z_2standard.nii.gz')


                logging.info(subj + 'ok\n')

#    elif method == 'roi':
#        for subj in subject_list:
#            resdir = os.path.join(data_dir,subj,'res4d')
#            resstddir = os.path.join(data_dir,subj,'res4dstandard')
#            regdir = os.path.join(data_dir,subj,'reg')
#            maskdir = os.path.join(data_dir,subj,'mask')
#            seed_ts_dir = os.path.join(outdir,subj,'seed_ts')
#            seedmask_dir = os.path.join(outdir,subj,'seedmask')
#            os.system('mkdir -p '+seed_ts_dir)
#            os.system('mkdir -p '+seedmask_dir)
#            seed_file = re.split('/',seed)[len(re.split('/',seed))-1]
#            seed_name = re.split('\.',seed_file)[0]
#            if os.path.exists(seed_ts_dir + '/' + seed_name + '_ts.txt'):
#                print 'The same file has already existed, please check!\n'
#                break 
#            else:
#                print 'Do functional connectivity analysis for mask ' + \
#                      seed_name + '\n'
#                print 'Registering mask flie to init space\n'
#                os.system('convert_xfm -omat ' + regdir + '/mni2func.mat -inverse ' + \
#                          regdir + '/func2anat_brain_flirt.mat')
#                os.system('flirt -in ' + seed + \
#                          ' -ref ' + resdir + '/res4d_demean_add100.nii.gz' + \
#                          ' -applyxfm -init ' + regdir + '/mni2func.mat -out ' + \
#                          seedmask_dir + '/' + seed_name + \
#                          '_2func.nii.gz -interp nearestneighbour')
#
#                print 'Extracting timeseries for mask ' + seed + '\n'
#                os.system('mri_segstats --seg ' + seedmask_dir + '/' + seed_name + \
#                          '_2func.nii.gz --i ' + resdir + '/res4d_demean_add100.nii.gz --mask ' + \
#                          maskdir + '/rest_dtype_mcf_bet_thresh_dil.nii.gz --avgwf ' + \
#                          seed_ts_dir + '/' + seed_name + '_ts.txt --sum ' + \
#                          seed_ts_dir + '/' + seed_name + '_summary.txt')
#                logging.info(subj + 'ok\n')
    elif method == 'roi':
        for subj in subject_list:
            resstddir = os.path.join(data_dir,subj,'res4dstandard')
            seed_ts_dir = os.path.join(outdir,subj,'seed_ts')
            os.system('mkdir -p '+seed_ts_dir)
            seed_file = re.split('/',seed)[len(re.split('/',seed))-1]
            seed_name = re.split('\.',seed_file)[0]
            if os.path.exists(seed_ts_dir + '/' + seed_name + '_ts.txt'):
                print subj+'The same file has already existed, please check!\n'
                #break
            else:
                print 'Do functional connectivity analysis for mask ' + \
                      seed_name + '\n'

                print 'Extracting timeseries for mask ' + seed + '\n'
                if not os.path.exists(resstddir+'/mnimask.nii.gz'):
                    print 'Making mask for ' + subj
                    os.system('fslmaths ' + resstddir + '/res4dmni.nii.gz -sub 100 -abs -Tmax -bin '+resstddir+'/mnimask.nii.gz')
                
                os.system('mri_segstats --seg ' + seed + ' --i ' + resstddir + '/res4dmni.nii.gz --mask ' + \
                          resstddir + '/mnimask.nii.gz --avgwf ' + \
                          seed_ts_dir + '/' + seed_name + '_ts.txt --sum ' + \
                          seed_ts_dir + '/' + seed_name + '_summary.txt --excludeid 0')
                logging.info(subj + 'ok\n')

if __name__ == '__main__':
    main()         
