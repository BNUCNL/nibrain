"""
Calculates fractional amplitude of low-frequency fluctuations (fALFF)
Usage:
    falff_nifti.py <func.nii.gz> <output.nii.gz> [options]
Arguments:
    <func.nii.gz>  The functional 4D nifti files
    <output.nii.gz>  Output filename
    <maskfile.nii.gz>  A brainmask for the functional file
Options:
  --min-low-freq 0.01  Min low frequency range value Hz [default: 0.01]
  --max-low-freq 0.08  Max low frequency range value Hz [default: 0.08]
  --min-total-freq 0.00  Min total frequency range value Hz [default: 0.00]
  --max-total-freq 0.25  Max total frequency range value Hz [default: 0.25]
  --mask-file <maskfile.nii.gz>  Input brain mask
  --calc-alff  Calculates amplitude of low frequency fluctuations (ALFF) instead of fALFF
  --debug  Debug logging
  -h,--help  Print help
"""

import os
import numpy as np
import nibabel as nib
from scipy.fftpack import fft
from docopt import docopt
import tempfile
import shutil
import subprocess as proc
import sys

def main():
    ''''''
    arguments = docopt(__doc__)
    funcfile = arguments['<func.nii.gz>']
    outputname = arguments['<output.nii.gz>']
    min_low_freq = arguments['--min-low-freq']
    max_low_freq = arguments['--max-low-freq']
    min_total_freq = arguments['--min-total-freq']
    max_total_freq = arguments['--max-total-freq']
    maskfile = arguments['--mask-file']
    calc_alff = arguments['--calc-alff']

    DEBUG = arguments['--debug']

    if DEBUG: print(arguments)

    # Makes a temp directory for fake input nifti and falff output nifti
    tmpdir = tempfile.mkdtemp()
    print(tmpdir)
  
    # IF INPUT IS A NIFTI FILE 
    # Sets input funcfile equal to inputfile 
    inputfile = funcfile
    
    # IF INPUT IS CIFTI FILE
    # Convert cifti input file to nifti input file
    if 'nii.gz' not in funcfile:
        inputfile = convert_cifti_to_nifti(funcfile, tmpdir)

    falff_nifti_output = calc_nifti(inputfile, maskfile, min_low_freq, max_low_freq, min_total_freq, max_total_freq, tmpdir, calc_alff)

    # Convert nifti output file to cifti output file
    if 'nii.gz' not in funcfile:
        convert_nifti_to_cifti(falff_nifti_output, funcfile, outputname)

    # IF INPUT IS NIFTI FILE
    # If funcfile was not cifti file, save as nifti file to outputname
    if 'nii.gz' in funcfile:
        run("mv {} {}".format(falff_nifti_output, outputname))

    # Remove tmpdir and all contents
    shutil.rmtree(tmpdir)

# Runs the wb command on separate terminal 
def run(cmd):
    '''
    Runs a subprocess command:
    Arguments:
        cmd                     BASH command to Runs
    Returns:
    '''
    p = proc.Popen(cmd,stdin=proc.PIPE, stdout=proc.PIPE, shell=True)
    std, err = p.communicate()

    if p.returncode:
        print('Connectome workbench crashed with error {}'.format(err))
        sys.exit(1)
    return

# If input is cifti - convert to fake nifti (fake_nifti_input)
# Convert to nifti
def convert_cifti_to_nifti(funcfile, tmpdir):
    fake_nifti_input = os.path.join(tmpdir, 'input_fake.nii.gz')
    run('wb_command -cifti-convert -to-nifti {} {} '.format(funcfile, fake_nifti_input))
    return fake_nifti_input

# If input is cifti - convert nifti output (falff_nifti_output) back to cifti
# Convert to cifti
def convert_nifti_to_cifti(falff_nifti_output, funcfile, outputname):
    run('wb_command -cifti-convert -from-nifti {} {} {} -reset-scalars'.format(falff_nifti_output, funcfile, outputname))

# Takes input files to give to falff function and returns output file 
def calc_nifti(inputfile, maskfile, min_low_freq, max_low_freq, min_total_freq, max_total_freq, tmpdir, calc_alff):
    '''
    calculates falff from nifti input and retruns nifti output 
    '''
    # Load in functional data
    func_img = nib.load(inputfile)
    func_data = func_img.get_data()

    # If given input of mask, load in mask file
    # OR if not given input of mask, create mask using std
    if maskfile:
        #1. Given input of mask file
        mask = (nib.load(maskfile)).get_data()
    else:
        #2. Manually create mask
        mask = np.std(func_data, axis=3)

    # Find indices where mask does not = 0
    indx,indy,indz = np.where(mask != 0)

    # Define affine array
    affine = func_img.affine

    # Define x,y,z,t coordinates
    x,y,z,t = func_data.shape

    # Create empty array to save values
    falff_vol = np.zeros((x,y,z))

    # Loop through x,y,z indices, send to calculate_falff function
    for x,y,z in zip(indx,indy,indz):
        falff_vol[x,y,z] = calculate_falff(func_data[x,y,z,:], min_low_freq, max_low_freq, min_total_freq, max_total_freq, calc_alff)

    # Save falff values to fake nifti output temp file
    output_3D = nib.Nifti1Image(falff_vol, affine)
    falff_nifti_output = os.path.join(tmpdir, 'output_fake.nii.gz') # Make temp directory for nifti output
    output_3D.to_filename(falff_nifti_output)
   
    return falff_nifti_output

# CALCULATES FALFF
def calculate_falff(timeseries, min_low_freq, max_low_freq, min_total_freq, max_total_freq, calc_alff):
    ''' this will calculate falff from a timeseries'''

    n = len(timeseries)
    time = (np.arange(n))*2

    # Takes fast Fourier transform of timeseries
    fft_timeseries = fft(timeseries)
    # Calculates frequency scale
    freq_scale = np.fft.fftfreq(n, 1/1)

    # Calculates power of fft
    mag = (abs(fft_timeseries))**0.5

    # Finds low frequency range (0.01-0.08) and total frequency range (0.0-0.25)
    low_ind = np.where((float(min_low_freq) <= freq_scale) & (freq_scale <= float(max_low_freq)))
    total_ind = np.where((float(min_total_freq) <= freq_scale) & (freq_scale <= float(max_total_freq)))

    # Indexes power to low frequency index, total frequency range
    low_power = mag[low_ind]
    total_power = mag[total_ind]
    # Calculates sum of lower power and total power
    low_pow_sum = np.sum(low_power)
    total_pow_sum = np.sum(total_power)
    
    # Calculates alff as the sum of amplitudes within the low frequency range 
    if calc_alff:
        calc = low_pow_sum
    # Calculates falff as the sum of power in low frequnecy range divided by sum of power in the total frequency range
    else:
        calc = np.divide(low_pow_sum, total_pow_sum)

    return calc

if __name__=='__main__':
    main()