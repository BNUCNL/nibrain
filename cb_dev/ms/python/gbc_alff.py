#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 07:47:53 2021

@author: masai
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy.fftpack import fft
from scipy.spatial.distance import cdist


class CapAtlas(object):

    def __init__(self, atlas_dir):
        '''

        Args:
            atlas_dir: str
                CAP folder directory

        '''

        self.map = nib.load(os.path.join(atlas_dir,
                                         'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii')).get_fdata()
        self.annot = pd.read_csv(
            os.path.join(atlas_dir, 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'),
            sep='\t', usecols=['KEYVALUE', 'LABEL', 'NETWORKKEY'])

    def get_parcel(self, roi_list=None):
        '''
        Get logical array mask of CAP parcels

        Args:
            roi_list: list
                The ROI numbers you want to obtain

        Returns:
            parcel_mask: numpy array
                Logical mask of parcels
        '''

        # if roi_list is None, it will be assigned all parcels
        if roi_list == None:
            roi_list = self.annot['KEYVALUE'].tolist()

        # get an empty array to store the parcels mask
        parcel_mask = np.zeros((len(roi_list), self.map.shape[1]))
        # loop the list to get the mask of each element
        for roi_index, roi_id in enumerate(roi_list):
            roi_keyvalue = self.annot[self.annot['NETWORKKEY'] == roi_id]['KEYVALUE']
            parcel_mask[roi_index, :] = (self.map[0, :] == np.array(roi_keyvalue)[:, None]).any(axis=0).astype(bool)

        return parcel_mask.astype(bool)

    def get_network(self, hemisphere=None):
        '''
        Get logical array mask of CAP networks

        Args:
            hemisphere: str
                To determine the hemisphere you want, enter "L", "R", "LR" or None

        Returns:
            network_mask: numpy array
                Logical mask of networks
        '''

        # get an empty array to store the networks mask
        network_mask = np.zeros((self.annot['NETWORKKEY'].max(), self.map.shape[1]))

        # get the mask according to the input hemisphere
        if hemisphere == 'L' or hemisphere == 'R':
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemisphere + '-')]
            # loop the array of network-id to get the mask of each element
            for network_id in np.arange(1, annot_hemi['NETWORKKEY'].max() + 1):
                network_keyvalue = annot_hemi[annot_hemi['NETWORKKEY'] == network_id]['KEYVALUE']
                network_mask[network_id - 1, :] = (self.map[0, :] == np.array(network_keyvalue)[:, None]).any(
                    axis=0).astype(bool)
        else:
            for network_id in np.arange(1, self.annot['NETWORKKEY'].max() + 1):
                network_keyvalue = self.annot[self.annot['NETWORKKEY'] == network_id]['KEYVALUE']
                network_mask[network_id - 1, :] = (self.map[0, :] == np.array(network_keyvalue)[:, None]).any(
                    axis=0).astype(bool)

        return network_mask.astype(bool)

    def get_cortex(self, hemisphere):
        '''
        Get logical array mask of cortex

        Args:
            hemisphere: str
                To determine the hemisphere you want, enter "L", "R" or "LR"

        Returns:
            cortex_mask: numpy array
                Logical mask of cortex
        '''

        # get an empty array to store the cortex mask
        cortex_mask = np.zeros((len(hemisphere), self.map.shape[1]))
        # get the mask according to the input hemisphere
        for index, hemi in enumerate(list(hemisphere)):
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            # get the id of cortex ROI from annotation dataframe
            cortex_keyvalue = annot_hemi[annot_hemi['LABEL'].str.endswith('Ctx')]['KEYVALUE']
            cortex_mask[index, :] = (self.map[0, :] == np.array(cortex_keyvalue)[:, None]).any(axis=0).astype(bool)

        # if shape is (1, 91282), convert it to (91282,)
        if cortex_mask.shape[0] == 1:
            cortex_mask = np.squeeze(cortex_mask)

        return cortex_mask.astype(bool)

    def get_subcortex(self, hemisphere):
        '''
        Get logical array mask of subcortex

        Args:
            hemisphere: str
                To determine the hemisphere you want, enter "L", "R" or "LR"

        Returns:
            subcortex_mask: numpy array
                Logical mask of subcortex
        '''

        # get an empty array to store the subcortex mask
        subcortex_mask = np.zeros((len(hemisphere), self.map.shape[1]))
        # get the mask according to the input hemisphere
        for index, hemi in enumerate(list(hemisphere)):
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            # get the id of subcortex ROI from annotation dataframe
            subcortex_keyvalue = annot_hemi[
                (~annot_hemi['LABEL'].str.endswith('Ctx')) & (~annot_hemi['LABEL'].str.endswith('Cerebellum')) & (
                    ~annot_hemi['LABEL'].str.endswith('Brainstem'))]['KEYVALUE']
            subcortex_mask[index, :] = (self.map[0, :] == np.array(subcortex_keyvalue)[:, None]).any(axis=0).astype(
                bool)

        # if shape is (1, 91282), convert it to (91282,)
        if subcortex_mask.shape[0] == 1:
            subcortex_mask = np.squeeze(subcortex_mask)

        return subcortex_mask.astype(bool)

    def get_cerebellum(self, hemisphere):
        '''
        Get logical array mask of cerebellum

        Args:
            hemisphere: str
                To determine the hemisphere you want, enter "L", "R" or "LR"

        Returns:
            cerebellum_mask: numpy array
                Logical mask of cerebellum
        '''

        # get an empty array to store the cerebellum mask
        cerebellum_mask = np.zeros((len(hemisphere), self.map.shape[1]))
        # get the mask according to the input hemisphere
        for index, hemi in enumerate(list(hemisphere)):
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            # get the id of cerebellum ROI from annotation dataframe
            cerebellum_keyvalue = annot_hemi[annot_hemi['LABEL'].str.endswith('Cerebellum')]['KEYVALUE']
            cerebellum_mask[index, :] = (self.map[0, :] == np.array(cerebellum_keyvalue)[:, None]).any(axis=0).astype(
                bool)

        # if shape is (1, 91282), convert it to (91282,)
        if cerebellum_mask.shape[0] == 1:
            cerebellum_mask = np.squeeze(cerebellum_mask)

        return cerebellum_mask.astype(bool)

    def get_brainstem(self):
        '''
        Get logical array mask of brainstem

        Returns:
            brainstem_mask: numpy array
                Logical mask of brainstem
        '''

        # get an empty array to store the brainstem mask
        brainstem_mask = np.zeros((self.map.shape[1], None))
        # get the id of brainstem ROI from annotation dataframe
        brainstem_keyvalue = self.annot[self.annot['LABEL'].str.endswith('Brainstem')]['KEYVALUE']
        brainstem_mask[:, None] = (self.map[0, :] == np.array(brainstem_keyvalue)[:, None]).any(axis=0).astype(bool)

        return brainstem_mask.astype(bool)


def global_brain_conn(cifti_ts, src_roi, targ_roi):
    '''
    Calculate global brain connectivity between source ROI and target ROI

    Args:
        cifti_ts: nibabel cifti2Image
            Time series object read by nibabel.load()
        src_roi: numpy array
            Logical mask of source ROI
        targ_roi: numpy array
            Logical mask of target ROI

    Returns:
        conn_mean: numpy array
            mean of connectivity between source and target
    '''

    # prepare source and target data index with src_roi & targ_roi
    cifti_matrix = cifti_ts.get_fdata()
    src_matrix = cifti_matrix[:, src_roi]
    targ_matrix = cifti_matrix[:, targ_roi]

    # calculate mean of connectivity between source and target
    source_brain_conn = 1 - cdist(src_matrix.T, targ_matrix.T, metric='correlation')
    conn_mean = np.mean(source_brain_conn, axis=1)

    return conn_mean.reshape(-1, 1)


def alff(cifti_ts, src_roi, tr, low_freq_band=(0.01, 0.1)):
    '''
    Calculate Amplitude of low-frequency fluctuation (ALFF) or Fractional amplitude of low-frequency fluctuation (fALFF) of source ROI

    Args:
        cifti_ts: nibabel cifti2Image
            time series object read by nibabel.load()
        src_roi: numpy array
            Logical mask of source ROI
        tr: int
            Repetation time (second)
        low_freq_band: tuple
            (m, n) low freqency scale is from m to n (Hz)

    Returns:
        alff_falff: tuple
            ALFF & fALFF numpy array of source ROI ()
    '''

    # prepare source data index with src_roi
    cifti_matrix = cifti_ts.get_fdata()
    src_matrix = cifti_matrix[:, src_roi]

    # fast fourier transform
    fft_array = fft(signal.detrend(src_matrix, axis=0, type='linear'), axis=0)
    # get fourier transform sample frequencies
    freq_scale = np.fft.fftfreq(src_matrix.shape[0], tr)
    # calculate total power of all freqency bands
    total_power = np.sqrt(np.absolute(fft_array[(0.0 <= freq_scale) & (freq_scale <= (1 / tr) / 2), :]))
    # calculate ALFF or fALFF
    # ALFF: sum of low band power
    # fALFF: ratio of Alff to total power
    time_half = int(cifti_matrix.shape[0] / 2)
    alff = np.sum(
        total_power[(low_freq_band[0] <= freq_scale)[0:time_half] & (freq_scale <= low_freq_band[1])[0:time_half], :],
        axis=0)
    falff = alff / np.sum(total_power, axis=0)

    return np.concatenate((alff.reshape(-1, 1), falff.reshape(-1, 1)), axis=1)


def Ciftiwrite(file_path, data, cifti_ts, src_roi):
    '''
    Restore and save as cifti

    Args:
        dir_path: str
            the output filename
        data: numpy array
            matrix of result calculate by global_brain_conn() or alff()
        cifti_ts: nibabel cifti2Image
            time series object read by nibabel.load()
    '''

    # get an empty array to store the restored source data
    save_data = np.zeros((data.shape[1], src_roi.shape[0]))
    # restore source data to standard cifti space
    save_data[:, src_roi] = data.T
    # save as cifti file
    img = nib.Cifti2Image(dataobj=save_data, header=cifti_ts.header)
    nib.cifti2.save(img, file_path)
