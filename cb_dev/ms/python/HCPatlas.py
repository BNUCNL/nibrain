#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:28:40 2021

@author: masai
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib


class CAP(object):

    def __init__(self, CAP_dir):

        self.map = nib.load(os.path.join(CAP_dir,
                                         'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii')).get_fdata()
        self.annot = pd.read_csv(
            os.path.join(CAP_dir, 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'),
            sep='\t', usecols=['KEYVALUE', 'LABEL', 'NETWORKKEY'])

    def get_parcel(self, parcel_list=[], hemisphere='LR'):

        if hemisphere == 'LR':
            annot = self.annot
        else:
            annot = self.annot[self.annot['LABEL'].str.contains('_' + hemisphere + '-')]

        if parcel_list == []:
            parcel_list = annot['KEYVALUE'].tolist()

        parcel_mask = (self.map == np.array(parcel_list)[:, None])

        return parcel_mask.astype(bool)

    def get_network(self, network_list=[], hemisphere='LR', return_idx=False):

        if network_list == []:
            network_list = list(range(1, 13))

        if hemisphere == 'LR':
            annot = self.annot
        else:
            annot = self.annot[self.annot['LABEL'].str.contains('_' + hemisphere + '-')]

        network_mask = np.zeros((len(network_list), self.map.shape[1]))
        parcel_list = list()
        for mask_row, network_id in enumerate(network_list):
            parcel_in_network = annot[annot['NETWORKKEY'] == network_id]['KEYVALUE']
            parcel_list.append(parcel_in_network.tolist())
            network_mask[mask_row, :] = (self.map == np.array(parcel_in_network)[:, None]).any(axis=0)

        if return_idx:
            return parcel_list
        else:
            return network_mask.astype(bool)

    def get_structure(self, structure_List=[], hemisphere='LR', return_idx=False):

        if structure_List == []:
            structure_List = ['Ctx', 'Subctx', 'Cerebellum']

        if hemisphere == 'LR':
            annot = self.annot
        else:
            annot = self.annot[self.annot['LABEL'].str.contains('_' + hemisphere + '-')]

        structure_mask = np.zeros((len(structure_List), self.map.shape[1]))
        parcel_list = list()
        for mask_row, structure_name in enumerate(structure_List):
            if structure_name == 'Subctx':
                parcel_in_structure = annot[
                    (~annot['LABEL'].str.endswith('Ctx')) & (~annot['LABEL'].str.endswith('Cerebellum')) & (
                        ~annot['LABEL'].str.endswith('Brainstem'))]['KEYVALUE']
            else:
                parcel_in_structure = annot[annot['LABEL'].str.endswith(structure_name)]['KEYVALUE']
            parcel_list.append(parcel_in_structure.tolist())
            structure_mask[mask_row, :] = (self.map == np.array(parcel_in_structure)[:, None]).any(axis=0)

        if return_idx:
            return parcel_list
        else:
            return structure_mask.astype(bool)
