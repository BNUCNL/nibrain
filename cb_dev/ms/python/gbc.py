"""

"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import cdist

class CapAtlas(object):
    
    def __init__(self, atlas_file, annot_file, hemi='LR'):
        '''

        Args:
            atlas_file: String of CAP-ROI file path (eg: ColeAnticevicNetPartition/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii)
            annot_file: String of CAP-annotation file path (eg: CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt)
            hemi: String of selected hemisphere, only L, R and LR

        '''
        self.map = nib.load(atlas_file).get_fdata()
        self.annot = pd.read_csv(annot_file, sep='\t', usecols=['KEYVALUE', 'LABEL', 'NETWORKKEY'])
        self.hemi = list(hemi)

    def get_parcel(self, roi_list=None):
        '''

        Args:
            roi: List of selected ROI range from 1-718

        Returns:
            parcel_mask: Logical array of 718 CAP parcels

        '''

        if roi_list == None:
            roi_list = self.map['KEYVALUE'].tolist()

        parcel_mask = np.zeros((len(roi_list), self.map.shape[1]))
        for roi_index, roi_id in enumerate(roi_list):
            roi_keyvalue = self.annot[self.annot['NETWORKKEY'] == roi_id]['KEYVALUE']
            parcel_mask[roi_index,:] = (self.map[0,:] == np.array(roi_keyvalue)[:,None]).any(axis=0).astype(bool)

        return parcel_mask

    def get_network(self, hemisphere):
        '''

        Args:
            hemisphere: String of selected hemisphere, only L, R and LR

        Returns:
            network_mask: Logical array of 12 CAP networks

        '''

        network_mask = np.zeros((self.annot['NETWORKKEY'].max()*len(self.hemi), self.map.shape[1]))
        for hemi in hemisphere:
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            for network_id in np.arange(1, annot_hemi['NETWORKKEY'].max() + 1):
                network_keyvalue = annot_hemi[annot_hemi['NETWORKKEY'] == network_id]['KEYVALUE']
                network_mask[network_id-1,:] = (self.map[0,:] == np.array(network_keyvalue)[:,None]).any(axis=0).astype(bool)

        return network_mask

    def get_cortex(self, hemisphere):
        '''

        Args:
            hemisphere: String of selected hemisphere, only L, R and LR

        Returns: cortex_mask: Logical array of cortex

        '''

        cortex_mask = np.zeros((len(list(hemisphere)), self.map.shape[1]))
        for index, hemi in enumerate(list(hemisphere)):
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            cortex_keyvalue = annot_hemi[annot_hemi['LABEL'].str.endswith('Ctx')]['KEYVALUE']
            cortex_mask[index,:] = (self.map[0,:] == np.array(cortex_keyvalue)[:,None]).any(axis=0).astype(bool)

        return cortex_mask

    def get_subcortex(self, hemisphere):
        '''

        Args:
            hemisphere: String of selected hemisphere, only L, R and LR

        Returns: subcortex_mask: Logical array of subcortex

        '''

        subcortex_mask = np.zeros((list(hemisphere), self.map.shape[1]))
        for index, hemi in enumerate(list(hemisphere)):
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            subcortex_keyvalue = annot_hemi[(~annot_hemi['LABEL'].str.endswith('Ctx'))&(~annot_hemi['LABEL'].str.endswith('Cerebellum'))&(~annot_hemi['LABEL'].str.endswith('Brainstem'))]['KEYVALUE']
            subcortex_mask[index,:] = (self.map[0,:] == np.array(subcortex_keyvalue)[:,None]).any(axis=0).astype(bool)

        return subcortex_mask

    def get_cerebellum(self, hemisphere):
        '''

        Args:
            hemisphere: String of selected hemisphere, only L, R and LR

        Returns: cerebellum_mask: Logical array of cerebellum

        '''

        cerebellum_mask = np.zeros((len(self.hemi), self.map.shape[1]))
        for index, hemi in enumerate(list(hemisphere)):
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            cerebellum_keyvalue = annot_hemi[annot_hemi['LABEL'].str.endswith('Cerebellum')]['KEYVALUE']
            cerebellum_mask[index,:] = (self.map[0,:] == np.array(cerebellum_keyvalue)[:,None]).any(axis=0).astype(bool)

        return cerebellum_mask

    def get_brainstem(self):
        '''

        Returns: brainstem_mask: Logical array of brainstem

        '''

        brainstem_mask = np.zeros((1, self.map.shape[1]))
        brainstem_keyvalue = self.annot[self.annot['LABEL'].str.endswith('Brainstem')]['KEYVALUE']
        brainstem_mask[0,:] = (self.map[0, :] == np.array(brainstem_keyvalue)[:, None]).any(axis=0).astype(bool)

        return brainstem_mask



def global_brain_conn(cifti_ts, src_roi, targ_roi):
    '''

    Args:
        cifti_ts: Nibabel cifti2 image object of cifti time series file
        cap_atlas: Object of CAP ROI read by CapAtlas()
        src_index: Logical array of source ROI

    Returns:
        conn_mean: Dict of connectivity mean, the order of keys is Parcel-L, Parcel-R, Network-L, Network-R, CortexSubcortexCerebellum-L, CortexSubcortexCerebellum-R

    '''

    cifti_matrix = cifti_ts.get_fdata(structure=None)
    src_matrix = cifti_matrix[:, src_roi]
    targ_matrix = cifti_matrix[:, targ_roi]

    source_brain_conn = 1 - cdist(src_matrix.T, targ_matrix.T, metric='correlation')
    conn_mean = np.mean(source_brain_conn, axis=1)

    return conn_mean

def Ciftiwrite(dir_path, gbc, cifti_ts, src_roi):
    '''

    Args:
        dir_path: String of saving directory path
        gbc: Dict of all connectivity mean matrix
        cifti_ts: Nibabel cifti2 image object of cifti time series file

    '''

    data = np.zeros(1, src_roi.shape[0])
    data[0,src_roi] = gbc
    img = nib.Cifti2Image(dataobj=data, header=cifti_ts.header)
    nib.cifti2.save(img, os.path.join(dir_path, 'gbc.dscalar.nii'))

