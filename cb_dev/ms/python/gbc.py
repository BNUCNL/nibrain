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
        self.annot = self.annot[~self.annot['LABEL'].str.endswith('Brainstem')]
        self.hemi = list(hemi)

    def get_parcel(self):
        '''

        Returns:
            parcel_mask: Logical array of 671 CAP parcels (no brainstem)

        '''

        parcel_mask = np.zeros((len(self.annot), self.map.shape[1]))
        for hemi in self.hemi:
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            for roi_index, roi_id in enumerate(annot_hemi['KEYVALUE'].tolist()):
                roi_keyvalue = annot_hemi[annot_hemi['NETWORKKEY'] == roi_id]['KEYVALUE']
                parcel_mask[roi_index, :] = (annot_hemi[0, :] == np.array(roi_keyvalue)[:, None]).any(axis=0)
        parcel_mask = parcel_mask[~(parcel_mask == 0).all(axis=1)].astype(bool)

        return parcel_mask

    def get_network(self):
        '''

        Returns:
            network_mask: Logical array of 12 CAP networks (no brainstem)

        '''

        network_mask = np.zeros((self.annot['NETWORKKEY'].max()*len(self.hemi), self.map.shape[1]))
        for hemi in self.hemi:
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            for network_id in np.arange(1, annot_hemi['NETWORKKEY'].max() + 1):
                network_keyvalue = annot_hemi[annot_hemi['NETWORKKEY'] == network_id]['KEYVALUE']
                network_mask[network_id-1, :] = (self.map[0, :] == np.array(network_keyvalue)[:, None]).any(axis=0).astype(bool)

        return network_mask

    def get_cortex(self):
        '''

        Returns: cortex_mask: Logical array of cortex

        '''

        cortex_mask = np.zeros((len(self.hemi), self.map.shape[1]))
        for index, hemi in enumerate(self.hemi):
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            cortex_keyvalue = annot_hemi[annot_hemi['LABEL'].str.endswith('Ctx')]['KEYVALUE']
            cortex_mask[index, :] = (self.map[0, :] == np.array(cortex_keyvalue)[:, None]).any(axis=0).astype(bool)

        return cortex_mask

    def get_subcortex(self):
        '''

        Returns: subcortex_mask: Logical array of subcortex

        '''

        subcortex_mask = np.zeros((len(self.hemi), self.map.shape[1]))
        for index, hemi in enumerate(self.hemi):
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            subcortex_keyvalue = annot_hemi[(~annot_hemi['LABEL'].str.endswith('Ctx'))&(~annot_hemi['LABEL'].str.endswith('Cerebellum'))]['KEYVALUE']
            subcortex_mask[index, :] = (self.map[0, :] == np.array(subcortex_keyvalue)[:, None]).any(axis=0).astype(bool)

        return subcortex_mask

    def get_cerebellum(self):
        '''

        Returns: cerebellum_mask: Logical array of cerebellum

        '''

        cerebellum_mask = np.zeros((len(self.hemi), self.map.shape[1]))
        for index, hemi in enumerate(self.hemi):
            annot_hemi = self.annot[self.annot['LABEL'].str.contains(hemi + '-')]
            cerebellum_keyvalue = annot_hemi[annot_hemi['LABEL'].str.endswith('Cerebellum')]['KEYVALUE']
            cerebellum_mask[index, :] = (self.map[0, :] == np.array(cerebellum_keyvalue)[:, None]).any(axis=0).astype(bool)

        return cerebellum_mask

def gbc_to_cap(cifti_ts, cap_atlas, src_index):
    '''

    Args:
        cifti_ts: Nibabel cifti2 image object of cifti time series file
        cap_atlas: Object of CAP ROI read by CapAtlas()
        src_index: Logical array of source ROI

    Returns:
        conn_mean: Dict of connectivity mean, the order of keys is Parcel-L, Parcel-R, Network-L, Network-R, CortexSubcortexCerebellum-L, CortexSubcortexCerebellum-R

    '''

    cifti_matrix = cifti_ts.get_data(structure=None)
    source_matrix = cifti_matrix[:, src_index]

    source_brain_conn = 1 - cdist(source_matrix.T, cifti_matrix.T, metric='correlation')

    cap_atlas.hemi = 'L'
    target_mask_L = {
        'Parcel-L' : cap_atlas.get_parcel(),
        'Network-L' : cap_atlas.get_network(),
        'CortexSubcortexCerebellum-L' : np.concatenate((cap_atlas.get_cortex(), cap_atlas.get_subcortex(), cap_atlas.get_cerebellum())).reshape(3,-1)
    }
    cap_atlas.hemi = 'R'
    target_mask_R = {
        'Parcel-R': cap_atlas.get_parcel(),
        'Network-R': cap_atlas.get_network(),
        'CortexSubcortexCerebellum-R': np.concatenate((cap_atlas.get_cortex(), cap_atlas.get_subcortex(), cap_atlas.get_cerebellum())).reshape(3, -1)
    }
    target_mask = dict(target_mask_L, **target_mask_R)

    conn_mean = dict()
    for target, mask in target_mask.items():
        conn_mean_matrix = np.zeros((source_brain_conn.shape[0], mask.shape[0]))
        for roi in np.arange(mask.shape[0]):
            current_target = mask[roi, :]
            _mask = current_target.astype(int)[1,:] - src_index
            current_target = np.where(_mask < 0, 0, _mask).astype(bool)
            roi_mean = np.mean(source_brain_conn[:, current_target], axis=1)
            conn_mean_matrix[:, roi] = roi_mean
        conn_mean[target] = conn_mean_matrix

    return conn_mean

def Ciftiwrite(dir_path, gbc, cifti_ts):
    '''

    Args:
        dir_path: String of saving directory path
        gbc: Dict of all connectivity mean matrix
        cifti_ts: Nibabel cifti2 image object of cifti time series file

    '''

    for name, data in gbc.items():
        img = nib.Cifti2Image(dataobj=data, header=cifti_ts.header)
        nib.cifti2.save(img, os.path.join(dir_path, 'source_'+name+'_connectivity.dscalar.nii'))

