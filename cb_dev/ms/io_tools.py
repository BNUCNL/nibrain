import numpy as np
from nibabel.cifti2 import cifti2

class cerebellum_Reader(object):

    def __init__(self, file_path):
        self.full_data = cifti2.load(file_path)

    @property
    def header(self):
        return self.full_data.header

    @property
    def brain_structures(self):
        return [_.brain_structure for _ in self.header.get_index_map(1).brain_models]

    def brain_models(self, structures=None):
        """
        get brain model from cifti file
        Parameter:
        ---------
        structures: list of str
            Each structure corresponds to a brain model.
            If None, get all brain models.
        Return:
        ------
            brain_models: list of Cifti2BrainModel
        """
        brain_models = list(self.header.get_index_map(1).brain_models)
        brain_models = [brain_models[self.brain_structures.index(s)] for s in structures]
        return brain_models

    def get_data(self, structure=None):
        """
        get data from cifti file
        Parameters:
        ----------
        structure: str
            One structure corresponds to one brain model.
            specify which brain structure's data should be extracted
            If None, get all structures, meanwhile ignore parameter 'zeroize'.
        zeroize: bool
            If true, get data after filling zeros for the missing vertices/voxels.
        Return:
        ------
        data: numpy array
            If zeroize doesn't take effect, the data's shape is (map_num, index_num).
            If zeroize takes effect and brain model type is SURFACE, the data's shape is (map_num, vertex_num).
            If zeroize takes effect and brain model type is VOXELS, the data's shape is (map_num, i_max, j_max, k_max).
        map_shape: tuple
            the shape of the map.
            If brain model type is SURFACE, the shape is (vertex_num,).
            If brain model type is VOXELS, the shape is (i_max, j_max, k_max).
            Only returned when 'structure' is not None and zeroize is False.
        index2v: list
            index2v[cifti_data_index] == map_vertex/map_voxel
            Only returned when 'structure' is not None and zeroize is False.
        """

        _data = np.array(self.full_data.get_fdata())

        brain_model = self.brain_models([structure])[0]
        offset = brain_model.index_offset
        count = brain_model.index_count

        print(brain_model.model_type)

        vol_shape = self.header.get_index_map(1).volume.volume_dimensions
        data_shape = (_data.shape[0],) + vol_shape
        data_ijk = np.array(list(brain_model.voxel_indices_ijk))
        data = np.zeros(data_shape, _data.dtype)
        data[:, data_ijk[:, 0], data_ijk[:, 1], data_ijk[:, 2]] = _data[:, offset:offset + count]
        return data

if __name__ == '__main__':
    cb_reader = cerebellum_Reader('/nfs/e1/HCPD/fmriresults01/HCD0001305_V1_MR/MNINonLinear/Results/rfMRI_REST1_AP/rfMRI_REST1_AP_Atlas_MSMAll_hp0_clean.dtseries.nii')
    data = cb_reader.get_data(structure='CIFTI_STRUCTURE_CEREBELLUM_LEFT')
    print(data.shape)