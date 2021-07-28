import cv2
import nibabel as nib
import numpy as np

from nibabel.cifti2 import cifti2
from collections import OrderedDict


class CiftiReader(object):

    def __init__(self, file_path):
        self.full_data = cifti2.load(file_path)

    @property
    def header(self):
        return self.full_data.header

    @property
    def brain_structures(self):
        return [_.brain_structure for _ in self.header.get_index_map(1).brain_models]

    @property
    def label_info(self):
        """
        Get label information from label tables
        Return:
        ------
        label_info[list]:
            Each element is a dict about corresponding map's label information.
            Each dict's content is shown as below:
                key[list]: a list of integers which are data values of the map
                label[list]: a list of label names
                rgba[ndarray]: shape=(n_label, 4)
                    The four elements in the second dimension are
                    red, green, blue, and alpha color components for label (between 0 and 1).
        """
        label_info = []
        for named_map in self.header.get_index_map(0).named_maps:
            label_dict = {'key': [], 'label': [], 'rgba': []}
            for k, v in named_map.label_table.items():
                label_dict['key'].append(k)
                label_dict['label'].append(v.label)
                label_dict['rgba'].append(v.rgba)
            label_dict['rgba'] = np.asarray(label_dict['rgba'])
            label_info.append(label_dict)

        return label_info

    @property
    def volume(self):
        return self.header.get_index_map(1).volume

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
        if structures is not None:
            if not isinstance(structures, list):
                raise TypeError("The parameter 'structures' must be a list")
            brain_models = [brain_models[self.brain_structures.index(s)] for s in structures]
        return brain_models

    def map_names(self, rows=None):
        """
        get map names
        Parameters:
        ----------
        rows: sequence of integer
            Specify which map names should be got.
            If None, get all map names
        Return:
        ------
        map_names: list of str
        """
        named_maps = list(self.header.get_index_map(0).named_maps)
        if named_maps:
            if rows is None:
                map_names = [named_map.map_name for named_map in named_maps]
            else:
                map_names = [named_maps[i].map_name for i in rows]
        else:
            map_names = []
        return map_names

    def label_tables(self, rows=None):
        """
        get label tables
        Parameters:
        ----------
        rows: sequence of integer
            Specify which label tables should be got.
            If None, get all label tables.
        Return:
        ------
        label_tables: list of Cifti2LableTable
        """
        named_maps = list(self.header.get_index_map(0).named_maps)
        if named_maps:
            if rows is None:
                label_tables = [named_map.label_table for named_map in named_maps]
            else:
                label_tables = [named_maps[i].label_table for i in rows]
        else:
            label_tables = []
        return label_tables

    def get_data(self, structure=None, zeroize=False):
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

        _data = np.array(self.full_data.get_data())
        if structure is not None:
            brain_model = self.brain_models([structure])[0]
            offset = brain_model.index_offset
            count = brain_model.index_count

            if zeroize:
                if brain_model.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
                    n_vtx = brain_model.surface_number_of_vertices
                    data = np.zeros((_data.shape[0], n_vtx), _data.dtype)
                    data[:, list(brain_model.vertex_indices)] = _data[:, offset:offset+count]
                elif brain_model.model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                    # This function have not been verified visually.
                    vol_shape = self.header.get_index_map(1).volume.volume_dimensions
                    data_shape = (_data.shape[0],) + vol_shape
                    data_ijk = np.array(list(brain_model.voxel_indices_ijk))
                    data = np.zeros(data_shape, _data.dtype)
                    data[:, data_ijk[:, 0], data_ijk[:, 1], data_ijk[:, 2]] = _data[:, offset:offset+count]
                else:
                    raise RuntimeError("The function can't support the brain model: {}".format(brain_model.model_type))
                return data
            else:
                if brain_model.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
                    map_shape = (brain_model.surface_number_of_vertices,)
                    index2v = list(brain_model.vertex_indices)
                elif brain_model.model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                    # This function have not been verified visually.
                    map_shape = self.header.get_index_map(1).volume.volume_dimensions
                    index2v = list(brain_model.voxel_indices_ijk)
                else:
                    raise RuntimeError("The function can't support the brain model: {}".format(brain_model.model_type))
                return _data[:, offset:offset+count], map_shape, index2v
        else:
            return _data


def save2cifti(file_path, data, brain_models, map_names=None, volume=None, label_tables=None):
    """
    Save data as a cifti file
    If you just want to simply save pure data without extra information,
    you can just supply the first three parameters.
    NOTE!!!!!!
        The result is a Nifti2Image instead of Cifti2Image, when nibabel-2.2.1 is used.
        Nibabel-2.3.0 can support for Cifti2Image indeed.
        And the header will be regard as Nifti2Header when loading cifti file by nibabel earlier than 2.3.0.
    Parameters:
    ----------
    file_path: str
        the output filename
    data: numpy array
        An array with shape (maps, values), each row is a map.
    brain_models: sequence of Cifti2BrainModel
        Each brain model is a specification of a part of the data.
        We can always get them from another cifti file header.
    map_names: sequence of str
        The sequence's indices correspond to data's row indices and label_tables.
        And its elements are maps' names.
    volume: Cifti2Volume
        The volume contains some information about subcortical voxels,
        such as volume dimensions and transformation matrix.
        If your data doesn't contain any subcortical voxel, set the parameter as None.
    label_tables: sequence of Cifti2LableTable
        Cifti2LableTable is a mapper to map label number to Cifti2Label.
        Cifti2Lable is a specification of the label, including rgba, label name and label number.
        If your data is a label data, it would be useful.
    """
    if file_path.endswith('.dlabel.nii'):
        assert label_tables is not None
        idx_type0 = 'CIFTI_INDEX_TYPE_LABELS'
    elif file_path.endswith('.dscalar.nii'):
        idx_type0 = 'CIFTI_INDEX_TYPE_SCALARS'
    else:
        raise TypeError('Unsupported File Format')

    if map_names is None:
        map_names = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(map_names), "Map_names are mismatched with the data"

    if label_tables is None:
        label_tables = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(label_tables), "Label_tables are mismatched with the data"

    # CIFTI_INDEX_TYPE_SCALARS always corresponds to Cifti2Image.header.get_index_map(0),
    # and this index_map always contains some scalar information, such as named_maps.
    # We can get label_table and map_name and metadata from named_map.
    mat_idx_map0 = cifti2.Cifti2MatrixIndicesMap([0], idx_type0)
    for mn, lbt in zip(map_names, label_tables):
        named_map = cifti2.Cifti2NamedMap(mn, label_table=lbt)
        mat_idx_map0.append(named_map)

    # CIFTI_INDEX_TYPE_BRAIN_MODELS always corresponds to Cifti2Image.header.get_index_map(1),
    # and this index_map always contains some brain_structure information, such as brain_models and volume.
    mat_idx_map1 = cifti2.Cifti2MatrixIndicesMap([1], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
    for bm in brain_models:
        mat_idx_map1.append(bm)
    if volume is not None:
        mat_idx_map1.append(volume)

    matrix = cifti2.Cifti2Matrix()
    matrix.append(mat_idx_map0)
    matrix.append(mat_idx_map1)
    header = cifti2.Cifti2Header(matrix)
    img = cifti2.Cifti2Image(data, header)
    cifti2.save(img, file_path)


class GiftiReader(object):

    def __init__(self, file_path):
        self._fpath = file_path
        self.full_data = nib.load(file_path)

    @property
    def coords(self):
        if self._fpath.endswith('.surf.gii'):
            return self.full_data.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
            # return self.full_data.darrays[0].data
        else:
            return None

    @property
    def faces(self):
        if self._fpath.endswith('.surf.gii'):
            return self.full_data.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
            # return self.full_data.darrays[1].data
        else:
            return None

    @property
    def scalar_data(self):
        if self._fpath.endswith('.surf.gii'):
            return None
        else:
            return self.full_data.darrays[0].data


def save2nifti(fpath, data, affine=None, header=None):
    """
    Save to a Nifti file.
    Parameters
    ----------
    fpath: str
        The file path to output
    data: numpy array
    affine: numpy array
    header: Nifti2Header
    """
    img = nib.Nifti2Image(data, affine, header=header)
    # Nifti1 standard uses a short int to represent the dimensions.
    # As a result, it can only represent -32767~32767.
    # Use Nifti2 standard can avoid this problem.
    nib.nifti2.save(img, fpath)


def save2mgh(fpath, data, affine=None, header=None):
    """
    Save to a MGH/MGZ file
    The .mgh file format is used to store high-resolution structural data and
    other data which are to be overlaid on the high-resolution structural volume.
    A .mgz (or .mgh.gz) file is a .mgh file that has been compressed with ZLib.
    NOTE!!! MGH file format seemingly has 3D dimensions at least. As a result, it essentially
    regards the first dimensions as a volume and the forth dimension as the number of volumes.
    References:
        https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
        http://nipy.org/nibabel/reference/nibabel.freesurfer.html#nibabel.freesurfer.mghformat.MGHImage
    :param fpath: str
        The file path to output. valid_exts = ('.mgh', '.mgz')
    :param data: numpy array
    :param affine: numpy array
    :param header: MGHHeader
    """
    img = nib.MGHImage(data, affine, header=header)
    nib.save(img, fpath)


class CsvReader(object):

    def __init__(self, file_path):
        with open(file_path) as rf:
            lines = rf.read().splitlines()
        self.rows = [line.split(',') for line in lines]
        del lines
        self.cols = list(map(list, zip(*self.rows)))

    def to_dict(self, axis=0, keys=None):
        """
        transform contents of csv file to python dictionary
        :param axis: 0 or 1
            If 0, elements in the first row will be regard as keys. And the rest elements of one key's
            corresponding column are collected into a list as the key's value.
            elif 1, elements in the first column will be regard as keys. And the rest elements of one key's
            corresponding row are collected into a list as the key's value.
            else, error.
        :param keys: sequence
            if None, transform all contents.
            else, just get contents whose keys are in 'keys'.
        :return: csv_dict: OrderedDict
        """
        csv_dict = OrderedDict()
        if axis == 0:
            if keys is None:
                keys = self.rows[0]
            for key in keys:
                csv_dict[key] = self.cols[self.rows[0].index(key)][1:]
        elif axis == 1:
            if keys is None:
                keys = self.cols[0]
            for key in keys:
                csv_dict[key] = self.rows[self.cols[0].index(key)][1:]
        else:
            raise ValueError('axis must be 0 or 1')

        return csv_dict


class VideoReader:
    """
    read video data
    """
    def __init__(self, vid_file, skip=0, interval=1):
        """
        Parameters:
        -----------
        vid_file[str]: video data file
        skip[float]: skip 'skip' seconds at the start of the video
        interval[int]: get one frame per 'interval' frames
        """
        assert skip >= 0, "Parameter 'skip' must be a nonnegtive value!"
        assert isinstance(interval, int) and interval > 0, "Parameter 'interval' must be a positive integer!"
        self.vid_cap = cv2.VideoCapture(vid_file)
        self.skip = skip
        self.interval = interval

        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        self.n_frame = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.init = int(self.skip * self.fps)  # the first frame's index

    def __getitem__(self, idx):
        # process index range
        assert isinstance(idx, int), 'Index must be a integer!'
        if idx >= self.__len__() or idx < -self.__len__():
            raise IndexError('index out of range')
        if idx < 0:
            idx = self.__len__() + idx

        frame_idx = self.init + idx * self.interval
        self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = self.vid_cap.read()
        return frame

    def __len__(self):
        length = (self.n_frame - self.init) / self.interval
        return int(np.ceil(length))


def save2label(fpath, vertices, hemi_coords=None):
    """
    save labeled vertices to a text file
    Parameters
    ----------
    fpath : string
        The file path to output
    vertices : 1-D array_like sequence
        labeled vertices
    hemi_coords : numpy array
        If not None, it means that saving vertices as the freesurfer style.
    """
    header = str(len(vertices))
    vertices = np.array(vertices)
    if hemi_coords is None:
        np.savetxt(fpath, vertices, fmt='%d', header=header,
                   comments="#!ascii, label vertexes\n")
    else:
        coords = hemi_coords[vertices]
        unknow = np.zeros_like(vertices, np.float16)
        X = np.c_[vertices, coords, unknow]
        np.savetxt(fpath, X, fmt=['%d', '%f', '%f', '%f', '%f'],
                   header=header, comments="#!ascii, label vertexes\n")