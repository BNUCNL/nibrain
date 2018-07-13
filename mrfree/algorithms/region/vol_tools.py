# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import os
import copy
import numpy as np
import nibabel as nib

from scipy import stats
from scipy.spatial.distance import pdist
from . import tools


def vox2MNI(vox, affine):
    """
    Voxel coordinates transformed to MNI coordinates
    ------------------------------------------------
    Parameters:
        vox: voxel coordinates
        affine: affine matrix
    Return:
        mni
    """
    vox = np.append(vox, 1)
    mni = np.dot(affine, vox)
    return mni[:3]

def MNI2vox(mni, affine):
    """
    MNI coordintes transformed to voxel coordinates
    ----------------------------------------------
    Parameters:
        mni: mni coordinates
        affine: affine matrix
    Return:
        vox
    """
    mni = np.append(mni,1)
    vox = np.dot(mni, np.linalg.inv(affine.T))
    return vox[:3]


def get_masksize(mask):
    """
    Compute mask size
    -------------------------------------
    Parameters:
        mask: mask.
    Return:
        masksize: mask size of each roi
    """
    labels = np.unique(mask)[1:]
    if mask.ndim == 3:
        mask = np.expand_dims(mask, axis = 3)
    masksize = np.empty((mask.shape[3], int(np.max(labels))))
    for i in range(mask.shape[3]):
        for j in range(int(np.max(labels))):
            if np.any(mask[...,i] == j+1):
                masksize[i, j] = np.sum(mask[...,i] == j+1)
            else:
                masksize[i, j] = np.nan
    return masksize

def get_signals(atlas, mask, method = 'mean', labelnum = None):
    """
    Extract roi signals of atlas
    --------------------------------------
    Parameters:
        atlas: atlas
        mask: masks. Different roi labelled differently
        method: 'mean', 'std', 'ste'(standard error), 'max', 'voxel', etc.
        labelnum: Mask's label numbers, by default is None. Add this parameters for group analysis
    Return:
        signals: nroi for activation data
                 resting signal x roi for resting data
    """
    labels = np.unique(mask)[1:]
    if labelnum is None:
        labelnum = int(np.max(labels))
    signals = []
    if method == 'mean':
        calfunc = np.nanmean
    elif method == 'std':
        calfunc = np.nanstd
    elif method == 'ste':
        calfunc = tools.ste
    elif method == 'max':
        calfunc = np.max
    elif method == 'voxel':
        calfunc = np.array
    else:
        raise Exception('Method contains mean or std or peak')
    for i in range(labelnum):
        loc_raw = np.where(mask == (i+1))
        roiloc = zip(loc_raw[0], loc_raw[1], loc_raw[2])
        roisignal = [atlas[roiloc[i]] for i in range(len(roiloc))]
        if np.any(roisignal):
            signals.append(roisignal)
        else:
            signals.append([np.nan])
    # return signals    
    return [calfunc(sg) for sg in signals]

def get_coordinate(atlas, mask, method = 'peak', labelnum = None):
    """
    Extract peak/center coordinate of rois
    --------------------------------------------
    Parameters:
        atlas: atlas
        mask: roi mask.
        method: 'peak' or 'center'
        labelnum: mask label numbers in total, by default is None, set parameters if you want to do group analysis
    Return:
        coordinates: nroi x 3 for activation data, It's voxel coordinate
                     Note that do not extract coordinate of resting data
    """
    labels = np.unique(mask)[1:]
    if labelnum is None:
        labelnum = np.max(labels)
    coordinate = np.empty((int(labelnum), 3))

    extractpeak = lambda x: np.unravel_index(x.argmax(), x.shape)
    extractcenter = lambda x: np.mean(np.transpose(np.nonzero(x)))

    if method == 'peak':
        calfunc = extractpeak
    elif method == 'center':
        calfunc = extractcenter
    else:
        raise Exception('Method contains peak or center')
    for i in range(labelnum):
        roisignal = atlas*(mask == (i+1))
        if np.any(roisignal):
            coordinate[i,:] = calfunc(roisignal)
        else:
            coordinate[i,:] = np.array([np.nan, np.nan, np.nan])
    return coordinate

def make_pm(mask, meth = 'all'):
    """
    Make probabilistic map
    ------------------------------
    Parameters:
        mask: mask
        meth: 'all' or 'part'. 
              all, all subjects are taken into account
              part, part subjects are taken into account
    Return:
        pm = probabilistic map
    """
    if mask.ndim != 4:
        raise Exception('Masks should be a 4D nifti file contains subjects')
    labels = np.unique(mask)[1:]
    pm = np.empty((mask.shape[0], mask.shape[1], mask.shape[2], labels.shape[0]))
    if meth == 'all':
        for i in range(labels.shape[0]):
            pm[..., i] = np.mean(mask == labels[i], axis = 3)
    elif meth == 'part':
        for i in range(labels.shape[0]):
            mask_i = mask == labels[i]
            subj = np.any(mask_i, axis = (0,1,2))
            pm[..., i] = np.mean(mask_i[..., subj], axis = 3)
    else:
        raise Exception('method not supported')
    return pm
        
def make_mpm(pm, threshold):
    """
    Make maximum probabilistic map (mpm)
    ---------------------------------------
    Parameters:
        pm: probabilistic map
        threshold: threholds to mask probabilistic maps
    Return:
        mpm: maximum probabilisic map
    """
    pm_temp = np.empty((pm.shape[0], pm.shape[1], pm.shape[2], pm.shape[3]+1))
    pm_temp[..., range(1, pm.shape[3]+1)] = pm 
    pm_temp[pm_temp < threshold] = 0
    mpm = np.argmax(pm_temp, axis=3)
    return mpm    

def sphere_roi(voxloc, radius, value, datashape = (91,109,91), data = None):
    """
    Generate a sphere roi which centered in (x,y,z)
    Parameters:
        voxloc: (x,y,z), center vox of spheres
        radius: radius (unit: vox), note that it's a list
        value: label value 
        datashape: data shape, by default is (91,109,91)
        data: Add sphere roi into your data, by default data is an empty array
    output:
        data: sphere roi image data
        loc: sphere roi coordinates
    """
    if data is not None:
        try:
            if data.shape != datashape:
                raise Exception('Data shape is not consistent with parameter datashape')
        except AttributeError:
            raise Exception('data type should be np.array')
    else:
        data = np.zeros(datashape)

    loc = []
    for n_x in range(int(voxloc[0]-radius[0]), int(voxloc[0]+radius[0]+1)):
        for n_y in range(int(voxloc[1]-radius[1]), int(voxloc[1]+radius[1]+1)):
            for n_z in range(int(voxloc[2]-radius[2]), int(voxloc[2]+radius[2]+1)):
                n_coord = np.array((n_x, n_y, n_z))
                coord = np.array((voxloc[0], voxloc[1], voxloc[2]))
                minus = coord - n_coord
                if (np.square(minus) / np.square(np.array(radius)).astype(np.float)).sum()<=1:
                    try:
                        data[n_x, n_y, n_z] = value
                        loc.append([n_x,n_y,n_z])
                    except IndexError:
                        pass
    loc = np.array(loc)
    return data, loc

def region_growing(image, coordinate, voxnumber):
    """
    Region growing
    Parameters:
        image: nifti data
        coordinate: raw coordinate
        voxnumber: max growing number
    Output:
        rg_image: growth region image
        loc: region growth location
    """
    loc = []
    nt = voxnumber
    tmp_image = np.zeros_like(image)
    rg_image = np.zeros_like(image)
    image_shape = image.shape
    
    x = coordinate[0]
    y = coordinate[1]
    z = coordinate[2]

    # ensuring the coordinate is in the image
    # inside = (x >= 0) and (x < image_shape[0]) and (y >= 0) and \
    #          (y <= image_shape[1]) and (z >= 0) and (z < image_shape[2])
    # if inside is not True:
    #     print "The coordinate is out of the image range"
    #     return False

    # initialize region_mean and region_size
    region_mean = image[x,y,z]
    region_size = 0
    
    # initialize neighbour_list with 10000 rows 4 columns
    neighbour_free = 10000
    neighbour_pos = -1
    neighbour_list = np.zeros((neighbour_free, 4))

    # 26 direct neighbour points
    neighbours = [[1,0,0],\
                  [-1,0,0],\
                  [0,1,0],\
                  [0,-1,0],\
                  [0,0,-1],\
                  [0,0,1],\
                  [1,1,0],\
                  [1,1,1],\
                  [1,1,-1],\
                  [0,1,1],\
                  [-1,1,1],\
                  [1,0,1],\
                  [1,-1,1],\
                  [-1,-1,0],\
                  [-1,-1,-1],\
                  [-1,-1,1],\
                  [0,-1,-1],\
                  [1,-1,-1],\
                  [-1,0,-1],\
                  [-1,1,-1],\
                  [0,1,-1],\
                  [0,-1,1],\
                  [1,0,-1],\
                  [1,-1,0],\
                  [-1,0,1],\
                  [-1,1,0]]
    while region_size < nt:
        # (xn, yn, zn) stored direct neighbour of seed point
        for i in range(6):
            xn = x + neighbours[i][0]
            yn = y + neighbours[i][1]
            zn = z + neighbours[i][2]
            
            inside = (xn >= 0) and (xn < image_shape[0]) and (yn >=0) and \
                 (yn < image_shape[1]) and (zn >= 0) and (zn < image_shape[2])
            # ensure the original flag 0 is not changed
            if inside and tmp_image[xn, yn, zn] == 0:
                neighbour_pos = neighbour_pos + 1
                neighbour_list[neighbour_pos] = [xn, yn, zn, image[xn,yn,zn]]
                tmp_image[xn,yn,zn] = 1
 
        # ensure there's enough space to store neighbour_list
        if (neighbour_pos + 100 > neighbour_free):
            neighbour_free += 10000
            new_list = np.zeros((10000,4))
            neighbour_list = np.vstack((neighbour_list, new_list))
        
        # the distance between every neighbour point value to new region mean value
        distance = np.abs(neighbour_list[:neighbour_pos+1, 3] - np.tile(region_mean, neighbour_pos+1))

        # chose min distance point
        index = distance.argmin()

        # mark the new region point with 2 and update new image
        tmp_image[x, y, z] = 2
        rg_image[x, y, z] = image[x, y, z]
        loc.append([x,y,z])
        region_size+=1
        
        # (x,y,z) the new seed point
        x = neighbour_list[index][0]
        y = neighbour_list[index][1]
        z = neighbour_list[index][2]
        
        # update region mean value
        region_mean = (region_mean*region_size + neighbour_list[index, 3])/(region_size + 1)
         
        # remove the seed point from neighbour_list
        neighbour_list[index] = neighbour_list[neighbour_pos]
        neighbour_pos -= 1

    loc = np.array(loc)
    return rg_image, loc

def peakn_location(data, ncluster = 5, rgsize = 10, reverse = False):
    """
    Using region growth to extract highest/lowest clusters
    --------------------------------------
    Parameters:
        data: raw data
        ncluster: cluster numbers by using information of data values
        rgsize: region growth size (voxel), constraint neighbouring voxels
        reverse: if True, get locations start from the largest values
                 if False, start from the lowest values
    Return:
        nth_loc: list of locations
    """
    if reverse is True:
        filterdata = np.argmin
    else:
        filterdata = np.argmax
    median_data = np.median(data)
    nth_loc = []
    for i in range(ncluster):
        temploc = np.unravel_index(filterdata(data), data.shape)
        nth_loc.append(temploc)
        tempdata, loc_rg = region_growing(data, temploc, rgsize)
        for j in loc_rg:
            data[j[0], j[1], j[2]] = median_data
    return nth_loc, tempdata

class ImageCalculator(object):
    def __init__(self):
        pass

    def merge4D(self, rawdatapath, outdatapath, outname, issave = True):    
        """
        Merge 3D images together
        --------------------------------------
        Parameters:
            rawdatapath: raw data path. Need to be a list contains path of each image
            outdatapath: output path.
            outname: output data name.
            issave: save data or not, by default is True
        Return:
            outdata: merged file
        """
        if isinstance(rawdatapath, np.ndarray):
            rawdatapath = rawdatapath.tolist()
        header = nib.load(rawdatapath[0]).get_header()
        datashape = nib.load(rawdatapath[0]).get_data().shape
        nsubj = len(rawdatapath)
        outdata = np.zeros((datashape[0], datashape[1], datashape[2], nsubj))
        for i in range(nsubj):
            if os.path.exists(rawdatapath[i]):
                outdata[...,i] = nib.load(rawdatapath[i]).get_data()
            else:
                raise Exception('File may not exist of {0}'.format(rawdatapath[i]))
        if issave is True:
            img = nib.Nifti1Image(outdata, None, header)
            if outdatapath.split('/')[-1].endswith('.nii.gz'):
                nib.save(img, outdatapath)
            else:
                outdatapath_new = os.path.join(outdatapath, outname)
                nib.save(img, outdatapath_new)
        return outdata

    def decompose_img(self, imagedata, header, outpath, outname=None):
        """
        Decompose 4D image into multiple 3D image to make it easy to show or analysis
        --------------------------------------------------------------------
        Parameters:
            imagedata: 4D image, the 4th dimension is the axis to decompose
            header: image header
            outpath: outpath
            outname: outname, should be a list. By default is None, system will distribute name into multiple 3D images automatically. If you want to generate images with meaningful name, please assign a list.
        Output:
            save outdata into multiple files
        Example:
            >>> imccls = ImageCalculator() 
            >>> imccls.decompose_img(imagedata, header, outpath)
        """
        assert np.ndim(imagedata) == 4, 'imagedata must be a 4D data'
        filenumber = imagedata.shape[3]
        if outname is None:
            digitname = range(1,filenumber+1,1)
            outname = [str(i) for i in digitname]
        else:
            assert len(outname) == filenumber, 'length of outname unequal to length of imagedata'
        for i,e in enumerate(outname):
            outdata = imagedata[...,i]
            img = nib.Nifti1Image(outdata, None, header)
            nib.save(img, os.path.join(outpath, e+'.nii.gz'))

    def combine_data(self, image1, image2, method = 'and'):
        """
        Combined data for 'and', 'or'
        ------------------------------------------
        Parameters:
            image1: dataset of the first image
            image2: dataset of the second image
            method: 'and' or 'or'
        """
        if (isinstance(image1, str) & isinstance(image2, str)):
            image1 = nib.load(image1).get_data()
            image2 = nib.load(image2).get_data()
        labels = np.unique(np.concatenate((np.unique(image1), np.unique(image2))))[1:]
        outdata = np.empty((image1.shape[0], image1.shape[1], image2.shape[2], labels.size))
        for i in range(labels.size):
            tempimage1 = copy.copy(image1)
            tempimage2 = copy.copy(image2)
            tempimage1[tempimage1 != labels[i]] = 0
            tempimage1[tempimage1 == labels[i]] = 1
            tempimage2[tempimage2 != labels[i]] = 0
            tempimage2[tempimage2 == labels[i]] = 1
            tempimage1.astype('bool')
            tempimage2.astype('bool')
            if method == 'and':
                tempimage = tempimage1 * tempimage2
            elif method == 'or':
                tempimage = tempimage1 + tempimage2
            else:
                raise Exception('Method support and, or now')
            outdata[...,i] = labels[i]*tempimage
        return outdata

    def relabel_roi(self, roiimg):
        """
        Relabel roi image, convert discontinous label image into label-continue image
        --------------------------
        Parameters:
            roiimg: roi image

        Output:
            relabelimg: relabeling image with continous label sequence
            corr_label: the correspond relationship between original label and new label
       
        Example:
            >>> relabelimg, corr_label = m.relabel_roi(roiimg)
        """
        relabelimg = np.zeros_like(roiimg)
        rawlabel = np.sort(np.unique(roiimg))[1:].astype('int')
        newlabel = np.array(range(1, len(rawlabel)+1)).astype('int')
        corr_label = zip(rawlabel, newlabel)
        for i,e in enumerate(rawlabel):
            relabelimg[roiimg==e] = newlabel[i]
        return relabelimg, corr_label

class ExtractSignals(object):
    def __init__(self, atlas, regions = None):
        masksize = get_masksize(atlas)
        
        self.atlas = atlas
        if regions is None:
            self.regions = masksize.shape[1]
        else:
            if isinstance(regions, int):
                self.regions = regions
            else:
                self.regions = len(regions)
        self.masksize = masksize

    def getsignals(self, targ, method = 'mean'):
        """
        Get measurement signals from target image by mask atlas.
        -------------------------------------------
        Parameters:
            targ: target image
            method: 'mean' or 'std', 'ste'(standard error), 'max' or 'voxel'
                    roi signal extraction method
        Return:
            signals: extracted signals
        """
        if targ.ndim == 3:
            targ = np.expand_dims(targ, axis = 3)
        signals = []
        
        for i in range(targ.shape[3]):
            if self.atlas.ndim == 3:
                signals.append(get_signals(targ[...,i], self.atlas, method, self.regions))
            elif self.atlas.ndim == 4:
                signals.append(get_signals(targ[...,i], self.atlas[...,i], method, self.regions))
        self.signals = np.array(signals)
        return np.array(signals)

    def getcoordinate(self, targ, size = [2,2,2], method = 'peak'):
        """
        Get peak coordinate signals from target image by mask atlas.
        -----------------------------------------------------------
        Parameters:
            targ: target image
            size: voxel size
            method: 'peak' or 'center'
                    coordinate extraction method
        """
        if targ.ndim == 3:
            targ = np.expand_dims(targ, axis = 3)
        coordinate = np.empty((targ.shape[3], self.regions, 3))

        for i in range(targ.shape[3]):
            if self.atlas.ndim == 3:
                coordinate[i, ...] = get_coordinate(targ[...,i], self.atlas, size, method, self.regions)
            elif self.atlas.ndim == 4:
                coordinate[i, ...] = get_coordinate(targ[...,i], self.atlas[...,i], size, method, self.regions)
        self.coordinate = coordinate
        return coordinate

    def getdistance_array2point(self, targ, pointloc, size = [2,2,2], coordmeth = 'peak', distmeth = 'euclidean'):
        """
        Get distance from each coordinate to a specific location
        -------------------------------------------------------
        Parameters:
            targ: target image
            pointloc: location of a specific voxel
            size: voxel size
            coordmeth: 'peak' or center
                       coordinate extraction method
            distmeth: distance method
        """
        if not hasattr(self, 'coordinate'):
            self.coordinate = get_coordinate(targ, self.atlas, size, coordmeth)
        dist_point = np.empty((self.coordinate.shape[0], self.coordinate.shape[1]))
        pointloc = np.array(pointloc)
        if pointloc.shape[0] == 1:
            pointloc = np.tile(pointloc, [dist_point.shape[1],1])
        for i in range(dist_point.shape[0]):
            for j in range(dist_point.shape[1]):
                if not isinstance(pointloc[j], np.ndarray):
                    raise Exception('pointloc should be 2 dimension array or list')
                dist_point[i,j] = tools.calcdist(self.coordinate[i,j,:], pointloc[j], distmeth)
        self.dist_point = dist_point
        return dist_point

class MakeMasks(object):
    def __init__(self):
        pass

    def makepm(self, atlas, meth = 'all'):
        """
        Make probabilistic maps
        ------------------------------
        Parameters:
            atlas: atlas mask
            meth: 'all' or 'part'
            maskname: output mask name, by default is 'pm.nii.gz'
        Return:
            pm
        """
        pm = make_pm(atlas, meth)
        self._pm = pm
        return pm

    def makempm(self, threshold, pmfile = None):
        """
        Make maximum probabilistic maps
        --------------------------------
        Parameters:
            threshold: mpm threshold
            maskname: output mask name. By default is 'mpm.nii.gz'
        """
        if pmfile is not None:
            self._pm = pmfile
        if self._pm is None:
            raise Exception('please execute makepm first or give pmfile in this method')
        mpm = make_mpm(self._pm, threshold)
        return mpm       
    
    def makemask_sphere(self, voxloc, radius, atlasshape = (91,109,91)):
        """
        Make mask by means of roi sphere
        -------------------------------------------------
        Parameters:
            voxloc: peak voxel locations of each region
                    Note that it's a list
            radius: sphere radius, such as [3,3,3],etc.
            atlasshape: atlas shape
        """ 
        spheremask = np.zeros(atlasshape)
        for i, e in enumerate(voxloc):
            spheremask, loc = sphere_roi(e, radius, i+1, datashape = atlasshape, data = spheremask)
        return spheremask, loc

    def makemask_rgrowth(self, valuemap, coordinate, voxnum):
        """
        Make masks using region growth method
        -----------------------------------
        Parameters:
            valuemap: nifti data. Z map, cope map, etc.
            coordinate: region growth origin points
            voxnum: voxel numbers, integer or list
        -----------------------------------
        Example:
            >>> outdata = INS.makemask_rgrowth(data, [[22,23,31], [22,22,31], [54,55,67]], 15)
        """
        import warnings
        warnings.simplefilter('always', UserWarning)       
 
        if isinstance(voxnum, int):
            voxnum = [voxnum]
        if len(voxnum) == 1:
            voxnum = voxnum*len(coordinate)
        if len(voxnum)!=len(coordinate):
            raise Exception('Voxnum length unequal to coodinate length.')
        rgmask = np.zeros_like(valuemap)
        for i,e in enumerate(coordinate):
            rg_image, loc = region_growing(valuemap, e, voxnum[i])
            rg_image[rg_image!=0] = i+1
            if ~np.any(rgmask[rg_image!=0]):
                # all zeros
                rgmask += rg_image
            else:
                warnings.warn('coordinate {} has been overlapped! Label {} will missing'.format(e, i+1))
        return rgmask

def data_preprocess(data, outlier_method = None, outlier_range = [-3,3], mergehemi = None):
    """
    Pipline to merge hemisphere and do outlier removed.
    ---------------------------------------------------------
    Parameters:
        data: raw data. Notes that when the dimension is 1, data means regions. When the dimension is 2, data is the form of nsubj*regions. When the dimension is 3, data is the form of timeseries*regions*nsubj.
        outlier_method: 'iqr' or 'std' or 'abs'. By default is None
        outlier_range: outlier standard threshold
        mergehemi: merge hemisphere or not. By default is False. Input bool expression to indicate left or right factor. True means left hemisphere, False means right hemisphere
    Output:
        n_removed: outlier_numbers
        residue_data: output data

    Example:
        >>> a = np.array([[1,2,3,4],[5,6,7,8]])
        >>> b = array([True,True,False,False], dtype=bool)
        >>> n_removed, residue_data = dataprocess(a,mergehemi = b)
    """
    if mergehemi is not None:
        if not len(mergehemi[mergehemi]) == len(mergehemi[~mergehemi]):
            raise Exception("length of left data should equal to right data")

    if data.ndim == 1:
        data = data[np.newaxis,...]
    if data.ndim == 2:
        data = data[...,np.newaxis]
    if data.ndim != 3:
        raise Exception('data dimensions should be 2 or 3!')
    if mergehemi is None:
        data_comb = data
        n_removed = np.empty((data.shape[1], data.shape[2]))
        data_removed = np.zeros_like(data)
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                n_removed[i,j], data_removed[:,i,j] = tools.removeoutlier(data_comb[:,i,j], meth = outlier_method, thr = outlier_range)
    else:
        if not mergehemi.dtype == np.bool:
            mergehemi = mergehemi.astype('bool')
        n_removed = np.empty((data.shape[1]/2, data.shape[2]))
        data_comb = np.empty((data.shape[0], data.shape[1]/2, data.shape[2]))
        data_removed = np.empty((data.shape[0], data.shape[1]/2, data.shape[2]))
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                data_comb[i,:,j] = tools.hemi_merge(data[i,mergehemi,j], data[i,~mergehemi,j])
        for i in range(data.shape[1]/2):
            for j in range(data.shape[2]):
                n_removed[i,j], data_removed[:,i,j] = tools.removeoutlier(data_comb[:,i,j], meth = outlier_method, thr = outlier_range)
    if n_removed.shape[-1] == 1:
        n_removed = n_removed[...,0]
    if data_removed.shape[-1] == 1:
        data_removed = data_removed[...,0]
    return n_removed, data_removed

class ComPatternMap(object):
    def __init__(self, data, regions, outlier_method = None, outlier_range = [-3, 3], mergehemi = None):
        """
        Parameters:
            data: raw data. It could be 2D or 3D data.
                  2D data is activation data. Which is the form of nsubj*regions
                  3D data is roi resting data. Which is the form of timeseries*regions*nsubj
            regions: region names
            outlier_method: criterion of outlier removed, 'iqr' or 'std' or 'abs'
            outlier_range: outlier range
            mergehemi: whether merge signals between hemispheres or not. Input bool expression to indicate left or right factor. True means left hemisphere, False means right hemisphere                      
        """
        if not isinstance(regions, list):
            regions = regions.tolist()
        
        n_removed, data_removed = data_preprocess(data, outlier_method, outlier_range, mergehemi)
        
        self.regions = regions
        self.data_removed = data_removed
        self.n_removed = n_removed
        self.mergehemi = mergehemi

    def patternmap(self, meth = 'correlation'):
        if self.data_removed.ndim == 2:
            self.data_removed = np.expand_dims(self.data_removed, axis = 2)
        distance = []
        corrmatrix = np.empty((self.data_removed.shape[1], self.data_removed.shape[1], self.data_removed.shape[2]))
        corrpval = np.empty((self.data_removed.shape[1], self.data_removed.shape[1], self.data_removed.shape[2]))
        for i in range(self.data_removed.shape[2]):
            cleandata = tools.listwise_clean(self.data_removed[...,i])
            corrmatrix[...,i], corrpval[...,i] = tools.pearsonr(cleandata.T, cleandata.T)
            distance.append(pdist(cleandata.T, meth))
            print('subject {} finished'.format(i+1))
        distance = np.array(distance)
        return corrmatrix, distance

def dice_map_evaluate(data1, data2, filename = 'dice.pkl'):
    """
    Evaluate delineation accuracy by dice coefficient
    -------------------------------------------
    Parameters:
        data1, data2: raw data
    Output:
        dice: dice coefficient
    """
    if data1.ndim != data2.ndim:
        raise Exception('Two raw data need have the same dimensions')
    label1 = np.unique(data1)[1:]
    label2 = np.unique(data2)[1:]
    label = np.sort(np.unique(np.concatenate((label1, label2))))
    if data1.ndim == 3:
        data1 = np.expand_dims(data1, axis = 3)
    if data2.ndim == 3:
        data2 = np.expand_dims(data2, axis = 3)
    dice = []
    for i in range(data1.shape[3]):
        dice.append(calc_overlap(data1[...,i], data2[...,i], label1, label2))
    dice = np.array(dice)
    return dice

class PositionRelationship(object):
    """
    Class for measure position relationship between images
    Pay attention that images should be labelled image!
    Note that we recommend you giving numbers of roi so to avoid mess.
    ---------------------------------------------------
    Parameters:
        roimask: roi label data
        roinumber: the number of roi in your label data
    """
    def __init__(self, roimask, roinumber = None):
        try:
            roimask.shape
        except AttributeError:
            roimask = nib.load(roimask).get_data()
        finally:
            self._roimask = roimask
        self._masklabel = np.unique(roimask)[1:]
        if roinumber is None:
            self._roinumber = self._masklabel.size
        else:
            self._roinumber = roinumber

    def template_overlap(self, template, para='percent', tempnumber = None):
        """
        Compute overlap between target data and template 
        -----------------------------------------
        Parameters:
            template: template image, 
            para: index call for computing
                  'percent', overlap #voxels/target region #voxels
                  'amount', overlap #voxels
                  'dice', 2*(intersection)/union
            tempnumber: template label number, set in case miss label in specific subjects
        Output:
            overlaparray, overlap array(matrix) in two images(target & template)
            uni_tempextlbl, overlap label within template(Note that label not within target) 
        """
        try:
            template.shape
        except AttributeError:
            template = nib.load(template).get_data()
            print('Template should be an array')
        if template.shape != self._roimask.shape:
            raise Exception('template should have the same shape with target data')
        templabel = np.unique(template)[1:]
        if tempnumber is not None:
            templabel = np.array(range(1,tempnumber+1))
        overlaparray = np.empty((templabel.size, self._roinumber))
        
        roiloc = np.transpose(np.nonzero(self._roimask))
        tup_roiloc = map(tuple, roiloc)
        tempextlabel_all = np.array([template[i] for i in tup_roiloc])
        roiextlabel_all = np.array([self._roimask[i] for i in tup_roiloc])
        tempextlabel = np.delete(tempextlabel_all, np.where(tempextlabel_all==0))
        roiextlabel = np.delete(roiextlabel_all, np.where(tempextlabel_all==0))
        uni_tempextlbl = np.unique(tempextlabel)
        for i, vali in enumerate(templabel):
            for j, valj in enumerate(range(1, 1+self._roinumber)):
                if para == 'percent':
                    try:
                        overlaparray[i,j] = 1.0*tempextlabel[(tempextlabel == vali)*(roiextlabel == valj)].size/self._roimask[self._roimask == valj].size
                    except ZeroDivisionError:
                        overlaparray[i,j] = np.nan
                elif para == 'amount':
                    overlaparray[i,j] = tempextlabel[(tempextlabel == vali)*(roiextlabel == valj)].size
                elif para == 'dice':
                    try:
                        overlaparray[i,j] = 2.0*tempextlabel[(tempextlabel == vali)*(roiextlabel == valj)].size/(template[template == vali].size + self._roimask[self._roimask == valj].size)
                    except ZeroDivisionError:
                        overlaparray[i,j] = np.nan
                else:
                    raise Exception("para should be 'percent', 'amount' or 'dice', please retype")
        return overlaparray, uni_tempextlbl

    def roidistance(self, targdata, extloc = 'peak', metric = 'euclidean'):
        """
        Compute distance between ROIs which contains in a mask
        ---------------------------------------------
        Input:
            targdata: target nifti data, pay attention that this data is not labelled data
            extloc: 'peak' or 'center' in extraction of coordinate
            metric: methods for calculating distance
        """
        try:
            targdata.shape
        except AttributeError:
            targdata = nib.load(targdata).get_data()
            print('targdata should be an array')
        if self._roimask.shape != targdata.shape:
            raise Exception('targdata shape should have the save shape as target data')

        peakcoord = get_coordinate(targdata, self._roimask, method = extloc, labelnum = self._roinumber)
        dist_array = np.empty((peakcoord.shape[0], peakcoord.shape[0]))
        for i in range(peakcoord.shape[0]):
            for j in range(peakcoord.shape[0]):
                dist_array[i,j] = tools.calcdist(peakcoord[i,:], peakcoord[j,:], metric = metric)
        return dist_array

class PatternSimilarity(object):
    """
    Compute connectivity between vox2vox, roi2vox and roi2roi in whole brain
    By default data consist of (nx,ny,nz,nt)
    In vox2vox, do pearson connectivity in a seed point(1 vox) with other voxels in whole brain
    In roi2vox, do pearson connectivity in one roi (average signals of roi) with other voxels
                in whole brain
    In roi2roi, do pearson connectivity between rois (average signals of rois)
    ------------------------------------------------------------------------
    Parameters:
        imgdata: image data with time/task series. Note that it's a 4D data
        transform_z: By default is False, if the output corrmatrix be z matrix, please flag it as True
    Example:
        >>> m = PatternSimilarity(imgdata, transform_z = True)
    """
    def __init__(self, imgdata, transform_z = False):
        try:
            assert imgdata.ndim == 4
        except AssertionError:
            raise Exception('imgdata should be 4 dimensions!')
        self._imgdata = imgdata
        self._transform_z = transform_z

    def vox2vox(self, vxloc):
        """
        Compute connectivity between a voxel and the other voxels.
        ----------------------------------------------------
        Parameters:
            vxloc: seed voxel location. voxel coordinate.
        Output:
            corrmap: corr values map, rmap or zmap
            pmap: p values map
        Example:
            >>> corrmap, pmap = m.vox2vox(vxloc)
        """
        rmap = np.zeros(self._imgdata.shape[:3])
        pmap = np.zeros_like(rmap)
        vxseries = self._imgdata[vxloc[0], vxloc[1], vxloc[2], :]
        vxseries = np.expand_dims(vxseries, axis=1).T
        for i in range(self._imgdata.shape[0]):
            for j in range(self._imgdata.shape[1]):
                r_tmp, p_tmp = tools.pearsonr(vxseries, self._imgdata[i,j,...])
                rmap[i,j,:], pmap[i,j,:] = r_tmp.flatten(), p_tmp.flatten()
            print('{}% finished'.format(100.0*i/self._imgdata.shape[0]))
        # solve problems as output of nifti data
        # won't affect fdr corrected result
        rmap[np.isnan(rmap)] = 0
        pmap[pmap == 1] = 0
        if self._transform_z is False:
            corrmap = rmap
        else:
            print('Perform the Fisher r-to-z transformation')
            corrmap = tools.r2z(rmap)
        return corrmap, pmap

    def roi2vox(self, roimask):
        """
        Compute connectivity between roi and other voxels
        --------------------------------------------------
        Parameters:
            roimask: roi mask. Contain one roi only, note!
        Output:
            corrmap: corr values map
            pmap: p values map
        Example:
            >>> corrmap, pmap = m.roi2vox(roimask)
        """
        roilabel = np.unique(roimask)[1:]
        assert len(roilabel) == 1
        rmap = np.zeros(self._imgdata.shape[:3])
        pmap = np.zeros_like(rmap)
        roiseries, roiloc = _avgseries(self._imgdata, roimask, roilabel[0])
        roiseries = np.expand_dims(roiseries, axis=1).T
        for i in range(self._imgdata.shape[0]):
            for j in range(self._imgdata.shape[1]):
                r_tmp, p_tmp = tools.pearsonr(roiseries, self._imgdata[i,j,...])
                rmap[i,j,:], pmap[i,j,:] = r_tmp.flatten(), p_tmp.flatten()
            print('{}% finished'.format(100.0*i/self._imgdata.shape[0]))
        rmap[np.isnan(rmap)] = 0
        pmap[pmap == 1] = 0
        if self._transform_z is False:
            corrmap = rmap
        else:
            print('Perform the Fisher r-to-z transformation')
            corrmap = tools.r2z(rmap)
        return corrmap, pmap

    def roi2roi(self, roimask):
        """
        Compute connectivity between rois
        ---------------------------------------
        Parameters:
            roimask: roi mask. Need to contain over 1 rois
        Output:
            corrmap: corr values map
            pmap: p values map
        Example:
            >>> corrmap, pmap = m.roi2roi(roimask)
        """
        avgsignals = self.roiavgsignal(roimask)
        rmap, pmap = tools.pearsonr(avgsignals, avgsignals)
        if self._transform_z is False:
            corrmap = rmap
        else: 
            print('Perform the Fisher r-to-z transformation')
            corrmap = tools.r2z(rmap)
        return corrmap, pmap

    def roiavgsignal(self, roimask):
        """
        Extract within-roi average signal from roimask
        ------------------------------------------------
        Parameters:
            roimask: roi mask
        Output:
            avgsignal: average signals
        Example:
            >>> avgsignal = m.roiavgsignal(roimask)
        """
        roimxlb = np.sort(np.unique(roimask)[1:])[-1]
        avgsignal = np.empty((roimxlb, self._imgdata.shape[3]))
        for i in range(int(roimxlb)):
            avgsignal[i, :], roiloc = _avgseries(self._imgdata, roimask, i+1)
        return avgsignal

def _avgseries(imgdata, roimask, label):
    """
    Extract average series from 4D image data of a specific label
    """
    roii, roij, roik = np.where(roimask == label)
    roiloc = zip(roii, roij, roik)
    return np.nanmean([imgdata[i] for i in roiloc], axis=0), roiloc

class MVPA(object):
    """
    Simple class for MVPA
    Class contains:
        space mvpa for correlation between sphere and sphere: mvpa_space_sph2sph
        space mvpa for roi between roi and roi: mvpa_space_roi2roi
        space mvpa for global brain: searchlight (sph to sph for global brain): mvpa_space_searchlight

        timeseries mvpa
        [Unfinished yet]         
    -----------------------------------
    Parameters:
        imgdata: nifti data
    Example:
        >>> mvpacls = MVPA(imgdata)  
    """
    def __init__(self, imgdata):
        self._imgdata = imgdata
        self._imgshape = imgdata.shape

    def mvpa_space_sph2sph(self, voxloc1, voxloc2, radius = [2,2,2]):
        """
        MVPA method for correlation between sphere to sphere
        ---------------------------------------------
        Parameters:
            voxloc1, voxloc2: voxel location, used for generate spheres
            radius: sphere radius
        Output:
            r: correlation coefficient of signals between two spheres
            p: significant level
        """
        assert np.ndim(self._imgdata) == 3, "Dimension of inputdata, imgdata, should be 3 in space mvpa"
        sphereroi1, _ = sphere_roi(voxloc1, radius, 1, datashape = self._imgshape)
        signals1 = get_signals(self._imgdata, sphereroi1, 'voxel')[0]
        sphereroi2, _ = sphere_roi(voxloc2, radius, 1, datashape = self._imgshape)
        signals2 = get_signals(self._imgdata, sphereroi2, 'voxel')[0]
        r, p = stats.pearsonr(signals1, signals2)        
        return r, p
   
    def mvpa_space_roi2roi(self, roimask1, roimask2):
        """
        MVPA method for correlation between roi to roi
        ---------------------------------------------
        Parameters:
            roimask1, roimask2: roimasks
                                I haven't considered method how to check two roi have the same shape
        Output:
            r: correlation coefficient of signals between two spheres
            p: significant level
        """
        assert np.ndim(self._imgdata) == 3, "Dimension of inputdata, imgdata, should be 3 in space mvpa"
        signal1 = get_signals(self._imgdata, roimask1, 'voxel')[0]
        signal2 = get_signals(self._imgdata, roimask2, 'voxel')[0]
        r, p = stats.pearsonr(signal1, signal2)
        return r, p
 
    def mvpa_space_searchlight(self, voxloc, radius = [2,2,2], thr = 1e-3):
        """
        Searchlight method search in global brain
        Note that I'm not quite sure whether the details are right
        In future maybe I need to fix details
        --------------------------------------------
        Parameters:
            voxloc: voxel location
            radius: sphere radius, by default is [2,2,2]
            thr: threshold values of raw activation values
                 higher value means smaller checking range, with smaller computational time
        Output:
            rdata: r maps
            pdata: p maps
        Example:
            >>> rdata, pdata = mvpa_space_searchlight(voxloc)
        """   
        assert np.ndim(self._imgdata) == 3, "Dimension of inputdata, imgdata, should be 3 in space mvpa"
        rdata = np.zeros_like(self._imgdata)
        pdata = np.zeros_like(self._imgdata)
        sphere_org, _ = sphere_roi(voxloc, radius, 1, self._imgshape)
        signal_org = get_signals(self._imgdata, sphere_org, 'voxel')[0]
        for i in range(self._imgshape[0]):
            for j in range(self._imgshape[1]):
                for k in range(self._imgshape[2]):
                    if np.abs(self._imgdata[i,j,k]) < thr:
                        continue
                    else:
                        sphere_dest, _ = sphere_roi([i,j,k], radius, 1, self._imgshape)
                        if sphere_dest[sphere_dest!=0].shape[0]!=sphere_org[sphere_org!=0].shape[0]:
                            continue
                        signal_dest = get_signals(self._imgdata, sphere_dest, 'voxel')[0]
                        rdata[i,j,k], pdata[i,j,k] = stats.pearsonr(signal_org, signal_dest)
            print('{}% finished'.format(100.0*i/91))
        return rdata, pdata

