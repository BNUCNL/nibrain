# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import numpy as np
from . import tools
import copy
import random

def get_masksize(mask, labelnum = None):
    """
    Compute mask size in surface space
    
    Parameters:
    ----------
    mask: label image (mask)
    labelnum: mask's label number, use for group analysis

    Return:
    --------
    masksize: mask size of each roi

    Example:
    --------
    >>> masksize = get_masksize(mask)
    """
    if mask.ndim == 3:
        mask = mask[:,0,0]
    labels = np.unique(mask)[1:]
    masksize = []
    if len(labels) != 0:
        if labelnum is None:
            labelnum = int(np.max(labels))
        for i in range(labelnum):
            masksize.append(len(mask[mask == i+1]))
    else:
        masksize.append(0)
    return np.array(masksize)
    
def get_signals(atlas, mask, method = 'mean', labelnum = None):
    """
    Extract roi signals of atlas from mask
    
    Parameters:
    -----------
    atlas: atlas
    mask: mask, a label image
    method: 'mean', 'std', 'ste', 'max', 'vertex', etc.
    labelnum: mask's label numbers, add this parameters for group analysis

    Return:
    -------
    signals: signals of specific roi
   
    Example:
    -------
    >>> signals = get_signals(atlas, mask, 'mean')
    """
    if atlas.ndim == 3:
        atlas = atlas[:,0,0]
    if mask.ndim == 3:
        mask = mask[:,0,0]
    
    
    labels = np.unique(mask)[1:]
    if labelnum is None:
        try:
            labelnum = int(np.max(labels))
        except ValueError as e:
            print('value in mask are all zeros')
            labelnum = 1
    if method == 'mean':
        calfunc = np.nanmean
    elif method == 'std':
        calfunc = np.nanstd
    elif method == 'max':
        calfunc = np.max
    elif method == 'vertex':
        calfunc = np.array
    elif method == 'ste':
        calfunc = tools.ste
    else:
        raise Exception('Miss paramter of method')
    signals = []
    for i in range(labelnum):
        if np.any(mask==i+1):
            signals.append(atlas[mask==i+1])
        else:
            signals.append(np.array([np.nan]))
    if atlas.ndim == mask.ndim+1:
        # time series
        if calfunc != np.array:
            return [calfunc(sg, axis=0) for sg in signals]
        else:
            return [calfunc(sg) for sg in signals]
    else:
        return [calfunc(sg) for sg in signals]

def get_vexnumber(atlas, mask, method = 'peak', labelnum = None):
    """
    Get vertex number of rois from surface space data
    
    Parameters:
    -----------
    atlas: atlas
    mask: mask, a label image
    method: 'peak' ,'center', or 'vertex', 
            'peak' means peak vertex number with maximum signals from specific roi
            'center', center of mass of roi
            'vertex' means extract all vertex of roi
    labelnum: mask's label numbers, add this parameters for group analysis
    
    Return:
    -------
    vexnumber: vertex number

    Example:
    --------
    >>> vexnumber = get_vexnumber(atlas, mask, 'peak')
    """
    if atlas.ndim == 3:
        atlas = atlas[:,0,0]
    if mask.ndim == 3:
        mask = mask[:,0,0]
    labels = np.unique(mask)[1:]
    if labelnum is None:
        try:
            labelnum = int(np.max(labels))
        except ValueError as e:
            labelnum = 0

    extractpeak = lambda x: np.unravel_index(x.argmax(), x.shape)[0]
    extractcenter = _extractcenter
    extractvertex = lambda x: x[x!=0]
    
    if method == 'peak':
        calfunc = extractpeak
    elif method == 'center':
        calfunc = extractcenter
    elif method == 'vertex':
        calfunc = extractvertex
    else:
        raise Exception('Miss parameter of method')

    vexnumber = []
    for i in range(labelnum):
        roisignal = atlas*(mask==(i+1))
        if np.any(roisignal):
            vexnumber.append(calfunc(roisignal))
        else:
            vexnumber.append(np.array([np.nan]))
    return vexnumber

def _extractcenter(roisignal):
    """
    Compute center of mass from ROI which included magnitudes in each vertex
    Surface has its special data construction, they're not consistent across whole ROI. It's better to use mapping method to solve this problem.

    Parameters:
    -----------
    roisignal: ROI that included magnitudes in each vertex

    Returns:
    --------
    center_point: center of mass from ROI
    """
    raw_idx = np.where(roisignal!=0)[0] 
    new_idx = tools.convert_listvalue_to_ordinal(raw_idx)
    new_center = sum(roisignal[e]*new_idx[i] for i,e in enumerate(raw_idx))/sum(roisignal[e] for _,e in enumerate(raw_idx))
    new_center = int(new_center)
    center_point = (raw_idx[i] for i,e in enumerate(new_idx) if e == new_center).next()
    return center_point


def surf_dist(vtx_src, vtx_dst, one_ring_neighbour):
    """
    Distance between vtx_src and vtx_dst
    Measured by edge number
    
    Parameters:
    -----------
    vtx_src: source vertex, int number
    vtx_dst: destinated vertex, int number
    one_ring_neighbour: one ring neighbour matrix, computed from get_n_ring_neighbour with n=1
    the format of this matrix:
    [{i1,j1,...}, {i2,j2,k2}]
    each element correspond to a vertex label

    Return:
    -------
    dist: distance between vtx_src and vtx_dst

    Example:
    --------
    >>> dist = surf_dist(vtx_src, vtx_dst, one_ring_neighbour)
    """
    if len(one_ring_neighbour[vtx_dst]) == 1:
        return np.inf
    
    noderep = copy.deepcopy(one_ring_neighbour[vtx_src])
    dist = 1
    while vtx_dst not in noderep:
        temprep = set()
        for ndlast in noderep:
            temprep.update(one_ring_neighbour[ndlast])
        noderep.update(temprep)
        dist += 1
    return dist
  
def hausdoff_distance(imgdata1, imgdata2, label1, label2, one_ring_neighbour):
    """
    Compute hausdoff distance between imgdata1 and imgdata2
    h(A,B) = max{max(i->A)min(j->B)d(i,j), max(j->B)min(i->A)d(i,j)}
    
    Parameters:
    -----------
    imgdata1: surface image data1
    imgdata2: surface image data2
    label1: label of image data1
    label2: label of image data2
    one_ring_neighbour: one ring neighbour matrix, similar description of surf_dist, got from get_n_ring_neighbour

    Return:
    -------
    hd: hausdorff distance
    
    Example:
    --------
    >>> hd = hausdoff_distance(imgdata1, imgdata2, 1, 1, one_ring_neighbour)
    """
    imgdata1 = tools.get_specificroi(imgdata1, label1)
    imgdata2 = tools.get_specificroi(imgdata2, label2)
    hd1 = _hausdoff_ab(imgdata1, imgdata2, one_ring_neighbour) 
    hd2 = _hausdoff_ab(imgdata2, imgdata1, one_ring_neighbour)
    return max(hd1, hd2)
 
def _hausdoff_ab(a, b, one_ring_neighbour):
    """
    Compute hausdoff distance of h(a,b)
    part unit of function hausdoff_distance
    
    Parameters:
    -----------
    a: array with 1 label
    b: array with 1 label
    one_ring_neighbour: one ring neighbour matrix

    Return:
    -------
    h: hausdoff(a,b)

    """
    a = np.array(a)
    b = np.array(b)
    h = 0
    for i in np.flatnonzero(a):
        hd = np.inf
        for j in np.flatnonzero(b):
            d = surf_dist(i,j, one_ring_neighbour)    
            if d<hd:
                hd = copy.deepcopy(d)
        if hd>h:
            h = hd
    return h

def median_minimal_distance(imgdata1, imgdata2, label1, label2, one_ring_neighbour):
    """
    Compute median minimal distance between two images
    mmd = median{min(i->A)d(i,j), min(j->B)d(i,j)}
    for detail please read paper:
    Groupwise whole-brain parcellation from resting-state fMRI data for network node identification
    
    Parameters:
    -----------
    imgdata1, imgdata2: surface data 1, 2
    label1, label2: label of surface data 1 and 2 used to comparison
    one_ring_neighbour: one ring neighbour matrix, similar description of surf_dist, got from get_n_ring_neighbour
    
    Return:
    -------
    mmd: median minimal distance

    Example:
    --------
    >>> mmd = median_minimal_distance(imgdata1, imgdata2, label1, label2, one_ring_neighbour)
    """
    imgdata1 = tools.get_specificroi(imgdata1, label1)
    imgdata2 = tools.get_specificroi(imgdata2, label2)
    dist1 = _mmd_ab(imgdata1, imgdata2, one_ring_neighbour)
    dist2 = _mmd_ab(imgdata2, imgdata1, one_ring_neighbour)
    return np.median(dist1 + dist2)

def _mmd_ab(a, b, one_ring_neighbour):
    """
    Compute median minimal distance between a,b
    
    part computational completion of median_minimal_distance

    Parameters:
    -----------
    a, b: array with 1 label
    one_ring_neighbour: one ring neighbour matrix

    Return:
    -------
    h: minimal distance
    """
    a = np.array(a)
    b = np.array(b)
    h = []
    for i in np.flatnonzero(a):
        hd = np.inf
        for j in np.flatnonzero(b):
            d = surf_dist(i, j, one_ring_neighbour)
            if d<hd:
                hd = d
        h.append(hd)
    return h

def get_n_ring_neighbor(vertx, faces, n=1, ordinal=False):
    """
    Get n ring neighbour from faces array

    Parameters:
    ---------
    vertex: vertex number
    faces : the array of shape [n_triangles, 3]
    n : integer
        specify which ring should be got
    ordinal : bool
        True: get the n_th ring neighbor
        False: get the n ring neighbor

    Return:
    ---------
    ringlist: array of ring nodes of each vertex
              The format of output will like below
              [{i1,j1,k1,...}, {i2,j2,k2,...}, ...]
			  each index of the list represents a vertex number
              each element is a set which includes neighbors of corresponding vertex

    Example:
    ---------
    >>> ringlist = get_n_ring_neighbour(24, faces, n)
    """
    if isinstance(vertx, int):
        vertx = [vertx]
    nth_ring = [set([vx]) for vx in vertx]
    nring = [set([vx]) for vx in vertx]

    while n != 0:
        n = n - 1
        for idx, neighbor_set in enumerate(nth_ring):
            neighbor_set_tmp = [_get_connvex_neigh(vx, faces) for vx in neighbor_set]
            neighbor_set_tmp = set().union(*neighbor_set_tmp)
            neighbor_set_tmp.difference_update(nring[idx])
            nth_ring[idx] = neighbor_set_tmp
            nring[idx].update(nth_ring[idx])
    if ordinal is True:
        return nth_ring
    else:
        return nring

def get_connvex(seedvex, faces, valuemask = None, labelmask = None, label = 1):
    """
    Get connected vertices that contain in mask
    We firstly need a start point to acquire connected vertices, then do region growing until all vertices in mask were included
    That means, output should satisfied two condition:
    1 overlap with mask
    2 connected with each other
    
    Parameters:
    -----------
    seedvex: seed point (start point)
    faces: faces array, vertex relationship
    valuemask: mask with values. if it exists, connection will be covered only region with decrease gradient
    mask: overlapping mask, a label mask
    masklabel: specific mask label used as restriction

    Return:
    -------
    connvex: connected vertice set

    Example:
    --------
    >>> connvex = get_connvex(24, faces, mask)
    """
    connvex = set()
    connvex.add(seedvex)
    neighbor_set = _get_connvex_neigh(seedvex, faces, labelmask, label)

    if valuemask is None:
        connvex_temp = neighbor_set
    else:
        assert valuemask.shape[0] == np.max(faces) + 1, "valuemask should has the same vertex number as faces connection relatonship"
        if valuemask.ndim != 2:
            valuemask = valuemask.reshape(valuemask.shape[0], 1)
        refpt = 1*seedvex
        connvex_temp = _mask_by_gradient(refpt, neighbor_set, valuemask)

    while not connvex_temp.issubset(connvex):
        connvex_dif = connvex_temp.difference(connvex)
        connvex.update(connvex_dif)
        connvex_temp = set()
        for sx in connvex_dif: 
            if valuemask is None:
                connvex_temp.update(_get_connvex_neigh(sx, faces, labelmask, label))
            else:
                refpt = 1*sx
                neighbor_set = _get_connvex_neigh(refpt, faces, labelmask, label)
                connvex_temp.update(_mask_by_gradient(refpt, neighbor_set, valuemask))
        print('Size of sulcus {0}'.format(len(connvex)))
    return connvex  

def _mask_by_gradient(refpt, neighbor_set, valuemask):
    """
    mask neighbor set by valuemask that choose vertices with value smaller than value of vertex refpt
    """
    return set([i for i in neighbor_set if valuemask[i]<valuemask[refpt]])


def _get_connvex_neigh(seedvex, faces, mask = None, masklabel = 1):
    """
    Function to get neighbouring vertices of a seed that satisfied overlap with mask
    """
    if mask is not None:
        assert mask.shape[0] == np.max(faces) + 1 ,"mask need to have same vertex number with faces connection relationship"
    assert isinstance(seedvex, (int, np.integer)), "only allow input an integer as seedvex"

    raw_faces, _ = np.where(faces == seedvex)
    rawconnvex = np.unique(faces[raw_faces])
    if mask is not None:
        connvex = set()
        list_connvex = [i for i in rawconnvex if mask[i] == masklabel]
        connvex.update(list_connvex)
    else:
        connvex = set(rawconnvex)
    connvex.discard(seedvex)
    return connvex

def inflated_roi_by_rg(orig_mask, ref_mask, faces):
    """
    Inflate orig_mask by extracting connected parcel of ref_mask
    A detailed explanation of this method:
    We have a more stricted orig_mask which pointed relatively correct position of a specific region (e.g. hMT/V5+)
    To use a larger ref_mask as constraints, by extracting connected parcels in ref_mask (the seed point comes from orig_mask), we can get a larger relatively accurate ROI of a specific region

    Parameters:
    -----------
    orig_mask: original mask with a/several small parcel(s)
    ref_mask: reference mask with larger parcel(s)
    faces: relationship of connection
    """
    orig_maskset = set(np.where(orig_mask!=0)[0])
    ref_maskset = set(np.where(orig_mask!=0)[0])

    connvex = set()
    dif_connvex = orig_maskset.difference(connvex)
    parcel_num = 0
    while len(dif_connvex) != 0:
        seedpt = random.choice(tuple(dif_connvex))
        parcel_connvex = get_connvex(seedpt, faces, ref_mask)
        connvex.update(parcel_connvex)
        dif_connvex = orig_maskset.difference(connvex)
        parcel_num += 1
        print('parcel number: {0}'.format(parcel_num))
    return connvex

def cutrg2parcels(orig_mask, faces, label = 1, label_rules = 'random'):
    """
    Use region growing method to cut discontinuous specific regions to parcels

    Parameters:
    -----------
    orig_mask: original mask has a/several small parcel(s) with labels
    faces: relationship of connection
    label[int]: specific label of orig_mask that used to cut into parcels
    label_ruls[string]: 'random', arrange new label randomly
                        'min2max', arrange new label from minimum vertex to maximum vertex
                        'max2min', arrange new label from maxmum vertex to minimum vertex

    Returns:
    --------
    parcel_mask: parcel mask that generated from specific label of orig_mask

    Example:
    --------
    >>> parcel_mask = cutrg2parcels(orig_mask, faces, 1)
    """
    if not isinstance(label, int):
        raise Exception('label need to be an int')
    lbl_orig_mask = tools.get_specificroi(orig_mask, label)
    parcel_mask = np.zeros_like(orig_mask)

    orig_maskset = set(np.where(lbl_orig_mask!=0)[0])
    connvex = set()
    dif_connvex = orig_maskset.difference(connvex)

    parcel_num = 0
    if label_rules == 'random':
        seedchoose = random.choice
    elif label_rules == 'min2max':
        seedchoose = np.min
    elif label_rules == 'max2min':
        seedchoose = np.max
    else:
        raise Exception('Set wrong parameter label_rules.')

    while len(dif_connvex) != 0:
        seedpt = seedchoose(tuple(dif_connvex))
        parcel_connvex = get_connvex(seedpt, faces, labelmask = lbl_orig_mask, label = label)
        connvex.update(parcel_connvex)
        dif_connvex = orig_maskset.difference(connvex)
        parcel_num += 1
        parcel_mask = tools.make_lblmask_by_loc(parcel_mask, tuple(parcel_connvex), parcel_num)
        print('parcel number: {0}'.format(parcel_num))
    return parcel_mask

def make_apm(act_merge, thr):
    """
    Compute activation probabilistic map

    Parameters:
    -----------
    act_merge: merged activation map
    thr_val: threshold of activation value

    Return:
    -------
    apm: activation probabilistic map

    Example:
    --------
    >>> apm = make_apm(act_merge, thr = 5.0)
    """
    import copy
    act_tmp = copy.deepcopy(act_merge)
    act_tmp[act_tmp<thr] = 0
    act_tmp[act_tmp!=0] = 1
    apm = np.mean(act_tmp,axis=-1)
    return apm

def make_pm(mask, meth = 'all', labelnum = None):
    """
    Compute probabilistic map
    
    Parameters:
    -----------
    mask: merged mask
          note that should be 2/4 dimension data
    meth: 'all' or 'part'
          'all', all subjects are taken into account
          'part', part subjects are taken into account, except subject with no roi label in specific roi
    labelnum: label number, by default is None
    
    Return:
    -------
    pm: probablistic map

    Example:
    --------
    >>> pm = make_pm(mask, 'all')
    
    """
    if (mask.ndim != 2)&(mask.ndim != 4):
        raise Exception('masks should be a 2/4 dimension file to get pm')
    if mask.ndim == 4:
        mask = mask.reshape(mask.shape[0], mask.shape[3])
    if labelnum is None:
        labels = range(1, int(np.max(mask))+1)
    else:
        labels = range(1, labelnum+1)
    pm = np.zeros((mask.shape[0],len(labels)))
    if meth == 'all':
        for i,e in enumerate(labels):
            pm[...,i] = np.mean(mask == e, axis = 1)
    elif meth == 'part':
        for i,e in enumerate(labels):
            mask_i = mask == e
            subj = np.any(mask_i, axis = 0)
            pm[...,i] = np.mean(mask_i[...,subj],axis = 1)
    else:
        raise Exception('Miss parameter meth')
    pm = pm.reshape((pm.shape[0], 1, 1, pm.shape[-1]))
    return pm

def make_mpm(pm, threshold, keep_prob = False, consider_baseline = False):
    """
    Make maximum probablistic map (mpm)
    
    Parameters:
    -----------
    pm: probabilistic map
        Note that pm.shape[3] should correspond to specific label of region
    threshold: threshold to filter vertex with low probability
    keep_prob: whether to keep maximum probability but not transform it into labels
               If True, return map with maximum probability
               If False, return map with label from maxmum probability
    consider_baseline: whether consider baseline or not when compute mpm
                       if True, check vertices that contain several probabilities, if p1+p2+...+pn < 0.5, then discard it
                       Details see Liang Wang, et al., Probabilistic Maps of Visual Topology in Human Cortex, 2015
    
    Return:
    -------
    mpm: maximum probabilistic map
    
    Example:
    >>> mpm = make_mpm(pm, 0.2)
    """
    if (pm.ndim != 4)&(pm.ndim != 2):
        raise Exception('Probablistic map should be 2/4 dimension to get maximum probablistic map')
    if pm.ndim == 4:
        pm = pm.reshape(pm.shape[0], pm.shape[3])
    pm[np.isnan(pm)] = 0
    pm_temp = np.zeros((pm.shape[0], pm.shape[1]+1))
    pm_temp[:, range(1,pm.shape[1]+1)] = pm
    pm_temp[pm_temp<threshold] = 0
    if consider_baseline is True:
        vex_discard = [(np.count_nonzero(pm_temp[i,:])>1)&((np.sum(pm_temp[i,:]))<0.5) for i in range(pm_temp.shape[0])]
        vex_disind = [i for i,e in enumerate(vex_discard) if e]
        pm_temp[vex_disind,:] = 0
    if not keep_prob: 
        mpm = np.argmax(pm_temp, axis=1)
    else:
        mpm = np.max(pm_temp, axis=1)
    mpm = mpm.reshape((-1, pm.ndim))
    return mpm
    
def nfold_location_overlap(imgdata1, imgdata2, labels, labelnum = None, index = 'dice', thr_meth = 'prob', prob_meth = 'part', n_fold=2, thr_range = [0,1,0.1], n_permutation=1, controlsize = False, actdata = None):
    """
    Decide the maximum threshold from raw image data.
    Here using the cross validation method to decide threhold using for getting the maximum probabilistic map
    
    Parameters:
    -----------
    imgdata1: A 2/4 dimensional data, used as train data
    imgdata2: A 2/4 dimensional data, used as test data
    labels: list, label number used to extract dice coefficient
    labelnum: by default is None, label number size. We recommend to provide label number here.
    index: 'dice' or 'percent'
    thr_meth: 'prob', threshold probabilistic map by probabilistic values (MPM)
              'number', threshold probabilistic map by numbers of vertex
    prob_meth: 'all' or 'part' subjects to use to compute probablistic map
    n_fold: split data into n_fold part, using first n_fold-1 part to get probabilistic map, then using rest part to evaluate overlap condition, by default is 2
    thr_range: pre-set threshold range to find the best maximum probabilistic threshold, the best threshold will search in this parameters, by default is [0,1,0.1], as the format of [start, stop, step]
    n_permuation: times of permutation, by default is 10
    controlsize: whether control label data size with template mpm label size or not, by default is False.
    actdata: if controlsize is True, please input actdata as a parameter. By default is None.

    Return:
    -------
    output_overlap: dice coefficient/percentage computed from function
                    output_dice consists of a 4 dimension array
                    permutation x threhold x subjects x regions
                    the first dimension permutation means the results of each permutation
                    the second dimension threhold means the results of pre-set threshold
                    the third dimension subjects means the results of each subject
                    the fourth dimension regions means the result of each region
    
    Example:
    --------
    >>> output_overlap = nfold_location_overlap(imgdata1, imgdata2, [2,4], labelnum = 4)
    """        
    if imgdata1.ndim == 4:
        imgdata1 = imgdata1.reshape((imgdata1.shape[0], imgdata1.shape[3]))
    if imgdata2.ndim == 4:
        imgdata2 = imgdata2.reshape((imgdata2.shape[0], imgdata2.shape[3]))
    if actdata is not None:
        if actdata.ndim == 4:
            actdata = actdata.reshape((actdata.shape[0], actdata.shape[3]))
    n_subj = imgdata1.shape[1]
    if labelnum is None:
        labelnum = int(np.max(np.unique(imgdata1)))
    assert (np.max(labels)<labelnum+1), "the maximum of labels should smaller than labelnum"
    output_overlap = []
    for n in range(n_permutation):
        print("permutation {} starts".format(n+1))
        test_subj = np.sort(np.random.choice(range(n_subj), n_subj-n_subj/n_fold, replace = False)).tolist()
        verify_subj = [val for val in range(n_subj) if val not in test_subj]
        test_data = imgdata1[:,test_subj]
        verify_data = imgdata2[:,verify_subj]
        if actdata is not None:
            verify_actdata = actdata[...,verify_subj]
        else:
            verify_actdata = None
        pm = make_pm(test_data, prob_meth, labelnum)
        pm_temp = cv_pm_overlap(pm, verify_data, labels, labels, thr_meth = thr_meth, thr_range = thr_range, index = index, cmpalllbl = False, controlsize = controlsize, actdata = verify_actdata)
        output_overlap.append(pm_temp)
    output_overlap = np.array(output_overlap)
    return output_overlap

def leave1out_location_overlap(imgdata1, imgdata2, labels, labelnum = None, index = 'dice', thr_meth = 'prob', prob_meth = 'part', thr_range = [0,1,0.1], controlsize = False, actdata = None):
    """
    A leave one out cross validation method for threshold to best overlapping in probabilistic map
    
    Parameters:
    -----------
    imgdata1: A 2/4 dimensional data, used as train data
    imgdata2: A 2/4 dimensional data, used as test data
    labels: list, label number used to extract dice coefficient
    labelnum: by default is None, label number size. We recommend to provide label number here.
    index: 'dice' or 'percent'
    thr_meth: 'prob', threshold probabilistic map by probabilistic values (MPM)
              'number', threshold probabilistic map by numbers of vertex
    prob_meth: 'all' or 'part' subjects to use to compute probablistic map
    thr_range: pre-set threshold range to find the best maximum probabilistic threshold
    controlsize: whether control label data size with template mpm label size or not, by default is False.
    actdata: if controlsize is True, please input actdata as a parameter. By default is None.

    Return:
    -------
    output_overlap: dice coefficient/percentage computed from function
                    outputdice consists of a 3 dimension array
                    subjects x threhold x regions
                    the first dimension means the values of each leave one out (leave one subject out)
                    the second dimension means the results of pre-set threshold
                    the third dimension means the results of each region

    Example:
    --------
    >>> output_overlap = leave1out_location_overlap(imgdata1, imgdata1, [2,4], labelnum = 4)
    """
    if imgdata1.ndim == 4:
        imgdata1 = imgdata1.reshape(imgdata1.shape[0], imgdata1.shape[-1])
    if imgdata2.ndim == 4:
        imgdata2 = imgdata2.reshape(imgdata2.shape[0], imgdata2.shape[-1])

    assert imgdata1.shape == imgdata2.shape, "The shape of imgdata1 and imgdata2 must be equal."
    
    if actdata is not None:
        if actdata.ndim == 4:
            actdata = actdata.reshape(actdata.shape[0], actdata.shape[-1])
    output_overlap = []
    for i in range(imgdata1.shape[-1]):
        data_temp = np.delete(imgdata1, i, axis=1)
        testdata = np.expand_dims(imgdata2[:,i],axis=1)
        if actdata is not None:
            test_actdata = np.expand_dims(actdata[:,i],axis=1)
        else:
            test_actdata = None
        pm = make_pm(data_temp, prob_meth, labelnum)
        pm_temp = cv_pm_overlap(pm, testdata, labels, labels, thr_meth = thr_meth, thr_range = thr_range, index = index, cmpalllbl = False, controlsize = controlsize, actdata = test_actdata)
        output_overlap.append(pm_temp)
    output_array = np.array(output_overlap)
    return output_array.reshape(output_array.shape[0], output_array.shape[2], output_array.shape[3])

def leave1out_magnitude(roidata, magdata, index = 'mean', thr_meth = 'prob', thr_range = [0,1,0.1], prob_meth = 'part'):
    """
    Function to use cross validation to extract magnitudes
    Compute probabilistic map to extract signal of the rest part of subject

    Parameters:
    ------------
    roidata: roidata used for probabilistic map
    magdata: magnitude data
    index: 'mean', 'std', 'ste', 'vertex', etc.
    thr_meth: 'prob', threshold probabilistic map by probabilistic values
              'number', threshold probabilistic map by numbers of vertex
    prob_meth: 'all' or 'part' used for probabilistic map generation
    thr_range: pre-set threshold range to compute thresholded labeled map

    Returns:
    --------
    mag_signals: magnitude signals
    
    Examples:
    ----------
    >>> mag_signals = leave1out_magnitude(roidata, magdata)
    """
    roidata = roidata.reshape(roidata.shape[0], roidata.shape[-1])
    magdata = magdata.reshape(magdata.shape[0], magdata.shape[-1])
    assert roidata.shape == magdata.shape, "roidata should have same shape as magdata"
    output_overlap = []
    n_subj = roidata.shape[-1]
    for i in range(roidata.shape[-1]):
        verify_subj = [i]
        test_subj = [val for val in range(n_subj) if val not in verify_subj]
        test_roidata = roidata[:, test_subj]
        verify_magdata = magdata[:, verify_subj]
        pm = make_pm(test_roidata, prob_meth)
        pm_temp = cv_pm_magnitude(pm, verify_magdata, index = index, thr_meth = thr_meth, thr_range = thr_range)
        output_overlap.append(pm_temp)
    output_array = np.array(output_overlap)
    output_array = output_array.reshape((-1,(thr_range[1]-thr_range[0])/thr_range[2]))
    return output_array

def nfold_magnitude(roidata, magdata, index = 'mean', thr_meth = 'prob', prob_meth = 'part', thr_range = [0,1,0.1], n_fold = 2, n_permutation = 1):
    """
    Using cross validation method to split data into nfold
    compute probabilistic map by first part of data, then extract signals of rest part of data using probabilistic map 

    Parameters:
    ------------
    roidata: roidata used for probabilistic map
    magdata: magnitude data
    index: 'mean', 'std', 'ste', 'vertex', etc.
    thr_meth: 'prob', threshold probabilistic map by probabilistic values (MPM)
              'number', threshold probabilistic map by numbers of vertex
    prob_meth: 'all' or 'part'. Subjects to use compute probabilistic map
    thr_range: pre-set threshold range to compute thresholded labeled map
    n_fold: split numbers for cross validation
    n_permutation: permutation times

    Returns:
    --------
    mag_signals: magnitude signals

    Examples:
    ---------
    >>> mag_signals = nfold_magnitude(roidata, magdata)
    """
    roidata = roidata.reshape(roidata.shape[0], roidata.shape[-1])
    magdata = magdata.reshape(magdata.shape[0], magdata.shape[-1])
    assert magdata.shape == roidata.shape, "roidata should have same shape as magdata"
    n_subj = roidata.shape[-1]
    output_overlap = []
    for n in range(n_permutation):
        print('permutation {} starts'.format(n+1))
        test_subj = np.sort(np.random.choice(n_subj, n_subj-n_subj/n_fold, replace = False)).tolist()
        verify_subj = [val for val in range(n_subj) if val not in test_subj]
        test_roidata = roidata[:, test_subj]
        verify_magdata = magdata[:, verify_subj]
        pm = make_pm(test_roidata, prob_meth)
        pm_temp = cv_pm_magnitude(pm, verify_magdata, index = index, thr_meth = thr_meth, thr_range = thr_range)
        output_overlap.append(pm_temp)
    output_array = np.array(output_overlap)
    return output_array

def pm_overlap(pm1, pm2, thr_range, option = 'number', index = 'dice'):
    """
    Analysis for probabilistic map overlap without using test data 
    The idea of this analysis is to control vertices number/threshold same among pm1 and pm2, binaried them then compute overlap
    
    Parameters:
    -----------
    pm1: probablistic map 1
    pm2: probabilistic map 2
    thr_range: threshold range, format as [min, max, step], which could be vertex numbers or probablistic threshold
    option: 'number', compute overlap between probablistic maps by multiple vertex numbers
            'threshold', compute overlap between probablistic maps by multiple thresholds
    index: 'dice', overlap indices as dice coefficient
           'percent', overlap indices as percent
    """
    assert (pm1.ndim == 1)|(pm1.ndim == 3), "pm1 should not contain multiple probablistic map"
    assert (pm2.ndim == 1)|(pm2.ndim == 3), "pm2 should not contain multiple probablistic map" 
    if pm1.ndim == 3:
        pm1 = pm1.reshape(pm1.shape[0], pm1.shape[-1])
    if pm1.ndim == 1:
        pm1 = np.expand_dims(pm1, axis=0)
    if pm2.ndim == 3:
        pm2 = pm2.reshape(pm2.shape[0], pm2.shape[-1])
    if pm2.ndim == 1:
        pm2 = np.expand_dims(pm2, axis=0)

    assert len(thr_range) == 3, "thr_range should be a 3 elements list, as [min, max, step]"
    if option == 'number':
        thre_func = tools.threshold_by_number
    elif option == 'threshold':
        thre_func = tools.threshold_by_values
    else:
        raise Exception('Missing option')

    output_overlap = []
    for i in np.arange(thr_range[0], thr_range[1], thr_range[2]):
        print('Computing overlap of vertices {}'.format(i))
        pm1_thr = thre_func(pm1, i)
        pm2_thr = thre_func(pm2, i)
        pm1_thr[pm1_thr!=0] = 1
        pm2_thr[pm2_thr!=0] = 1
        output_overlap.append(tools.calc_overlap(pm1_thr, pm2_thr, 1, 1))
    output_overlap = np.array(output_overlap)
    output_overlap[np.isnan(output_overlap)] = 0
    return output_overlap
         
def cv_pm_overlap(pm, test_data, labels_template, labels_testdata, index = 'dice', thr_meth = 'prob', thr_range = [0, 1, 0.1], cmpalllbl = True, controlsize = False, actdata = None):
    """
    Compute overlap(dice) between probabilistic map and test data
    
    Parameters:
    -----------
    pm: probabilistic map
    test_data: subject specific label data used as test data
    labels_template: list, label number of template (pm) used to extract overlap values 
    label_testdata: list, label number of test data used to extract overlap values
    index: 'dice' or 'percent'
    thr_meth: 'prob', threshold probabilistic map by probabilistic values (MPM)
              'number', threshold probabilistic map by numbers of vertex
    thr_range: pre-set threshold range to find the best maximum probabilistic threshold
    cmpalllbl: compute all overlap label one to one or not.
               e.g. labels_template = [2,4], labels_testdata = [2,4]
                    if True, get dice coefficient of (2,2), (2,4), (4,2), (4,4)
                    else, get dice coefficient of (2,2), (4,4)
    controlsize: whether control label data size with template mpm label size or not, by default is False.
    actdata: if controlsize is True, please input actdata as a parameter. By default is None.

    Return:
    -------
    output_overlap: dice coefficient/percentage
                    outputdice consists of a 3 dimension array
                    subjects x thr_range x regions
                    the first dimension means the values of each leave one out (leave one subject out)
                    the second dimension means the results of pre-set threshold
                    the third dimension means the results of each region
                 
    Example:
    --------
    >>> output_overlap = cv_pm_overlap(pm, test_data, [2,4], [2,4])
    """
    if cmpalllbl is False:
        assert len(labels_template) == len(labels_testdata), "Notice that labels_template should have same length of labels_testdata if cmpalllbl is False"
    if test_data.ndim == 4:
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[-1])
    if actdata is not None:
        if actdata.ndim == 4:
            actdata = actdata.reshape(actdata.shape[0], actdata.shape[-1])
    output_overlap = []
    for i in range(test_data.shape[-1]):
        thrmp_temp = []
        if actdata is not None:
            verify_actdata = actdata[:,i]
        else:
            verify_actdata = None
        for j,e in enumerate(np.arange(thr_range[0], thr_range[1], thr_range[2])):
            print("threshold {} is verifing".format(e))
            if thr_meth == 'prob':    
                thrmp = make_mpm(pm, e)
            elif thr_meth == 'number':
                if pm.shape[-1] > 1:
                    raise Exception('only support 1 label for this situation')
                thrmp = tools.threshold_by_number(pm, e)
                thrmp[thrmp!=0] = 1
            if cmpalllbl is True:
                thrmp_temp.append([tools.calc_overlap(thrmp.flatten(), test_data[:,i], lbltmp, lbltst, index, controlsize = controlsize, actdata = verify_actdata) for lbltmp in labels_template for lbltst in labels_testdata])
            else:
                thrmp_temp.append([tools.calc_overlap(thrmp.flatten(), test_data[:,i], labels_template[idx], lbld, index, controlsize = controlsize, actdata = verify_actdata) for idx, lbld in enumerate(labels_testdata)])
        output_overlap.append(thrmp_temp)
    return np.array(output_overlap)

def cv_pm_magnitude(pm, test_magdata, index = 'mean', thr_meth = 'prob', thr_range = [0,1,0.1]):
    """
    Function to extract signals from probabilistic map with varied threshold

    Parameters:
    ------------
    pm: probablistic map
    test_magdata: magnitude data used as test dataset
    index: type of signals, by default is 'mean'
    thr_meth: 'prob', threshold probabilistic map using probabilistic threshold
              'number', threshold probabilistic map using numbers of vertex
    thr_range: threshold range

    Results:
    ---------
    signals: signals of each threshold 

    Example:
    ---------
    >>> signals = cv_pm_magnitude(pm, test_magdata)
    """
    test_magdata = test_magdata.reshape(test_magdata.shape[0], test_magdata.shape[-1])
    pm = pm.reshape(pm.shape[0], pm.shape[-1])
    signal = []
    for i in range(test_magdata.shape[-1]):
        signal_thr = []
        for j,e in enumerate(np.arange(thr_range[0], thr_range[1], thr_range[2])):
            if thr_meth == 'prob':
                thrmp = make_mpm(pm, e)
            elif thr_meth == 'number':
                if pm.shape[-1] > 1:
                    raise Exception('only support 1 label for this situation')
                thrmp = tools.threshold_by_number(pm, e)
                thrmp[thrmp!=0] = 1
            else:
                raise Exception('Threshold probability only contains by probability values or vertex numbers')
            signal_thr.append(get_signals(test_magdata[:,i], thrmp[:,0], method = index))
        signal.append(signal_thr)
    return np.array(signal)
                
def overlap_bysubject(imgdata, labels, subj_range, labelnum = None, prob_meth = 'part', index = 'dice'):
    """
    A function used for computing overlap between template (probilistic map created by all subjects) and probabilistic map of randomly chosen subjects.
    
    Parameters:
    -----------
    imgdata: label image data
    labels: list, label number indicated regions
    subj_range: range of subjects, the format as [minsubj, maxsubj, step]
    labelnum: label numbers, by default is None
    prob_meth: method for probabilistic map, 'all' to compute all subjects that contains non-regions, 'part' to compute part subjects that ignore subjects with non-regions.

    Returns:
    --------
    overlap_subj: overlap indices of each amount of subjects

    Example:
    --------
    >>> overlap_subj = overlap_bysubject(imgdata, [4], [0,100,10], labelnum = 4) 
    """
    nsubj = imgdata.shape[-1]
    pm = make_pm(imgdata, meth = prob_meth, labelnum = labelnum)
    overlap_subj = []
    for i in np.arange(subj_range[0], subj_range[1], subj_range[2]):
        subj_num = np.random.choice(nsubj, i, replace=False)        
        sub_imgdata = imgdata[...,subj_num]
        if sub_imgdata.ndim == 3:
            sub_imgdata = np.expand_dims(sub_imgdata, axis=-1)
        pm_sub = make_pm(sub_imgdata, meth = prob_meth, labelnum = labelnum)
        overlap_lbl = []
        for lbl in labels:
            pm_lbl = pm[...,lbl-1]
            pm_sub_lbl = pm_sub[...,lbl-1]
            pm_lbl[pm_lbl!=0] = 1
            pm_sub_lbl[pm_sub_lbl!=0] = 1
            overlap_lbl.append(tools.calc_overlap(pm_lbl, pm_sub_lbl, 1, 1, index = index))
        overlap_subj.append(overlap_lbl)
    return np.array(overlap_subj)

class GetLblRegion(object):
    """
    A class to get template label regions
    
    Parameters:
    -----------
    template: template
    """
    def __init__(self, template):
        self._template = template

    def by_lblimg(self, lbldata):
        """
        Get specific template regions by rois given by user
        All regions overlapped with a specific label region will be covered

        Parameters:
        -----------
        lbldata: rois given by user

        Return:
        -------
        out_template: new template contains part of regions
                      if lbldata has multiple different rois, then new template will extract regions with each of roi given by user

        Example:
        --------
        >>> glr_cls = GetLblRegion(template)
        >>> out_template = glr_cls.by_lblimg(lbldata)
        """
        assert lbldata.shape == self._template.shape, "the shape of template should be equal to the shape of lbldata"
        labels = np.sort(np.unique(lbldata)[1:]).astype('int')
        out_template = np.zeros_like(lbldata)
        out_template = out_template[...,np.newaxis]
        out_template = np.tile(out_template, (1, len(labels)))
        for i,lbl in enumerate(labels):
            lbldata_tmp = tools.get_specificroi(lbldata, lbl)
            lbldata_tmp[lbldata_tmp!=0] = 1
            part_template = self._template*lbldata_tmp
            template_lbl = np.sort(np.unique(part_template)[1:])
            out_template[...,i] = tools.get_specificroi(self._template, template_lbl)
        return out_template

def get_border_vertex(data, faces, n = 2):
    """
    extract vertices that be in border of original data.

    Parameters:
    -----------
    data: original data (scalar data)
    faces: faces relationship  
    n: by default is 2. Neighboring ring number. 

    Returns:
    --------
    vx: vertices from border

    Examples:
    ---------
    >>> border_vertex = get_border_vertex(data, faces)
    """
    data_vertex = np.where(data!=0)[0]
    one_ring_neighbor = get_n_ring_neighbor(data_vertex, faces, n)
    border_check = [not np.all(data[list(i)]) for i in one_ring_neighbor]   
    border_vertex = data_vertex[np.array(border_check)]
    return border_vertex
    
def get_local_extrema(scalar_data, faces, surf_dist, n_extrema = None, mask = None, option = 'maxima'):
    """
    Get local extrema from scalar data 

    Parameters:
    -----------
    scalar_data[array]: scalar data that used for local extrema.
    faces[array]: nfaces * 3 array of defining mesh triangules.
    surf_dist[int]: the minimum distance between extrema of surface. Its value could be 1, 2, 3, etc..., as n ring neighbour.
    n_extrema: by default is None. Set the number of extrema point you'd like to get.
    mask[array]: by default is None. if not None, local extrema will be found in mask. Note that mask should have same shape like scalar_data.
    option[string]: 'maxima', find the maxima.
            'minima', find the minima.

    Returns:
    ---------
    extre_points[list]: vertex number of local extrema points.

    Example:
    --------
    >>> extre_points = get_local_extrema(scalar_data, faces, surf_dist=3)
    """
    if option == 'maxima':
        temp_scalar = scalar_data - np.min(scalar_data)
        argextre = np.argmax
    elif option == 'minima':
        temp_scalar = scalar_data - np.max(scalar_data)
        argextre = np.argmin
    else:   
        raise Exception('please input maxima or minima in option.') 
    if mask is not None:
        assert scalar_data.shape == mask.shape, "Mask should have same shape like scalar_data."
        temp_scalar = temp_scalar * (mask!=0)
    extre_points = []
    # median_value = np.abs(np.median(temp_scalar[temp_scalar!=0]))
    while np.any(temp_scalar):
        # if np.max(np.abs(temp_scalar))<median_value:
        #     break
        if (n_extrema is not None) & (len(extre_points)>=n_extrema):
            break
        temp_extre_point = argextre(temp_scalar)
        extre_points.append(temp_extre_point)
        ringlist = get_n_ring_neighbor(temp_extre_point, faces, n=surf_dist)
        temp_scalar[list(ringlist[0])] = 0
    return extre_points    





