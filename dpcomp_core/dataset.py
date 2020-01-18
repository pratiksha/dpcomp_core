from __future__ import division
from builtins import zip
from builtins import range
import hashlib
import os
import numpy
import pandas as pd
from dpcomp_core import util
from dpcomp_core.mixins import Marshallable
from functools import reduce

# short cuts for dataset names, filenames
filenameDict = {

    # 1D data with domain 4096
    "HEPTH"         : '1D/HEPTH.n4096.npy',
    "ADULT"    : '1D/ADULTFRANK.n4096.npy',
    "INCOME"        : '1D/INCOME.n4096.npy',
    "MEDCOST"       : '1D/MEDCOST.n4096.npy',
    "NETTRACE"      : '1D/NETTRACE.n4096.npy',
    "PATENT"        : '1D/PATENT.n4096.npy',
    "SEARCHLOGS"    : '1D/SEARCHLOGS.n4096.npy',
    "BIDS-FJ"       : '1D/bids_j.npy',
    "BIDS-FM"       : '1D/bids_m.npy',
    "BIDS-ALL"      : '1D/bids.npy',
    "MDSALARY"      : '1D/GrossCompenMDSalary1.npy',
    "MDSALARY-FA"   : '1D/GrossCompenMDSalary2_filtered.npy',
    "LC-REQ-F1"     : '1D/RejectStats_0_4096_no_outliers_filter_7_1.npy',
    "LC-REQ-F2"     : '1D/RejectStats_0_4096_no_outliers_filter_7_2.npy',
    "LC-REQ-ALL"    : '1D/RejectStats_0_4096_no_outliers.npy',
    "LC-DTIR-F1"    : '1D/RejectStats_4_4096_no_outliers_filter_7_1.npy',
    "LC-DTIR-F2"    : '1D/RejectStats_4_4096_no_outliers_filter_7_2.npy',
    "LC-DTIR-ALL"   : '1D/RejectStats_4_4096_no_outliers.npy',

    # 2D data with domain 256 by 256
    "SF-CABS-E"         : '2D/cabspottingE_256_256.npy',
    "SF-CABS-S"         : '2D/cabspottingS_256_256.npy',
    "GOWALLA"           : '2D/checkin_256_256.npy',
    "BEIJING-CABS-E"    : '2D/BeijingTaxiE_256_256.npy',
    "BEIJING-CABS-S"    : '2D/BeijingTaxiS_256_256.npy',
    "ADULT-2D"          : '2D/Adult_2D_256_256.npy',
    "MDSALARY-OVERT"    : '2D/MDSalary2D_256_256.npy',
    "LOAN-FUNDED-INCOME": '2D/Loan_2D_256_256.npy',
    "STROKE"            : '2D/Stroke_2D_256_256.npy',
    "TWITTER"           : '2D/twitter_256_256.npy',

    # Hillview data
    "ONTIME" : 'ontime_private/2018_1.csv',
    }

class Dataset(Marshallable):

    def __init__(self, hist, reduce_to_domain_shape=None, dist=None):
        """
            Any instances with equal key() values should have equal hash() values
            domain_shape will be result of regular grid partition
        """
        if isinstance(reduce_to_domain_shape, int): # allow for integers in 1D, instead of shape tuples
            reduce_to_domain_shape = (reduce_to_domain_shape, )

        if dist is not None:
            self._dist_str = numpy.array_str(numpy.array(dist))
        else:
            self._dist_str = ''

        self._hist = hist
        self._reduce_to_domain_shape = reduce_to_domain_shape
        self._dist = dist
        self._payload = None

        self._compiled = False

    def reduce_data(self,p_grid,data):
        """reduce data to the domain indicated by p_grid"""
        assert data.shape == p_grid.shape, 'Shape of x and shape of partition vector must match.'
        #get out_shape
        if p_grid.ndim == 1:
            out_shape =(len(numpy.unique(p_grid)), )
        else:
            out = [len(numpy.unique(numpy.compress([True], p_grid, axis=i))) for i in range(p_grid.ndim)]
            out.reverse()   # rows/cols need to be reversed here
            out_shape = tuple(out)
        #reduce
        unique, indices, inverse, counts = numpy.unique(p_grid, return_index=True, return_inverse=True, return_counts=True)
        res = numpy.zeros_like(unique, dtype=float)
        for index, c in numpy.ndenumerate(data.ravel()):   # needs to be flattened for parallel indexing with output of unique
            res[ inverse[index] ] += c
           
        return numpy.array(res).reshape(out_shape)


    def compile(self):
        if self._compiled:
            return self

        self._payload = self._hist
        if self._reduce_to_domain_shape:    # reduce cells to get desired shape
            q = [divmod(dom,grid) for (dom,grid) in zip(self._payload.shape, self._reduce_to_domain_shape)]
            for red, rem in q:
                assert rem == 0, 'Domain must be reducible to target domain by uniform grid: %i, %i' % (red,rem)
            grid_shape = [r[0] for r in q]
            p_grid = partition_grid(self._payload.shape, grid_shape)
            self._payload = self.reduce_data(p_grid,self._payload)  # update payload

            if self._dist is not None:
                self._dist = self.reduce_data(p_grid,self._dist)  # update dist

        self._compiled = True
        self._payload = self._payload.astype("int") # partition engines need payload to be of type int

        return self


    @property
    def payload(self):
        return self.compile()._payload

    @property
    def dist(self):
        return self.compile()._dist

    @property
    def scale(self):
        return self.payload.sum()

    @property
    def domain_shape(self):
        return self.payload.shape

    @property
    def fractionZeros(self):
        zero_count = (self.payload == 0).sum()
        return util.old_div(float(zero_count), self.payload.size)

    @property
    def maxCount(self):
        return self.payload.max()

    @property
    def key(self):
        """ Using leading 8 characters of hash as key for now """
        return self.hash[:8]

    @property
    def hash(self):
        m = hashlib.sha1()
        m.update(util.prepare_for_hash(self.__class__.__name__))
        m.update(util.prepare_for_hash(numpy.array(self._hist)))
        m.update(util.prepare_for_hash(numpy.array(self._reduce_to_domain_shape)))
        m.update(util.prepare_for_hash(self._dist_str))
        return m.hexdigest()

    # represent object state as dict (for storage in datastore)
    def asDict(self):
        d = util.class_to_dict(self, ignore_list=['_dist', 'dist', '_payload', 'payload', '_compiled',
                                                  '_dist_str', '_hist', '_reduce_to_domain_shape'])
        ## add decomposed shape attrs ??
        return d

    def analysis_payload(self):
        return {'payload': self.payload}


class DatasetFromFile(Dataset):

    def __init__(self, nickname, reduce_to_dom_shape=None):
        self.fname = nickname
        assert nickname in filenameDict, 'Filename parameter not recognized: %s' % nickname
        hist = load(filenameDict[self.fname])
        super(DatasetFromFile,self).__init__(hist, reduce_to_dom_shape, None)


class DatasetSampled(Dataset):

    def __init__(self, dist, sample_to_scale, reduce_to_dom_shape=None, seed=None):
        self.seed = seed
        prng = numpy.random.RandomState(self.seed)
        hist = subSample(dist, sample_to_scale, prng)
        super(DatasetSampled,self).__init__(hist, reduce_to_dom_shape, dist)


class DatasetSampledFromFile(DatasetSampled):

    def __init__(self, nickname, sample_to_scale, reduce_to_dom_shape=None, seed=None):
        self.init_params = util.init_params_from_locals(locals())

        self.fname = nickname
        assert nickname in filenameDict, 'Filename parameter not recognized: %s' % nickname
        hist = load(filenameDict[self.fname])
        dist = util.old_div(hist, float(hist.sum()))
        super(DatasetSampledFromFile,self).__init__(dist, sample_to_scale, reduce_to_dom_shape, seed)

'''
Compute a histogram from an underlying CSV file of un-aggregated data points. 
The parameter `bounds` is a list of (min, max, binsize) tuples that specifies the bin values for each
desired column.
'''
class DoubleDatasetFromRaw(Dataset):
    def __init__(self, nickname, colnames, bounds, reduce_to_dom_shape=None):
        assert(len(colnames) in [1, 2])
        assert(len(bounds) == len(colnames))
        
        self.fname = nickname
        assert nickname in filenameDict, 'Filename parameter not recognized: %s' % nickname
        rows = load_csv(filenameDict[self.fname], colnames)
        hist = double_rows_to_hist(rows, bounds)
        print(hist)
        super(DoubleDatasetFromRaw, self).__init__(hist, reduce_to_dom_shape, None)

class StringDatasetFromRaw(Dataset):
    def __init__(self, nickname, colnames, left_bounds, reduce_to_dom_shape=None):
        assert(len(colnames) in [1, 2])
        assert(len(left_bounds) == len(colnames))
        
        self.fname = nickname
        assert nickname in filenameDict, 'Filename parameter not recognized: %s' % nickname
        rows = load_csv(filenameDict[self.fname], colnames)
        hist = string_rows_to_hist(rows, left_bounds)
        super(StringDatasetFromRaw, self).__init__(hist, reduce_to_dom_shape, None)

class MixedDatasetFromRaw(Dataset):
    def __init__(self, nickname, colnames, left_bounds, bounds_types, reduce_to_dom_shape=None):
        assert(len(left_bounds) == len(colnames))
        
        self.fname = nickname
        assert nickname in filenameDict, 'Filename parameter not recognized: %s' % nickname
        rows = load_csv(filenameDict[self.fname], colnames)
        hist = rows_to_nd_hist(rows, left_bounds, bounds_types)
        super(MixedDatasetFromRaw, self).__init__(hist, reduce_to_dom_shape, None)

'''
Dummy dataset with zero counts and the specified shape.
'''
class AllZerosDataset(Dataset):

    def __init__(self, shape, reduce_to_dom_shape=None):
        hist = numpy.zeros(shape)
        self.fname = ""
        super(AllZerosDataset, self).__init__(hist, reduce_to_dom_shape, None)
        
tryPaths = [os.path.join(os.environ['DPCOMP_CORE'], 'dpcomp_core/datafiles'),
            os.environ['HILLVIEW_DATA_DIR']]

def get_path(filename):
    """
    Load from file and return original counts (should be integral)
    """
    filepaths = [os.path.join(p, filename) for p in tryPaths]
    valid_paths = [p for p in filepaths if os.path.exists(p)]
    assert valid_paths, 'data path not found.'

    return valid_paths[0]

def load(filename):
    fullpath = get_path(filename)
    _, file_extension = os.path.splitext(fullpath)
    if file_extension == '.txt':
        x = []
        with open(fullpath, 'r') as fp:
            for ln in fp.readlines():
                x.append(int(ln))
        return numpy.array(x, dtype='int32')
    elif file_extension == '.npy':
        return numpy.load(fullpath)
    else:
        raise Exception('Unrecognized file extension')


'''
Create a 1- or 2-d non-private histogram from the rows.
'''
def double_rows_to_hist(rows, bounds):
    print(rows)
    if len(bounds) == 1:
        # 1D histogram
        (min_, max_, binsize) = bounds[0]
        nbins = int((max_ - min_) / float(binsize))
        return numpy.histogram(rows, bins=nbins, range=(min_, max_))[0] # second element is bin labels
    else:
        raise NotImplementedError('Unsupported number of columns for histogram')

def double_row_to_idx(row, bounds):
    (min_, max_, binsize) = bounds
    return int((row-min_)//binsize)
    
'''
Find the bin index for this value
'''
def binary_search(val, left_bounds):
    (lb, max_val) = left_bounds
    if val > max_val or val < lb[0]:
        return -1
    
    hi = len(lb) - 1
    lo = 0

    while (lo <= hi):
        mid = lo + ((hi - lo) // 2)
        if val == lb[mid]:
            return mid
        if val < lb[mid]:
            hi = mid - 1
        elif val > lb[mid]:
            lo = mid + 1
    return mid

def string_row_to_idx(row, left_bounds):
    idx = binary_search(row, left_bounds)
    return idx

def string_rows_to_hist(rows, left_bounds):
    if len(left_bounds) == 1:
        lb = left_bounds[0]
        
        # 1D histogram
        ret = numpy.zeros(len(lb))
        for r in rows:
            idx = binary_search(r, lb)
            if idx == -1:
                continue
            ret[idx] += 1
        return ret
    else:
        raise NotImplementedError('Unsupported number of columns for histogram')

def rows_to_nd_hist(rows, left_bounds_arr, bounds_types):
    if len(left_bounds_arr) > 2:
        raise NotImplementedError('Unsupported number of columns')

    if len(left_bounds_arr) != len(bounds_types):
        raise ValueError('Mismatch in bounds lengths')

    dimlens = []
    for (lb, bt) in zip(left_bounds_arr, bounds_types):
        if bt == 'DOUBLE':
            (min_, max_, binsize) = lb
            nbins = int((max_ - min_) / float(binsize)) + 1
            dimlens.append(nbins)
        elif bt == 'STRING':
            (bounds, max_) = lb
            dimlens.append(len(bounds))
        else:
            raise TypeError('Invalid column type')

    hist = numpy.zeros(dimlens)
    for r in rows:
        idxs = []
        for (x, lb, bt) in zip(r, left_bounds_arr, bounds_types):
            if bt == 'DOUBLE':
                idx = double_row_to_idx(x, lb)
            elif bt == 'STRING':
                idx = string_row_to_idx(x, lb)
            else:
                raise TypeError('Invalid column type')
            idxs.append(idx)
        print(idxs)
        if -1 in idxs: # outside of specified private bounds
            continue
        hist[tuple(idxs)] += 1

    return hist
        
'''
Load raw CSV data where each row corresponds to an individual, selecting columns specified by colnames.
'''
def load_csv(csv_fname, colnames):
    full_fname = get_path(csv_fname)
    df = pd.read_csv(full_fname, delimiter=',')
    assert(x in df.columns for x in colnames)
    arr = df[list(colnames)].values
    return arr

def subSample(dist, sampleSize, prng):
    ''' Generate a subsample of given sampleSize from an input distribution '''
    samples = prng.choice(a=dist.size, replace=True, size=int(sampleSize), p=dist.flatten())
    hist = numpy.histogram(samples, bins=dist.size, range=(0,dist.size))[0].astype('int32')		# return only counts (not bins)
    return hist.reshape( dist.shape )

# partition vector generation for data reduction
def cantor_pairing(a, b):
    """
    A function returning a unique positive integer for every pair (a,b) of positive integers
    """
    return (a+b)*(a+b+1)/2 + b

def general_pairing(tup):
    """ Generalize cantor pairing to k dimensions """
    if len(tup) == 0:
        return tup[0]  # no need for pairing in 1D case
    else:
        return reduce(cantor_pairing, tup)

def canonicalTransform(vector):
    """ transform a partition vector according to the canonical order.
     if bins are noncontiguous, use position of first occurrence.
     e.g. [3,4,1,1] => [1,2,3,3]; [3,4,1,1,0,1]=>[0,1,2,2,3,2]
    """
    unique, indices, inverse = numpy.unique(vector, return_index=True, return_inverse=True)
    uniqueInverse, indexInverse = numpy.unique(inverse,return_index =True)

    indexInverse.sort()
    newIndex = inverse[indexInverse]
    tups = list(zip(uniqueInverse, newIndex)) #replace uniqueInverse with unique if we want to use the exact numbers in partition vector
    tups.sort(key=lambda x: x[1])
    u = numpy.array( [u for (u,i) in tups] )
    vector = u[inverse].reshape(vector.shape)
    return vector

def partition_grid(domain_shape, grid_shape):
    """
    :param domain_shape: a shape tuple describing the domain, e.g (6,6) (in 2D)
    :param grid_shape: a shape tuple describing cells to be grouped, e.g. (2,3) to form groups of 2 rows and 3 cols
        note: in 1D both of the above params can simply be integers
    :return: a partition array in which grouped cells are assigned some unique 'group id' values
             no guarantee on order of the group ids, only that they are unique
    """

    # allow for integers instead of shape tuples in 1D
    if isinstance(domain_shape, int):
        domain_shape = (domain_shape, )
    if isinstance(grid_shape, int):
        grid_shape = (grid_shape,)

    assert sum(divmod(d,b)[1] for (d,b) in zip(domain_shape, grid_shape)) == 0, "Domain size along each dimension should be a multiple of size of block"

    def g(*idx):
        """
        This function will receive an index tuple from numpy.fromfunction
        It's behavior depends on grid_shape: take (i,j) and divide by grid_shape (in each dimension)
        That becomes an identifier of the block; then assign a unique integer to it using pairing.
        """
        x = numpy.array(idx)
        y = numpy.array(grid_shape)
        return general_pairing( util.old_div(x,y) )  # broadcasting integer division

    h = numpy.vectorize(g)

    # numpy.fromfunction builds an array of domain_shape by calling a function with each index tuple (e.g. (i,j))
    partition_array = numpy.fromfunction(h, domain_shape, dtype=int)
    # transform to canonical order 
    partition_array = canonicalTransform(partition_array)
    return partition_array

