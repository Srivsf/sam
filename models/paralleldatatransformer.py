from functools import wraps

import numpy as np
import scipy.sparse as sp
import pandas as pd
from joblib import delayed

from dhi.dsmatch.util.parallel import ProgressParallel, get_n_splits
from dhi.dsmatch.util.parallel import CHUNKSIZE, MIN_CHUNKSIZE, N_JOBS
from dhi.dsmatch.sklearnmodeling.models.custombase import CustomTransformer

USE_TQDM = True

class ParallelDataTransformer(CustomTransformer):
    """Base class for transformers that can operate on splitted/chunked/sharded data during fitting and larger
    transformations.
    """
    
    def __init__(self, use_tqdm=USE_TQDM, desc='', leave=True, n_jobs=N_JOBS, fkwargs={}, split_func=np.array_split,
            chunksize=CHUNKSIZE, min_chunksize=MIN_CHUNKSIZE):
        """[summary]

        Args:
            use_tqdm (bool, optional): Show TQDM status bars in parallelization mode. Defaults to USE_TQDM.
            desc (str, optional): TQDM "desc" parameter to describe status bars when used. Defaults to ''.
            leave (bool, optional): TQDM "leave" parameter. Defaults to True.
            n_jobs (int, optional): Number of processors to use during parallel processing. Follows
                joblib/sklearn's modes where -1 is all processors, -2 is all but one, and a positive integer
                is the number of allocated processors. Defaults to N_JOBS.
            fkwargs (dict, optional): For a fun that might be used during parallel split, these are additional
                arguments passed to that function. Defaults to {}.
            split_func (function, optional): Function used to split the data into chunks. For Pandas DataFrames
                that need to maintain group boundaries instead of arbitrarily splitting rows, consider incorporating
                the GroupBySplitter. Defaults to np.array_split.
            chunksize (int or float, optional): If a float between 0 and 1, it is the fraction of data per processor. 
                Otherwise, as an int, this is the number of rows per chunk. Defaults to CHUNKSIZE.
            min_chunksize (int, optional): Minimum number of rows to include in each chunk. Defaults to MIN_CHUNKSIZE.
        """
        self.use_tqdm = use_tqdm
        self.desc = desc
        self.leave = leave
        self.n_jobs = n_jobs
        self.fkwargs = fkwargs
        self.split_func = split_func
        self.chunksize = chunksize
        self.min_chunksize = min_chunksize
        
    def transform_parallel_data(self, func, X, n_splits, use_tqdm=USE_TQDM, desc='', n_jobs=N_JOBS, **pkwargs):
        """Perform the `transform()` method in parallel by splitting the data.

        Args:
            func (function): the `self.transform()` method.
            X (Array, DataFrame, Series, dict, list, sparse_matrix): Data to transform.
            n_splits (integer): Number of data splits
            use_tqdm (bool, optional): Whether to show the TQDM status bar or not. Defaults to USE_TQDM.
            desc (str, optional): TQDM description label. Defaults to ''.
            n_jobs (int, optional): Number of processors to use during parallel processing. Follows
                joblib/sklearn's modes where -1 is all processors, -2 is all but one, and a positive integer
                is the number of allocated processors. Defaults to N_JOBS.

        Returns:
            X_out : sparse matrix if possible, else a 2-d array
            Transformed input.

        """
        transform_splits = ProgressParallel(use_tqdm=use_tqdm, total=n_splits, desc=desc, n_jobs=n_jobs, **pkwargs)(
            delayed(func)(self, X_split, **self.fkwargs)
            for X_split in self.split_func(X, n_splits))
        # We try and return the data in a few prioritized forms:
        #   1. Sparse matrix (when the number of zeros are greater than the square root of all possible values)
        #   2. Pandas DataFrame
        #   3. Numpy ndarray
        #   4. Numpy array
        #   5. List of the transformed data.
        try:
            try:
                res = sp.vstack(transform_splits)
                if np.sqrt(np.product(res.shape)) < res.nnz:
                    return pd.concat(transform_splits)
                return res
            except (KeyError, ValueError, TypeError):
                return pd.concat(transform_splits)
        except (KeyError, ValueError, TypeError):
            try:
                return np.vstack(transform_splits)
            except ValueError:
                pass
            
            try:
                return np.concatenate(transform_splits)
            except ValueError:
                pass
        
        return transform_splits

    @staticmethod
    def parallelize_data(func):
        """Decorator to determine if we should apply the transform in parallel or not, then execute accordingly.

        Args:
            func (function): Our `self.transform()` method.

        Returns:
            Numpy-compatible object: Transformation of X.
        """
        @wraps(func)
        def wrapper(self, X, **kwargs):
            try:
                min_chunksize = self.min_chunksize
            except AttributeError:
                min_chunksize = MIN_CHUNKSIZE

            try:
                n_jobs = self.n_jobs
            except AttributeError:
                n_jobs = N_JOBS

            try:
                chunksize = self.chunksize
            except AttributeError:
                chunksize = CHUNKSIZE

            n_splits = get_n_splits(X, n_jobs=n_jobs, chunksize=chunksize, min_chunksize=min_chunksize)

            if n_splits > 1:
                try:
                    use_tqdm = self.use_tqdm
                except AttributeError:
                    use_tqdm = USE_TQDM

                try:
                    desc = self.desc
                except AttributeError:
                    desc = ''
                return self.transform_parallel_data(func, X, n_splits, 
                        use_tqdm=use_tqdm, desc=desc, n_jobs=n_jobs, **kwargs)
            return func(self, X, **kwargs)        
        return wrapper
