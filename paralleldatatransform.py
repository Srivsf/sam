"""Method for splitting data in transformers for parallelization.

NOTE: Use models/paralleldatatransformer, if possible, over this method.
"""
import multiprocessing as mp

import scipy.sparse as sp
import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.preprocessing import FunctionTransformer

# from ..util.cache import make_cacheable
from  dhi.dsmatch.util.parallel import ProgressParallel, get_n_splits
from  dhi.dsmatch.util.parallel import CHUNKSIZE, MIN_CHUNKSIZE, N_JOBS


USE_TQDM = True


def _transform_parallel_data(self, X, n_splits, use_tqdm=USE_TQDM, desc='', n_jobs=N_JOBS, **pkwargs):        
    transform_splits = ProgressParallel(use_tqdm=use_tqdm, total=n_splits, desc=desc, n_jobs=n_jobs, **pkwargs)(
        delayed(self._transform_orig)(X_split)
        for X_split in np.array_split(X, n_splits))
    try:
        return sp.vstack(transform_splits)
    except ValueError:
        try:
            return pd.concat(transform_splits)
        except:
            pass
#     except (KeyError, ValueError, TypeError):
    return np.vstack(transform_splits)

def _transform(self, X, **kwargs):

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
            desc = self.label
        except AttributeError:
            desc = ''
        return self._transform_parallel_data(X, n_splits, use_tqdm=use_tqdm, desc=desc, n_jobs=n_jobs, **kwargs)
    return self._transform_orig(X, **kwargs)

def make_parallel_data_transform(self):
    """Given a Transformer instance (an instantiation of TransformerMixin with a `transform()` method),
    Provide a monkey patch to so that when `transform()` is called, the data is split and applied
    on the chunks, then reassembled.

    This functionality is largely for existing functions and objects outside our library. See the caveats below.
    In general, approaches using the ApplyTransformer or subclassing of ParallelDataTransformer should be applied.

    Note: If trying to cache in a Sklearn Pipeline by passing a `memory` argument to the Pipeline
    creation, this method does not work. During `Pipeline.fit()`, Sklearn clones instantiations, which
    does not keep the modified `transform()` method that does the parallelization.

    Note: To cache with Joblib's Memory mechanism, that assignment must follow this application.
    
    Example:
        The following example uses an off-the-shelf Transformer, a CountVectorizer, and also uses
        the Memory mechanism for caching.

            from sklearn.feature_extraction.text import CountVectorizer
            from ..util.cache import make_cacheable
            c = CountVectorizer()
            make_parallel_data_transform(c)
            make_cacheable(c.transform, './cachedir')
            c.fit(["this and that", "the other"])
            c.transform(['this','other', 'other', 'thing','yeah']).toarray()

        Outputs

            array([[0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])

    Raises:
        err: AttributeError if this is object does not have a `transform` method.
    """
    try:
        getattr(self, 'transform')
    except AttributeError as err:
        raise err
    setattr(self, '_transform_parallel_data', _transform_parallel_data.__get__(self))
    setattr(self, '_transform_orig', getattr(self, 'transform'))
    setattr(self, 'transform', _transform.__get__(self))
