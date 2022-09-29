"""
Various utility functions for operating in parallel.

"""
from itertools import repeat
import multiprocessing as mp
import logging
import uuid

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel

CHUNKSIZE = .1
MIN_CHUNKSIZE = 32
N_JOBS = -2

class ProgressParallel(Parallel):
    """Subclass to joblib.Parallel, which displays a TQDM status bar.
    
    Borrowed from https://stackoverflow.com/a/61027781/394430
    """
    def __init__(self, use_tqdm=True, total=None, desc='', leave=True, *args, **kwargs):
        """
        Args:
            use_tqdm (bool, optional): Whether or not to display tqdm status bars. Defaults to True.
            total (int, optional): Size of the data, the number of splits or jobs. If unknown and unspecified,
                then status bars will show status, but progress toward the end will be open-ended. Defaults to None.
            desc (str, optional): Label applied to the progress bar. Defaults to ''.
            leave (bool, optional): For nested tqdm. Should be False if a nested progress bar should hide when complete.
        """
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        self._leave = leave
        super().__init__(*args, **kwargs)
        
    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, desc=self._desc, leave=self._leave) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        """This gets called with every process as it completes. This is where status bars get updated.
        """
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def get_n_splits(X, n_jobs: int=N_JOBS, chunksize=CHUNKSIZE, min_chunksize: int=MIN_CHUNKSIZE) -> int:
    """Given the size of data, the number of parallel jobs and chunksize, determine the appropriate number
    of splits to the data.

    Args:
        X (Array, DataFrame, Series, dict, list, sparse_matrix): The data to split.
        n_jobs (int, optional): Number of processors to use during parallel processing. Follows
            joblib/sklearn's modes where -1 is all processors, -2 is all but one, and a positive integer
            is the number of allocated processors. Defaults to N_JOBS.
        chunksize (float or int, optional):  

            * If a float, the number of subdivisions per processor. For example, if 0.1, then each processor 
            used will get 1/10th of the data and the data will be split 10*n_processors times. 
            * If an int, then this is the desired number of rows per chunk to use.
            
            Defaults to CHUNKSIZE.

        min_chunksize (int, optional): Minimum number of rows necessary for each chunk. If the data is too small,
            then it will not be split and this function will return 1. Defaults to MIN_CHUNKSIZE.

    Returns:
        int: The number of splits to be applied to the data.
    """
    if isinstance(X, tuple):  # If a tuple, then assume zipped objects of the same length.
        X = X[0]
    try:
        len_x = X.shape[0]
    except AttributeError:
        len_x = len(X)

    if n_jobs < 0:
        n_jobs = mp.cpu_count() + 1 + n_jobs
    if chunksize == 1:
        n_splits = n_jobs
    elif chunksize < 1:
        n_splits = np.ceil(n_jobs/chunksize)
    else:
        n_splits = np.ceil(len_x/chunksize)
    
    if len_x/n_splits <= min_chunksize:
        n_splits = np.ceil(len_x/min_chunksize)

    if n_splits < 1:
        n_splits = 1
    
    return int(n_splits)
    
class ParallelFileLogger(object):
    """Object for file logging when running joblib.Parallel.

    Example:

        import os
        try:
            os.remove('parallel-logging.log')
        except FileNotFoundError:
            pass

        import time

        from joblib import Parallel, delayed
        from dhi.dsmatch.util.parallel import ParallelFileLogger

        def func(i, pfl):
            logger = pfl.get_logger()
            logger.warning(f"Function calling {i}")
            time.sleep(0.5)

        pfl = ParallelFileLogger(filename='parallel-logging.log')

        with Parallel(n_jobs=2) as parallel:
            parallel(delayed(func)(i, pfl) for i in range(10))

    """
    def __init__(self, userlevel=logging.DEBUG, formatter=logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d]'), **kwargs):
        """Object for file logging when running joblib.Parallel.

        Args:
            userlevel: One of logging.DEBUG, INFO, WARNING, ERROR, CRITICAL. Defaults to logging.DEBUG.
            formatter: [logging.Formatter](https://docs.python.org/3/library/logging.html#formatter-objects) object.
            kwargs: kwargs as given to [logging.FileHandler]
            (https://docs.python.org/3/library/logging.handlers.html#logging.FileHandler).
            `filename` always needs to be set, and `mode` should always equal 'a', which
            is the default.
        """
        if 'mode' in kwargs and kwargs['mode'] != 'a':
            errstr = 'Logging file needs to be in append mode when multiple processes are writing to the same file.'
            raise ValueError(errstr)
        self.userlevel = userlevel
        self.formatter = formatter
        self.fh_kwargs = kwargs
    
    def get_logger(self, name: str=None):
        """Get a logging.logger object.

        Args:
            name (str, optional): A unique name for at least each process. If None, a uuid is automatically generated
            to ensure uniqueness. Defaults to None.

        Returns:
            logging.logger object: [description]
        """
        if name is None:
            name = str(uuid.uuid4())
        logger = logging.getLogger(name)
        handler = logging.FileHandler(**self.fh_kwargs)
        handler.setLevel(self.userlevel)
        handler.setFormatter(self.formatter)
        logger.addHandler(handler)
        logger.setLevel(self.userlevel)
        return logger   
