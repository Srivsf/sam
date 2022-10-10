from warnings import warn

import pandas as pd

from dhi.dsmatch.util.parallel import N_JOBS
from dhi.dsmatch.sklearnmodeling.models.paralleldatatransformer import ParallelDataTransformer, USE_TQDM

class CoupledTransformer(ParallelDataTransformer):
    """Conjoins two or more string columns into one column, adding a period and space between the columns.
    This is especially useful for combining cells in a DataFrame as a single document.
    """
    def __init__(self, feature_names_out, use_tqdm=USE_TQDM, desc='', n_jobs=N_JOBS):
        super().__init__(use_tqdm=use_tqdm, desc=desc, n_jobs=n_jobs)
        self.feature_names_out = feature_names_out
        warn_str = "This class has been deprecated. \nConsider"
        warn_str += "`ApplyTransformer(applyrows, lambda x: '. '.join(x), "
        warn_str += "fkwargs=dict(output_cols=['-'.join(feature_names_out)])`"
        warn(warn_str, DeprecationWarning, 2)

    def get_name(self):
        return '-'.join(self.feature_names_out)
    
    @ParallelDataTransformer.parallelize_data
    def transform(self, X, y=None):
        """Conjoin our columns.

        Args:
            X (DataFrame or dict): Must have the feature_names_out as columns or keys, which are text columns.

        Returns:
            Array-like: Conjoined text columns with a ". " between the columns.
        """
        df = None
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            df = X[self.feature_names_out]
        elif isinstance(X, dict):
            df = pd.DataFrame.from_dict(X)[self.feature_names_out]
        else:
            df = pd.DataFrame(X, columns=self.feature_names_out)
        return df.apply(lambda x: '. '.join(x), axis=1)
