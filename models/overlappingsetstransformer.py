from warnings import warn

from collections import Counter
import multiprocessing as mp

import numpy as np
import pandas as pd
from joblib import delayed
from pandas.core.frame import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import QuantileTransformer

from dhi.dsmatch.util.parallel import ProgressParallel
from dhi.dsmatch.sklearnmodeling.models.quantilehelpers import QuantileConfidenceTransformer, QuantilePredictMixin, QTResult
from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer
from dhi.dsmatch.sklearnmodeling.functiontransformermapper import try_func
from dhi.dsmatch.util.misc import split_list

class OverlappingSetsTransformer(QuantileConfidenceTransformer):
    """Transformer that contrasts the overlapping items in two sets.

    For two columns of sets of data (each cell in each column is a list of items), we report how many of the 
    items on the right side overlap with the left. Items may also be weighted.

    To glean confidence, we collect the distribution of overlapping items in the standard condition, then we
    shuffle the right column and get another distribution of overlapping items. We then perform logistic regression
    to get a probability of each condition at a given value. We define the **confidence** of a score belonging to 
    one group or another as $2 \cdot \max(P[0], P[1]) - 1$ where $P[0]$ is the probability of the NULL hypothesis 
    (shuffled) and $P[1]$ is the probability of the actual distribution.

    > **Note:** Overlaps are "left-based" such that the left column is held constant, and we want to know
    the number of overlapping items from the right column to the left. When shuffling, we shuffle the
    right column and the left column stays fixed. A future extension may make the set distribution more symmetric.
    """
    _version = '1.0.0'

    def __init__(self, n_permutations: int=100, 
                 n_limit_threshold: float=.975, 
                 data_threshold: int=100,
                 overlap_col: str='frac_overlap', 
                 duplicate_scaling_func=np.sqrt, 
                 prediction_thresholds: list=None,
                 default_transform_mode: str='full',
                 capture_histograms: bool=False):
        """
        Args:
            n_permutations (int, optional): Number of permutations/shuffles of the right column. Defaults to 100.
            n_limit_threshold (float, optional): The quantile limit for number of elements. 
                When considering overlapping sets, we assume the number of overlapping items likely tails off at 
                the right side. That is, there is a higher probability of having a few items matching instead of 
                nearly all of them matching. As such, the tails will not have enough data to obtain a realistic
                distribution for capturing confidence. The quantile specified here, clips `n_elem` on the left 
                side to this quantile value. Defaults to .975.
            data_threshold (int, optional): Similar to the above, confidences will only be derived when the training
                data for a given number of matching elements is above this number. Defaults to 100.
            overlap_col (str, optional): Column for the output to specify the fraction of overlap. 
                Defaults to 'frac_overlap'.
            duplicate_scaling_func (func, optional): For items that are duplicated in either the left or right side,
                we only give them one weight.
                
                Examples:
                
                    * The function `lambda x: 1` acts as a binary feature. The value is constant, regardless of the
                      number of duplicates.
                    * Function `lambda x: x` would linearly scale the representations. If an item occurred 5 times,
                      it would have a weight of 5.
                    * Function `np.sqrt` applies the square root to the number of entries. If an item occurred 9 times,
                      it would have a weight of 3.
                Defaults to np.sqrt.

            prediction_thresholds (list, optional): List of floating point values between 0-1 that represents the
                quantile cutoffs for prediction values. Defaults to None, which accepts the defaults in 
                QuantilePredictMixin.
            default_transform_mode (str, optional): Defaults to 'full'. (Other options are used internally)
            capture_histograms (bool, optional): Data distributions are made for making the logistic regression
                classifiers used in the confidence score. Setting this to True retains histograms for model interrogation
                and interpretation later. Defaults to False.
        """
        warn_msg = 'The "OverlappingSetsTransformer" class is deprecated and has been supplanted by '
        warn_msg += '"LeftOverlappingSetsTransformer", which has a different API.'
        warn(warn_msg, DeprecationWarning)
        self.n_permutations = n_permutations
        self.n_limit_threshold = n_limit_threshold
        self.data_threshold = data_threshold
        self.overlap_col = overlap_col
        self.duplicate_scaling_func = duplicate_scaling_func
        self.default_transform_mode = default_transform_mode
        self.capture_histograms = capture_histograms
        QuantilePredictMixin.__init__(self, prediction_thresholds)
        
    def _frac_overlap(self, row, transform_mode: str='full'):
        """Get the fraction of overlap between a row from a DataFrame where row[0] is the left
        set to match with row[1], the right set.

        Args:
            row: A row from a Pandas DataFrame with at least 2 columns.
            transform_mode (str, optional): Defaults to 'full'. Other options are use internally.

        Returns:
            tuple: Tuple of (Number of elements, overall overlapping match score, 
                DataFrame of individual item match scores)
        """
        df_ = pd.DataFrame.from_dict(Counter(row[0]), orient='index', columns=[row.index[0]])
        if len(df_) == 0:
            if transform_mode == 'fitting':
                return 0, 0.
            if transform_mode == 'shuffling':
                return 0.
            return 0, 0., None
        df_right = pd.DataFrame.from_dict(Counter(row[1]), orient='index', columns=[row.index[1]])
        if len(df_right) == 0:
            if transform_mode == 'fitting':
                return 0, 0.
            if transform_mode == 'shuffling':
                return 0.
            return df_.shape[0], 0., None
        
        df_ = pd.merge(df_, df_right, left_index=True, right_index=True, how='left')

        df_ = df_.transform(self.duplicate_scaling_func)
        df_ /= df_.mean(axis=0)
        df_.fillna(0, inplace=True)  # Filling NA here means NaNs don't impact the mean above.
        df_['penalty'] = (df_.iloc[:, 0] - df_.iloc[:, 1]).apply(lambda x: np.max([0, x]))
        
        if transform_mode == 'fitting':
            return df_.shape[0], 1 - df_.penalty.mean()
        
        if transform_mode == 'shuffling':
            return 1 - df_.penalty.mean()
        
        # transform_mode == 'full'
        return df_.shape[0], 1 - df_.penalty.mean(), df_.sort_values(by=[row.index[0], row.index[1]], ascending=False)

    def _shuffle_frac_overlap(self, df: DataFrame, subrange: list):
        """Get the shuffled frac_overlap scores in a DataFrame by adding a number of columns to the dataframe
        with shuffled items from the right column.

        Args:
            df (pd.DataFrame): DataFrame with the left and right columns that we get our `frac_overlap` scores from.
                We will shuffle the right column.
            subrange (list): Range of integers for the columns we will add. This allows us to add items in parallel.

        Returns:
            [dict]: Dict with the column names that can be expanded after running in parallel to a DataFrame.
        """
        dfs = {}

        for i in subrange:
            np.random.seed(i)
            rand_idxs = np.random.permutation(df.index)
            cols = df.columns
            df_ = pd.concat([df[cols[0]], df.loc[rand_idxs, cols[1]].reset_index(drop=True)], 
                            axis=1, ignore_index=True)
            dfs[f'{self.overlap_col}_rand_{i}'] = df_.apply(self._frac_overlap, 
                                                      transform_mode='shuffling',
                                                      axis=1)
        return dfs
    
    def _get_shuffled(self, df: DataFrame) -> DataFrame:
        """Get a DataFrame that is the same length as `df`, but with a number of columns that contain the
        `frac_overlap()` scores from shuffled values.

        Args:
            df (DataFrame): DataFrame where the left and right columns that contain the overlapping sets
                are in the [0, 1] column position, respectively.

        Returns:
            [DataFrame]: DataFrame of shuffled values.
        """
        n_jobs = mp.cpu_count()
        n_splits = np.max([self.n_permutations, n_jobs * 2])

        ranges = list(split_list(range(self.n_permutations), n_splits))
        try:
            desc = f'{self.name} - shuffling'
        except AttributeError:
            desc = 'shuffling'
        shuffled_n = ProgressParallel(use_tqdm=True, desc=desc, total=len(ranges), n_jobs=n_jobs)(
                    delayed(self._shuffle_frac_overlap)(df, subrange)
                    for subrange in ranges
        )

        df_shuffled = pd.concat(map(lambda x: DataFrame(x), shuffled_n), axis=1)
        return df_shuffled
        
    def _get_frac_overlap_subframe(self, X, transform_mode='full'):
        """Helper function to `_get_frac_overlap()` to run in parallel mode."""
        return X.apply(self._frac_overlap, transform_mode=transform_mode, axis=1)

    def _get_frac_overlap(self, X, transform_mode='full') -> DataFrame:
        try:
            desc = f'{self.name} - get_frac_overlap'
        except AttributeError:
            desc = 'get_frac_overlap'
        txfmr = ApplyTransformer(try_func, self._get_frac_overlap_subframe, desc=desc,
                                 fkwargs=dict(transform_mode=transform_mode))
        
        X = txfmr.transform(X)
        if transform_mode == 'full':
            return pd.DataFrame(X.tolist(), columns=['n_elem', f'{self.overlap_col}', 'df'])
        return pd.DataFrame(X.tolist(), columns=['n_elem', f'{self.overlap_col}'])
    
    def _fit_confidences(self, df: DataFrame):
        """Fit the logistic regression classifiers

        Args:
            df (DataFrame): DataFrame that contains the `n_elem`, `frac_overlap`, and
                `frac_overlap_rand...` columns.
        """
        self.n_limit_ = int(df[f'n_elem'].quantile(self.n_limit_threshold))
        shuffled_cols = [c for c in df.columns if c.startswith(f'{self.overlap_col}_rand_')]

        # a = list(set(range(1, self.n_limit_+1)).difference(set(df.n_elem.unique())))
        # if len(a) > 0:
        #     self.n_limit_ = int(np.min(a))

        if self.capture_histograms:
            self.hists_ = {'actual': {}, 'shuffled': {}}
        self.lr_classifiers_ = {}

        for n in range(1, self.n_limit_+1):
            if n == self.n_limit_:
                df_ = df[df['n_elem'] >= n]
            else:
                df_ = df[df['n_elem'] == n]
            if len(df_) < self.data_threshold:
                continue

            X1 = df_[f'{self.overlap_col}'].copy().values
            X1 = X1[np.isfinite(X1)]
            X0 = df_[shuffled_cols].copy()
            X0 = X0.stack().reset_index(drop=True).values
            X0 = X0[np.isfinite(X0)]

            # Create confidence via logistic regression
            X = np.concatenate([X0, X1]).reshape(-1, 1)
            # Logistic regression does not work for values in the range of [0,1]. We need to rescale to something
            # resembling our discrete range, so we multiply by n
            X *= n  
            y = np.zeros(X.shape[0])
            y[-X1.shape[0]:] = 1
            self.lr_classifiers_[n] = LogisticRegression(penalty='l1', solver='liblinear', random_state=0, 
                    class_weight='balanced').fit(X, y)
            
            if self.capture_histograms:
                granularity = 5
                bins = np.linspace(-1/n/2/granularity, 1+1/n/2/granularity, n*granularity+2)
                self.hists_['actual'][n] = np.histogram(X1, bins=bins, 
                                                        weights=np.ones(len(X1))/len(X1), density=False)
                self.hists_['shuffled'][n] = np.histogram(X0, bins=bins, 
                                                          weights=np.ones(len(X0))/len(X0), density=False)

    def _fit(self, X, y=None):
        """Our main fit function.

        Args:
            X (DataFrame): Input data that contains our left and right columns that each have sublists to map.
        """
        df = X.copy()
#         _get_frac_overlap = memory.cache(self._get_frac_overlap, ignore=['self'])
        df_frac = self._get_frac_overlap(df, transform_mode='fitting')
#         _get_shuffled = memory.cache(self._get_shuffled, ignore=['self'])
        df_shuffled = self._get_shuffled(df)
        df = pd.concat([df, df_frac, df_shuffled], axis=1)
        self._fit_confidences(df)
        self.qt_ = QuantileTransformer(n_quantiles=np.min([1000, df.shape[0]]), random_state=df.shape[0])
        self.qt_.fit(df[f'{self.overlap_col}'].values.reshape(-1, 1))
        
    def _get_qtresult(self, row):
        # try:
        n = int(row[0])
        if n <= 0:
            return QTResult(0, 0.)
        if n > self.n_limit_:
            n = self.n_limit_ - 1
        while n not in self.lr_classifiers_:
            n -= 1
            if n == 0:
                return QTResult(0, 0.)
        v = row[1]
        log_prob = self.lr_classifiers_[n].predict_proba(v.reshape(1,-1))
        conf = (2 * np.max(log_prob) - 1).item()
        return QTResult(self.qt_.transform(v.reshape(1,-1)).item(), conf)
        # except:
        #     pass
        # return QTResult(0, 0.) 

    def _get_qtresults_subframe(self, X):
        return X.apply(self._get_qtresult, axis=1)

    def _get_qtresults(self, X):
        try:
            desc = f'{self.name} - get_qtresults'
        except AttributeError:
            desc = 'get_qtresults'
        txfmr = ApplyTransformer(try_func, self._get_qtresults_subframe, desc=desc)
        X = txfmr.transform(X)
        return pd.DataFrame(X.tolist(), columns=QTResult._fields)
    
    def fit(self, X, y=None):
        self._fit(X, y)
        return self
    
#     def fit_transform(self, X, y=None, **kwargs):
#         self._fit(X, y)
#         return self.transform(X, y, **kwargs)

    def transform(self, X, y=None, **kwargs):
        """Main transform function.

        Args:
            X (DataFrame): DataFrame with the first two columns containing the left and right data sets to evaluate.

        Returns:
            [DataFrame]: DataFrame with the original values, concatenated with the overall fraction overlap,
                individual item scores, overall quantile scores, and confidences.
        """
        check_is_fitted(self)
        df = X.iloc[:, :2].reset_index(drop=True)
        transform_mode = kwargs.pop('transform_mode', self.default_transform_mode)
        df_frac = self._get_frac_overlap(df, transform_mode=transform_mode)
        df_qtr = self._get_qtresults(df_frac.iloc[:, :2])
        df = pd.concat([df, df_frac, df_qtr], axis=1)
        self.transform_cols = df.columns
        return df
