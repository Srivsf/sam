"""Classes and methods for splitting data into chunks."""

import numpy as np
import pandas as pd

class GroupBySplitter(object):
    """Provide an iterator on a Pandas DataFrame into n_splits, keeping elements of groups together.

    Example:

        grpby_cols = ['b', 'c']
        n_rows = 20
        n_splits = 4

        df = pd.DataFrame(np.arange(n_rows*3).reshape(n_rows,3), columns=['a', 'b', 'c'])
        df['a'] = np.arange(n_rows, 0, -1)
        df['b'] = np.arange(n_rows) % 2
        df['c'] = np.arange(n_rows, 0, -1) % 5
        display(df)

        splitter = GroupBySplitter(['b', 'c'])
        for X in splitter.split(df, n_splits):
            display(X)

    Example:

        from dhi.dsmatch.util.datasplitters import GroupBySplitter
        from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer

        df = pd.DataFrame(np.arange(n_rows*3).reshape(n_rows,3), columns=['a', 'b', 'c'])
        df['a'] = np.arange(n_rows, 0, -1)
        df['b'] = np.arange(n_rows) % 2
        df['c'] = np.arange(n_rows, 0, -1) % 5

        splitter = GroupBySplitter(['b', 'c'])
        txfmr = ApplyTransformer(try_func, training_similarities, split_func=splitter.split)

    """
    def __init__(self, grpby_cols):
        """
        Args:
            grpby_cols (list or str): List of columns for a Pandas GroupBy.
        """
        self.grpby_cols = grpby_cols

    def split(self, df, n_splits: int=1):
        """Provide an iterator on a Pandas DataFrame into n_splits, keeping elements of groups together

        Args:
            df (pd.DataFrame): Pandas DataFrame with columns in the `grpby_cols` list to keep together.
            n_splits (int): Number of splits. Default is 1, which simply returns df.

        Yields:
            pd.DataFrame: DataFrames while maintaining `grpby_cols` boundaries.
        """
        try:
            df.drop(['grp_id'], axis=1, inplace=True)
        except KeyError:
            pass

        grpby = df.groupby(self.grpby_cols, sort=False)
        df_ = pd.Series(np.arange(grpby.ngroups) % n_splits, index=grpby.indices.keys()).to_frame('grp_id')
        df_.index.set_names(self.grpby_cols, inplace=True)
        df_.reset_index(inplace=True)
        df = pd.merge(df, df_)
        
        for _, g in df.groupby('grp_id'):
            yield g.iloc[:, :-1]

        try:
            df.drop(['grp_id'], axis=1, inplace=True)
        except KeyError:
            pass

class MatrixSplitter(object):
    """Provide an iterator to split a (possibly sparse) matrix into `n_splits` rows.
    """
    def split(self, a, n_splits: int=1):
        """Provide an iterator to split a sparse matrix into `n_splits` rows.

        Args:
            a (sparse csr matrix): Scipy CSR sparse matrix that will be splitted by rows.
            n_splits (int): Number of splits. Default is 1, which simply returns a.

        Yields:
            sparse csr matrix: sub-matrices.
        """
        d, r = divmod(a.shape[0], n_splits)
        for i in range(n_splits):
            si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
            yield a[si:si+(d+1 if i < r else d)]

class MatrixSplitterTuple(object):
    """Provide an iterator to split a (possibly sparse) matrix into `n_splits` rows.
    """
    def split(self, X, n_splits: int=1):
        """Provide an iterator to split a sparse matrix into `n_splits` rows.

        Args:
            X (tuple of 2 sparse csr matrices): A tuple of two Scipy CSR sparse matrices of the same size 
                that will be splitted by rows.
            n_splits (int): Number of splits. Default is 1, which simply returns a.

        Yields:
            tuple of two sparse csr matrices: sub-matrices.
        """
        if not isinstance(X, tuple):
            raise ValueError('X must be a zipped tuple of two sparse matrices.')
        a, b = X
        if a.shape[0] != b.shape[0]:
            raise ValueError('Zipped sparse matrices need to have the same number of rows.')
        d, r = divmod(a.shape[0], n_splits)
        for i in range(n_splits):
            si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
            yield a[si:si+(d+1 if i < r else d)], b[si:si+(d+1 if i < r else d)]
