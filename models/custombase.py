"""Abstract Base Class for Models. Supports saving and loading so all models are consistent that way.
"""
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import joblib

from dhi.dsmatch import s3_ds_bucket
from dhi.dsmatch import local_bucket

class CustomEstimator(BaseEstimator):

    @property
    def version(self) -> str:
        return self._version
        
    @version.setter
    def version(self, v):
        raise ValueError("Cannot set the version in this instance variable. It is set as a static member variable.")

    @abstractmethod
    def _version(cls):
        return NotImplementedError('Subclasses need to have a static "_version" member in "Major.Minor.Patch" syntax.')

    def fit(self, X, y=None, **kwargs):
        return self
    
    @staticmethod
    def load_model(path: str, filename: str=None, bucket: str=local_bucket):
        """Load a model from storage. This function has two modes of operating:

          1. If `filename` is None, then `path` must specify the full path to the file, including the filename.
          2. If `filename` is not None, then `path` is the directory less the bucket name where the file is.
              It will first look in `local_bucket` and if it cannot find it there, it will look for it on S3.

        Args:
            path (str): path to the folder containing the model file, excluding the root/bucket, or complete path
                including the filename.
            filename (str): name of the stored model file, e.g., "mymodel.joblib". Default is None, which means
                path should be a complete path to the file.
            bucket (str, optional): Root/Bucket. Defaults to local_bucket.

        Returns:
            object: A sklearn model file with `predict()` and/or `transform()` methods.
        """
        global s3_ds_bucket

        if filename is None:
            return joblib.load(path)

        try:
            return joblib.load(os.path.join(bucket, path, filename))
        except FileNotFoundError:
            pass
        return joblib.load(os.path.join('s3://', s3_ds_bucket, path, filename))

    @staticmethod
    def save_model(model, path: str, filename: str=None, bucket: str=local_bucket, lean=False):
        """Save a model for later reuse.

        Args:
            model (model): The object to store.
            path (str): path to the folder containing the model file, excluding the root/bucket.
            filename (str): name of the stored model file, e.g., "mymodel.joblib".
            bucket (str, optional): Root/Bucket. Defaults to local_bucket.

        Returns:
            object: A model file with `predict()` and/or `transform()` methods.
        """
        global s3_ds_bucket
        if lean:
            model.make_lean()  # Will raise an Attribute error if the inherited object does not implement this.
        if filename is None:
            return joblib.dump(model, path)
        try:
            return joblib.dump(model, os.path.join(bucket, path, filename))
        except FileNotFoundError:
            pass
        return joblib.dump(model, os.path.join('s3://', s3_ds_bucket, path, filename))

    @staticmethod
    def get_fitted_variables(obj):
        """Get the names of this object's fitted variables. This assumes that fitted variables follow 
        [sklearn's guidelines](https://scikit-learn.org/stable/developers/develop.html#estimated-attributes)
        that fitted variables a named with a trailing underscore.

        Returns:
            list: A list of the names this object's fitted variables.
        """
        return [v for v in vars(obj) if v.endswith('_') and not v.startswith('__')]

    @staticmethod
    def clear_fitted_variables(obj):
        """Clears this object's fitted variables as well as all sub-transformers' fitted variables."""
        pass
    
class CustomClassifier(ABC, CustomEstimator, ClassifierMixin):
    @abstractmethod
    def predict(self, X, y=None, **kwargs):
        return
    
class CustomTransformer(ABC, CustomEstimator, TransformerMixin):
    @abstractmethod
    def transform(self, X):
        return

class EchoTransformer(BaseEstimator, TransformerMixin):
    """Dummy class to be used at the end of a caching pipeline so that the step before is
    cached in fit_transform.
    """
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return X

class SelectTransformer(BaseEstimator, TransformerMixin):
    """Transformer that assumes input is a DataFrame or dict and then selects filters for only those columns or keys.
    """
    def __init__(self, feature_names_out: List[str]):
        """
        Args:
            feature_names_out (list[str]): List of column names or keys to select for.
        """
        self.feature_names_out = feature_names_out
        
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        """Filter X for our features.

        Args:
            X (pd.DataFrame or dict): Should have `self.feature_names_out` as columns or keys. 

        Returns:
            X filtered.
        """
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names_out]
        elif isinstance(X, dict):
            filtered_d = dict((k, v) for k, v in X.items() if k in self.feature_names_out)
            if len(filtered_d) == 1:
                return list(filtered_d.values())[0]
            return filtered_d
        return X
