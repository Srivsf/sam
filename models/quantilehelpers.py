from collections import namedtuple

import numpy as np
import pandas as pd

from dhi.dsmatch.sklearnmodeling.models.custombase import CustomTransformer

QTResult = namedtuple('QTResult', ['qtile', 'confidence'])

class QuantilePredictMixin:
    def __init__(self, prediction_thresholds=None):
        """Set the prediction_thresholds parameter. Inherited classes should call 
        `QuantilePredictMixin.__init__(self, prediction_thresholds)` during their `__init__()`.

        Args:
            prediction_thresholds (list): List of floats to specify match boundaries of quantiles during predict.
                Default is [.1, .3, .7, .9] -- everything less than 0.1 is a 1, .1 to .3 is a 2, .3 to .7 is a 3,
                .7 to .9 is a 4 and over .9 is a 5.
        """
        if prediction_thresholds is None:
            prediction_thresholds = [.1, .3, .7, .9]
        self.prediction_thresholds = prediction_thresholds

    def map_prediction(self, quantile, confidence):
        """Translate the percentile rank into a match score, 1, 2, 3, 4, or 5."""
        if np.isnan(quantile) or confidence <= 0.01:
            return 1  # np.nan  # For now we must always return something. Worst case, a 1.

        for i, v in enumerate(self.prediction_thresholds):
            if quantile < v:
                return i + 1
        return len(self.prediction_thresholds) + 1

    def _predict_after_transform(self, X):
        ret = list(map(lambda row: self.map_prediction(row[1], row[2]), X[['qtile', 'confidence']].itertuples()))
        if len(ret) == 1:
            return ret[0]
        return ret

    def predict(self, X):
        """Predict 1, 2, 3, 4, or 5 between job description skills and resume skills similarity.

        Args:
            X (pd.DataFrame): Assumes the first column is the job description skills and the second
                column is the resume skills. Each row is from a job application, and each cell in the
                row is a list of skills.

        Returns:
            list: List (or scalar) of 1, 2, 3, 4, or 5 for predictions to a particular class, or np.nan
                if the confidence is 0.
        """
        X = self.transform(X)  # check_is_fitted is called here...
        return self._predict_after_transform(X)

    def set_prediction_thresholds_from_labels(self, s: pd.Series):
        prediction_thresholds = s.value_counts(normalize=True).sort_index().cumsum().iloc[:-1].values
        self.prediction_thresholds = prediction_thresholds

class QuantileConfidenceTransformer(CustomTransformer, QuantilePredictMixin):
    # @abstractmethod
    # def predict_quantile(self, X, with_confidence=True, **kwargs):
    #     return
    
    @property
    def importance(self) -> float:
        return self._importance
    
    @importance.setter
    def importance(self, v) -> float:
        self._importance = np.clip(v, 0., 1.)

class QuantileComposite:
    @staticmethod
    def update_importance(idx, val, importances):
        factor = 0.
        for i, v in enumerate(importances):
            if i != idx:
                factor += v
        factor = (1 - val) / factor
        for i in range(len(importances)):
            if i != idx:
                importances[i] *= factor
        
        importances[idx] = val
        
        return QuantileComposite.qualify_importances(importances)

    @staticmethod
    def qualify_importances(importances):
        if not np.isclose(np.sum(importances), 1):
            return importances / np.sum(importances)
        return importances

    @staticmethod
    def calculate_score(df, importances):
        epsilon = 1e-5
        cols = [c for c in df.columns if c.endswith('confidence')]
        df[cols] = df[cols].mask(df[cols] < epsilon, other=epsilon)
        df['confidence'] = df[cols].mean(axis=1)
        importances = QuantileComposite.qualify_importances(importances)

        for c, importance in zip(cols, importances):
            if importance == 1:
                p = 1.
            else:
                p = .5
            root = c[:-10]  # Everything except "confidence"
            df[root + 'confimprt'] = df[c].multiply(importance).pow(p)

        cols = [c for c in df.columns if c.endswith('confimprt')]
        df['confidence_total'] = df[cols].sum(axis=1)
        df['confidence_total'] = df['confidence_total'].mask(df['confidence_total'] < epsilon, other=epsilon)

        for c in cols:
            root = c[:-9]
            df[root + 'relative_weight'] = df[c] / df['confidence_total']

        cols = [c for c in df.columns if c.endswith('relative_weight')]
        for c in cols:
            root = c[:-15]
            df[root + 'contribution'] = df[c] * df[root + 'qtile']

        cols = [c for c in df.columns if c.endswith('contribution')]
        df['composite_score'] = df[cols].sum(axis=1)

#         if full:
        return df
        
    def update_fitted_quantiles(self):
        """If sub-model importances have changed, update our fitted_quantiles and quantile_transformer.
        """
        df = QuantileComposite.calculate_score(self.fitted_quantiles_.copy(), self.importances)
        self.quantile_transformer_.fit(df['composite_score'].values.reshape(-1, 1))
    
    def _predict_quantile_composite(self, df_parts):
        """Return the composite quantile and confidence.

        Args:
            df_parts (pd.DataFrame): sub-model quantile results.
            with_confidence (bool, optional): Whether or not to also return the mean confidence score. Defaults to False.

        Returns:
            pd.Series or pd.DataFrame: Quantile scores as well as confidence scores that each range between 0-1.
        """
        df = QuantileComposite.calculate_score(df_parts.copy(), self.importances)
        df['qtile'] = self.quantile_transformer_.transform(df['composite_score'].values.reshape(-1, 1))
        return df
