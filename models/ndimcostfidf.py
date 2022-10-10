from itertools import chain

import numpy as np
from scipy.sparse import hstack

import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin

from dhi.dsmatch.sklearnmodeling.models.custombase import CustomClassifier
from dhi.dsmatch.sklearnmodeling.models.quantilehelpers import QuantileComposite, QuantilePredictMixin
from dhi.dsmatch.sklearnmodeling.models.cleanstemtransformer import make_cleanstem_pipeline
from dhi.dsmatch.sklearnmodeling.models.coupledtransformer import CoupledTransformer
from dhi.dsmatch.sklearnmodeling.models.cosinesimilaritytransformer import CosineSimilarityTransformer

class NDimCosTfidf(QuantilePredictMixin, CustomClassifier, TransformerMixin, QuantileComposite):
    """An unsupervised model that takes pairs of string columns such as `resume` and `job_description`,
    trains a TfidfVectorizer with those documents, and then permits the cosine similarity of an arbitrary
    resume and job_description running through the vectorizer independently. For any new pair, the percentile
    of their cosine similarity is captured and thresholded.

    Note that any joblib warnings are related to the caching and should not be a problem.

    See: train-test-ndimcosinetfidf-model.ipynb in the notebooks for example training and analysis.

    """
    _version = '1.0.0'

    def __init__(self, cosine_pairs=(('resume', 'job_description'), ('resume', 'job_title')), 
                corpus_cols=('resume', ['job_title', 'job_description']),
                vectorizer=None, min_df=.01, ngram_range=(1,1), prediction_thresholds=None, 
                default_transform_mode='predict_quantile', memory=None):
        """
        Args:
            cosine_pairs (tuple, optional): Dimensions to capture cosine similarity. 
                Defaults to (('resume', 'job_description'), ('resume', 'job_title')).
            corpus_cols (tuple, optional): Columns to include in the corpus. The corpus is comprised of these columns
                all aggregated, after dropping duplicates per column. If an item in the tuple is a list, the columns 
                in that list are concatenated as belonging to a given document. 
                Defaults to ('resume', ['job_title', 'job_description'])
            vectorizer (TfidfVectorizer, optional): If using a previously trained TfidfVectorizer, this can be loaded
                and passed in. Otherwise, if None, a new vectorizer is created and trained.
            min_df (float): If a vectorizer is trained, this is its `min_df` parameter. Default is 0.01, which means
                a term has to be in at least 1% of the documents to be part of the vectorizer's vocabulary.
            ngram_range (float): If a vectorizer is trained, this is its `ngram_range` parameter. Default is (1, 1),
                which is a one-gram.
            prediction_thresholds (list): List of floats to specify match boundaries of quantiles during predict.
                Default is [.1, .3, .7, .9] -- everything less than 0.1 is a 1, .1 to .3 is a 2, .3 to .7 is a 3,
                .7 to .9 is a 4 and over .9 is a 5.
            default_transform_mode: Default transform_mode. See `transform()`. Default is `predict_quantile`.
            memory (Memory object or str, optional): If specified, the Memory object or directoy location where 
                cached results can be written to. This is only useful when training, but can make preprocessing faster
                if one is training multiple times. Defaults to None.
        """
        self.feature_names_out = []
        for f in chain(*cosine_pairs):
            if f not in self.feature_names_out:
                self.feature_names_out.append(f)

        self.cosine_pairs = cosine_pairs
        self.corpus_cols = corpus_cols
        self.min_df = min_df  # Technically, this isn't used, but sklearn demands all passed arguments map to self.
        self.ngram_range = ngram_range

        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
        else:
            self.vectorizer = vectorizer

        self.memory = memory
        self.resolution = 1000
        self.default_transform_mode = default_transform_mode
        
        self.cleanstem = make_cleanstem_pipeline(self.feature_names_out, memory=memory)
        
        self.exposed_models = []  # CosineSimilarityTransformer models that are exposed to ensemble
        for pair in cosine_pairs:
            cossim_model = CosineSimilarityTransformer(columns=list(pair))
            cossim_model.name = 'cossim-' + '_'.join(pair)
            cossim_model.importance = 1/len(cosine_pairs)
            self.exposed_models.append(cossim_model)
        
        QuantilePredictMixin.__init__(self, prediction_thresholds)
        self._fitting = False
        
    @property
    def importances(self):
        return [m.importance for m in self.exposed_models]

    def _fit_preprocess(self, X, y=None):
        """Make a cleaned, stemmed corpus for training the vectorizer.

        Args:
            X (pd.DataFrame): Input DataFrame that contains the columns in `self.corpus_cols`.

        Returns:
            pd.Series: Series where each row is a cleaned and stemmed document for the TfidfVectorizer.
        """
        assert(isinstance(X, pd.DataFrame))
        self._fitting = True
        df_cleaned = self.cleanstem.fit_transform(X)  # Call fit_transform instead of fit to take advantage of caching.
        s_corpus = None
        for c in self.corpus_cols:
            s = df_cleaned[c]
            s = s.drop_duplicates()
            if isinstance(c, list):
                ct = CoupledTransformer(c, desc=f'coupling {"-".join(c)}')
                s = ct.transform(s)
            s_corpus = pd.concat([s_corpus, s], ignore_index=True)

        df_cleaned = None  # Free some memory

        s_corpus = s_corpus.str.replace(r'[^\x00-\x7F]', ' . ')
        s_corpus = s_corpus.str.replace('.', ' . ')
        s_corpus = s_corpus.str.replace('\n', '\r')

        return s_corpus
        
    def fit_transform(self, X, y=None, **kwargs):
        """Given data that contains the columns of interest, clean and stem, couple into documents, fit 
        Tfidfvectorizers, then get a distribution for thresholds so that we can make a prediction.

        Generally, this function should be used over `fit()` since `fit_transform()` caches intermediate results.

        Note that some parts of the pipeline contain status bars and others do not. We have found it can take over 
        30 minutes to fit 165K records on 16 processors.

        Args:
            X (DataFrame): DataFrame that needs to contain the columns specified in the initialization pairs.
        """
        self._fitting = True
        s_corpus = self._fit_preprocess(X, y)
        self.vectorizer.fit_transform(s_corpus)
        s_corpus = None  # Free some memory

        X = self.transform(X)
        self.quantile_transformer_ = QuantileTransformer(n_quantiles=1000, random_state=0)
        self.update_fitted_quantiles()
        self.transform_cols = X.columns

        return X
        
    def _get_pairs_locs(self, X):
        """Given a DataFrame, get the column indices of our `self.cosine_pairs`.

        Args:
            X (pd.DataFrame): DataFrame that must contain the columns in `self.cosine_pairs`

        Returns:
            list: list of pairs of column indices that correspond with `self.cosine_pairs`.
        """
        pairs_locs = []
        for pair in self.cosine_pairs:
            pairs_locs.append([X.columns.get_loc(c) for c in pair])
        return pairs_locs
    
    def _transform_clean(self, X):
        """Clean and stem a DataFrame

        Args:
            X (pd.DataFrame): DataFrame that must contain the columns in `self.cosine_pairs`

        Returns:
            pd.DataFrame: DataFrame with the same columns as those passed in, but with columns cleaned and stemmed.
        """
        if self._fitting:
            X = self.cleanstem.fit_transform(X)  # Pull cached when training because this was the first thing we did
        elif X.shape[0] >= 100:
            X = self.cleanstem.fit_transform(X)  # Try pulling from cached
        else:
            X = self.cleanstem.transform(X)
        return X

    def transform(self, X, y=None, **kwargs):
        """Given data that contains the features of interest, clean and stem those columns, then run each through
        the TfidfVectorizers and with each pair, get a cosine similarity for each pair and then a composite
        value that is a floating point number between 0-1.

        Args:
            X (DataFrame): DataFrame that contains our self.feature_names_out columns.

            transform_mode (str): kwargs may contain this argument, which may be one of `clean`, `vectors`, `cossims`, 
                `quantile_parts`, or `analytic` that produce the following:
            * `percent_quantile`: two-column DataFrame of `qtile` and `confidence`.
            * `cossims`: One-column vector of cosine similarity scores.
            * `analytic`: Cosine similarities, quantiles, number of nonzero features of the two columns and the minimum
                as a DataFrame.
            * `tuples`: Same as `analytic`, but in tuple form instead of a formatted DataFrame.
            Default is `percent_quantile`.
            with_confidence: (bool): Whether or not to return the confidence with the quantile score. Default is True.

        Returns:
            This varies depending on the transform_mode. The default behavior is a two-column DataFrame of `qtile` and 
            `confidence`.
        """
        transform_mode = kwargs.pop('transform_mode', self.default_transform_mode)
        pairs_locs = self._get_pairs_locs(X)
        n_columns = X.shape[1]
        X = self._transform_clean(X)
        if transform_mode == 'clean':
            self.transform_cols = X.columns
            return X
        
        # sparse matrix where rows = n_columns * length of original X, columns are the features.
        X = self.vectorizer.transform(X.values.ravel())
        if transform_mode == 'vectors':
            self.transform_cols = None
            return X

        if self._fitting or transform_mode == 'cossims':
            results = None
            if self._fitting:
                self.fitted_quantiles_ = None
            for locs, cossim_model in zip(pairs_locs, self.exposed_models):
                # Create a sparse matrix of the features of interest that has
                # the original number of rows to process and where each row has
                # two halves of feature vectors. 
                v = hstack((X[locs[0]::n_columns], X[locs[1]::n_columns])).tocsr()
                if self._fitting:
                    result = cossim_model.fit_transform(v, transform_mode='predict_quantile')
                else:
                    result = cossim_model.transform(v, transform_mode='analytic')
                                        
                cols = [cossim_model.name + '_' + c for c in result.columns]
                result.columns = cols                
                results = pd.concat([results, result], axis=1)
            
            if self._fitting:      
                self._fitting = False
                cols = [c for c in results.columns if c.endswith('qtile') or c.endswith('confidence')]
                self.fitted_quantiles_ = results[cols]
            
            self.transform_cols = results.columns
            return results

        results = None
        for locs, cossim_model in zip(pairs_locs, self.exposed_models):
            # Create a sparse matrix of the features of interest that has
            # the original number of rows to process and where each row has
            # two halves of feature vectors. 
            v = hstack((X[locs[0]::n_columns], X[locs[1]::n_columns])).tocsr()
            result = cossim_model.transform(v, transform_mode='analytic')
            cols = [cossim_model.name + '_' + c for c in result.columns]
            result.columns = cols
            results = pd.concat([results, result], axis=1)

        if transform_mode == 'quantile_parts':
            self.transform_cols = results.columns
            return results

        X = self._predict_quantile_composite(results)
        self.transform_cols = X.columns
        return X
        
    def make_lean(self):
        """Delete unnecessary model data to make smaller memory."""
        for model in self.exposed_models:
            model.make_lean()
