import inspect

from dhi.dsmatch.util.parallel import N_JOBS
from dhi.dsmatch.sklearnmodeling.models.paralleldatatransformer import ParallelDataTransformer
from dhi.dsmatch.sklearnmodeling.models.mixins import DataFrameMixin, FeatureNamesMixin, FilterMixin

class ApplyTransformerCore(ParallelDataTransformer):
    """Apply a function on data as a Transformer, splitting the data in parallel if large enough.

    Example:
        This example creates new columns in a DataFrame, processing `description_bg_parse` and `resume_bg_parse`.

            from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer
            from dhi.dsmatch.sklearnmodeling.functiontransformermapper import applymap
            from dhi.dsmatch.preprocess.bgprocessing import extract_occ_codes

            tx = ApplyTransformer(applymap, extract_occ_codes, 
                    keys={'description_bg_parse': 'description_occ', 'resume_bg_parse', 'resume_occ'})
            # Note that df may have various columns, but `description_bg_parse` and `resume_bg_parse`
            tx.transform(df)

    See `dhi.dsmatch.sklearnmodeling.functiontransformermapper` for various functions that can be applied.

    """
    def __init__(self, apply_func, func, **kwargs):
        """
        Args:
            apply_func (function): The "apply" function, such as `applymap` or other methods in 
                `dhi.dsmatch.sklearnmodeling.functiontransformermapper`.
            func (function): Transforming function applied to the data.
            fkwargs (dict, optional): kwargs needed by `func`. Defaults to {}.
            keys (list or dict, optional): When specified as a list, this acts as a filter to apply the
                transform on only the keys or column names in this list. Note that the output will replace
                the value of this key in the original object. When this is a dict, the key specifies the 
                column to process, but the value will specify an additional column with the transformed outputs.
            use_tqdm (bool, optional): Whether to show the TQDM status bar or not. Defaults to USE_TQDM.
            desc (str, optional): TQDM description label. Defaults to ''.
            n_jobs (int, optional): Number of processors to use during parallel processing. Follows
                joblib/sklearn's modes where -1 is all processors, -2 is all but one, and a positive integer
                is the number of allocated processors. Defaults to N_JOBS.
        """
        super().__init__(**kwargs)
        self.apply_func = apply_func
        self.func = func
        if self.desc == '':
            if isinstance(func, str):
                self.desc = func
            else:
                self.desc = func.__name__

    def __repr__(self):
        if isinstance(self.func, str):
            func_str = self.func
        elif self.func.__name__ == '<lambda>':
            func_str = inspect.getsource(self.func)
            func_str = func_str[func_str.find('lambda'):]
            func_str = func_str[:func_str.rfind(')')]
        else:
            func_str = self.func.__name__
        return f'ApplyTransformer(apply_func={self.apply_func.__name__}, func={func_str})'
        
    @ParallelDataTransformer.parallelize_data
    def transform(self, X, **kwargs):
        """Call the apply function on X.

        Args:
            X (Array, DataFrame, Series, dict, list, sparse_matrix):  
                Data to transform, dimension of [n_samples, n_features].

        Returns:
            X_out : sparse matrix if possible, else a 2-d array
                Transformed input.
        """
        return self.apply_func(X, self.func, **self.fkwargs)

class ApplyTransformer(DataFrameMixin, FeatureNamesMixin, FilterMixin, ApplyTransformerCore):
    _version = '1.1.0'
    # In normal Python programming, we might be able to avoid an __init__ method that calls super() 
    # as we are doing below. However, sklearn transformers do not like *args and their member variables 
    # need to be specified explicitly. We echo the arguments of our "Core" object.
    def __init__(self, apply_func, func, **kwargs):
        super().__init__(apply_func=apply_func, func=func, **kwargs)
