import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer

class FeatureNamesMixin:
    """Permits a Transformer to have a `get_feature_names_out()` function. Some sklearn transformers have this
    functionality and others do not. For use with the 
    [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) and 
    [`FeatureUnion`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) transformers, 
    when trying to query them with `get_feature_names_out()`, submodels may throw errors if they do not have feature_names_out.

    If `feature_names_out` is specified during Transformer initialization, then this list of names is always used.
    Otherwise, if the return type of `fit_transform()` or `transform()` is a Pandas DataFrame, then the 
    feature names are set to the column names of the resulting DataFrame. If unspecified and the inputs or outputs
    are Numpy arrays, then the transformer will work as normal and not support `get_feature_names_out()`.

    Example:

    sklearn's 
    [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
    does not support `get_feature_names_out()`, so when creating a ColumnTransformer as per their
    [demo](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html),
    the resulting transformer will throw an error when trying to call `get_feature_names_out()`. The code below rectifies
    this.

    ```python
    # Compare with https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html
        
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.compose import make_column_selector
    import pandas as pd
    import numpy as np

    from dhi.dsmatch.sklearnmodeling.models.mixins import FeatureNamesMixin

    class StandardScalerNames(FeatureNamesMixin, StandardScaler):
        pass

    X = pd.DataFrame({'city': ['London', 'London', 'Paris', 'Sallisaw'],
                    'rating': [5, 3, 4, 5]
                    })  
    ct = make_column_transformer(
        (StandardScalerNames(),
        make_column_selector(dtype_include=np.number)),  # rating
        (OneHotEncoder(),
        make_column_selector(dtype_include=object)))  # city
    Xt = ct.fit_transform(X) 
    feature_names_out = ct.get_feature_names_out()
    print(pd.DataFrame(Xt, columns=feature_names_out))
    ```

    This provides the following output:

    ```
    standardscalernames__rating  onehotencoder__x0_London  \
    0                     0.904534                       1.0   
    1                    -1.507557                       1.0   
    2                    -0.301511                       0.0   
    3                     0.904534                       0.0   

    onehotencoder__x0_Paris  onehotencoder__x0_Sallisaw  
    0                      0.0                         0.0  
    1                      0.0                         0.0  
    2                      1.0                         0.0  
    3                      0.0                         1.0  
    ```

    """
    def __init__(self, *args, feature_names_out=None, **kwargs):
        self.feature_names_out = feature_names_out
        super().__init__(*args, **kwargs)
        
    def _postprocess(self, ret, X):
        if self.feature_names_out is not None:
            self.feature_names_out_ = self.feature_names_out
        elif isinstance(ret, pd.DataFrame):
            self.feature_names_out_ = [str(x) for x in ret.columns]
        elif isinstance(X, pd.DataFrame):
            self.feature_names_out_ = [str(x) for x in X.columns]

    def _fitter(self, X, fit_name, y=None, **kwargs):
        super_fit = getattr(super(), fit_name)
        if super_fit and callable(super_fit):
            ret = super_fit(X, y=y, **kwargs)
        self._postprocess(ret, X)
        return ret
            
    def fit(self, X, y=None, **kwargs):
        ret = self._fitter(X, 'fit', y=y, **kwargs)
        return ret
        
    def fit_transform(self, X, y=None, **kwargs):
        ret = self._fitter(X, 'fit_transform', y=y, **kwargs)
        return ret
    
    def transform(self, X):
        super_transform = getattr(super(), 'transform')
        if super_transform and callable(super_transform):
            ret = super_transform(X)        
        self._postprocess(ret, X)
        return ret

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

class DataFrameMixin:
    """Works with the FeatureNamesMixin (and should be to the left of it when defining a class) to 
    call `get_feature_names_()` of the transformer and used those columns.

    Example:

    ```python
    df = pd.DataFrame({'first_name': ['Anne', 'Bob', 'Charlie', 'Bob'],
                   'last_name': ['Bancroft', 'Dylan', 'Chaplin', 'Marley'],
                   'age': [20, 21, 22, 23],
                   })

    class FilterFunctionTransformer(DataFrameMixin, FilterMixin, FunctionTransformer):
        pass

    def strcat_cols(df, fill=''):
        return [f'{fill}'.join(row[1:]) for row in df.itertuples()]

    concat_tx = FilterFunctionTransformer(
        strcat_cols, 
        keys={('first_name', 'last_name'): 'concat'},
        feature_names_out=FilterFunctionTransformer.calling_feature_names_out,
        kw_args=dict(fill=' ')
    )
    concat_tx.called_feature_names_out_ = ['age', 'concat']
    concat_tx.transform(df.copy())
    ```

    In the above example, the columns kept are ['age', 'concat']. If feature_names_out is undefined,
    then all columns of the DataFrame are provided as output.
    """
    def _post_process(self, Xt):
        try:
            columns = self.get_feature_names_out()
            columns = [c[:-2] if c.endswith('__') else c for c in columns]

            has_multiindex = any(c.startswith('(') and c.endswith(')') for c in columns)
            if has_multiindex:
                columns = [eval(c) if c.startswith('(') and c.endswith(')') else c for c in columns]

            if isinstance(Xt, np.ndarray):
                return pd.DataFrame(Xt, columns=columns)

            if len(set(columns).difference(set(Xt.columns))) > 0:
                return pd.DataFrame(Xt.values, columns=columns)

            return Xt[columns]
        except:
            pass
        return Xt

    def fit_transform(self, X, y=None, **kwargs):
        Xt = super().fit_transform(X, y=y, **kwargs)
        return self._post_process(Xt)

    def transform(self, X):
        Xt = super().transform(X)
        return self._post_process(Xt)

class IntermediateTransformerMixin:
    """Works with the `intermediate_transforms` context to permit transformers to submit their own
    subset of intermediate steps instead of forcing each transformer to be just one substep.

    When used in this context, there will be a `self.intermediate_results__` and `self.intermediate_results_name__`
    attributes that are available to store information during `transform()`. These attributes are only available
    within the intermediate_transforms context.

    Example:

    ```python

    class MyIntTransformer(IntermediateTransformerMixin, MyTransformer):
        
        .
        .
        .

        def transform(self, X):
            vals = self.operation_on_data(X)
            try:
                self.intermediate_results__[self.intermediate_results_name__ + '__vals'] = vals
            except AttributeError:
                pass
            Xt = self.some_other_operation(X, vals)
            return Xt
    ```

    """
    def set_intermediate_results_dict(self, intermediate_results: dict, name: str):
        self.intermediate_results__ = intermediate_results
        self.intermediate_results_name__ = name

class RandomRightMixin:
    """For getting and retrieving random samples of the right column of a training set.

    This is useful for transformers that build a NULL set that compares the left column
    with a random sample of right values.
    """
    @staticmethod
    def extract_X0X1(X):
        if isinstance(X, tuple):
            X0, X1 = X
        elif isinstance(X, dict):
            X0, X1, *_ = X.values()
        else:
            if isinstance(X, pd.DataFrame):
                X0 = X.iloc[:, 0]
                X1 = X.iloc[:, 1]
            else:
                X0 = X[:, 0]
                X1 = X[:, 1]
        return X0, X1

    def __init__(self, n_confidence_samples=1000, **kwargs):
        self.n_confidence_samples = n_confidence_samples
        super().__init__(**kwargs)
        
    def set_X1_random_(self, X):
        X0, X1 = RandomRightMixin.extract_X0X1(X)

        np.random.seed(self.n_confidence_samples + 3)
        self.n_confidence_samples_ = np.min([self.n_confidence_samples, X1.shape[0]])
        indices = np.arange(X1.shape[0])
        np.random.shuffle(indices)
        self.X1_random_ = X1[list(indices)[:self.n_confidence_samples_]]

    def _fitter(self, X, fit_name, y=None, **kwargs):
        self.set_X1_random_(X)
        super_fit = getattr(super(), fit_name)
        if super_fit and callable(super_fit):
            ret = super_fit(X, y=y, **kwargs)
            return ret
            
    def fit(self, X, y=None, **kwargs):
        ret = self._fitter(X, 'fit', y=y, **kwargs)
        return ret
        
    def fit_transform(self, X, y=None, **kwargs):
        ret = self._fitter(X, 'fit_transform', y=y, **kwargs)
        return ret
    
    def get_random_examples(self):
        return self.X1_random_

class RandomRightFunctionTransformer(RandomRightMixin, FunctionTransformer):
    pass

class FilterMixin:
    """Permits a subset of columns to be fitted and transformed as well as for new columns or
    dictionary keys to be added to the inputs to be transformed.

    To use, this class should precede the default base class. The following example shows the

    ```python
    class MyTransformer(FilterMixin, BaseTransformer):
        pass
    ```

    Then, to use, specify the keys (i.e., columns) of interest. Say one has a DataFrame with 
    columns ['A', 'B', 'C', 'D', 'E'], but we want to only apply the transformer on 'A' and 'C':

    ```python
    my_tx = MyTransformer(keys=['A', 'C'])
    df = my_tx.transform(df)  # Produces a DataFrame with columns=['A', 'B', 'C', 'D', 'E'] with A and C changed.
    ```

    Note that like other transformations, the contents of columns ['A', 'C'] are replaced with
    the transformations, but 'B' and 'D' remain untouched. If one wants to make a new column
    with the transformations, keeping the original columns, then make keys a dict where the 
    key is the name of the column to process and the value is the new column.

    ```python
    my_tx = MyTransformer(keys={'A': 'new_A'}, 'C': 'new_C'})
    df = my_tx.transform(df)  # Produces a DataFrame with columns=['A', 'B', 'C', 'D', 'E', 'new_A', 'new_C']
    ```

    A transformation may also take multiple inputs and provide a single column output. In this case,
    the keys that are grouped are made into a tuple.

    ```python
    my_tx = MyTransformer(keys={('A', 'B'): 'A_and_B'}, ('C', 'D'): 'C_and_D'})
    df = my_tx.transform(df)  # Produces a DataFrame with columns=['A', 'B', 'C', 'D', 'E', 'A_and_B', 'C_and_D']
    ```

    Used with the DataFrameMixin and FeatureNamesMixin, we can also filter the output columns:

    ```python
    class MyTransformer(DataFrameMixin, FeatureNamesMixin, FilterMixin, BaseTransformer):
        pass
    my_tx = MyTransformer(keys={('A', 'B'): 'A_and_B'}, ('C', 'D'): 'C_and_D'},
            feature_names_out=['E', 'A_and_B', 'C_and_D'])
    df = my_tx.transform(df)  # Produces a DataFrame with columns=['E', 'A_and_B', 'C_and_D']
    ```

    Similarly, a column may split into new columns.
    
    ```python
    my_tx = MyTransformer(keys={'A': ('A1', 'A2'), 'C': ('C1', 'C2')})
    my_tx.transform(df)
    ```

    """
    def __init__(self, *args, keys=None, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)
        
    def fit(self, X, y=None, **kwargs):
        if self.keys is None:
            return super().fit(X, y=y, **kwargs)
        
        if isinstance(X, dict):
            for k, v in X.items():
                if k in self.keys:
                    super().fit(v, y=y, **kwargs)
            return self

        if isinstance(X, pd.Series):
            X = X.to_frame()     
            
        if isinstance(X, pd.DataFrame):
            if isinstance(self.keys, dict):
                for c in self.keys:
                    if isinstance(c, tuple):
                        if all(isinstance(x, int) for x in c):
                            super().fit(X.iloc[:, list(c)], y=y, **kwargs)
                        else:
                            super().fit(X.loc[:, list(c)], y=y, **kwargs)
                    else:
                        if isinstance(c, int):
                            if X.iloc[:, c].ndim == 1:
                                super().fit(X.iloc[:, c].values.reshape(-1, 1), y=y, **kwargs)
                            else:
                                super().fit(X.iloc[:, c], y=y, **kwargs)
                        else:
                            if X.loc[:, c].ndim == 1:
#                                 super().fit(X.loc[:, c].values.reshape(-1, 1), y=y, **kwargs)
                                super().fit(X.loc[:, c].to_frame(), y=y, **kwargs)
                            else:
                                super().fit(X.loc[:, c], y=y, **kwargs)
            else:
                if all(isinstance(x, int) for x in self.keys):
                    super().fit(X.iloc[:, self.keys], y=y, **kwargs)
                else:        
                    super().fit(X.loc[:, self.keys], y=y, **kwargs)          
                
        return self

    def transform(self, X):
        if self.keys is None:
            return super().transform(X)
        
        if isinstance(X, dict):
            X_new = {}
            for k, v in X.items():
                if k in self.keys:
                    if isinstance(self.keys, dict):
                        X_new[self.keys[k]] = super().transform(v)
                    else:
                        X_new[k] = super().transform(v)
            X.update(X_new)
            return X
        
        if isinstance(X, pd.Series):
            X = X.to_frame()
            
        if isinstance(X, pd.DataFrame):
            if isinstance(self.keys, dict):
                for c in self.keys:
                    if isinstance(c, tuple):
                        if all(isinstance(x, int) for x in c):
                            X[self.keys[c]] = super().transform(X.iloc[:, list(c)])
                        else:
                            if isinstance(self.keys[c], tuple):
                                out_cols = list(self.keys[c])
                                for x in out_cols:
                                    X[x] = None
                            else:
                                out_cols = self.keys[c]
                            X[out_cols] = super().transform(X.loc[:, c])
                    else:
                        if isinstance(c, int):
                            if X.iloc[:, c].ndim == 1:
                                X[self.keys[c]] = super().transform(X.iloc[:, c].to_frame())
                            else:
                                X[self.keys[c]] = super().transform(X.iloc[:, c])
                        else:
                            if isinstance(self.keys[c], tuple):
                                out_cols = list(self.keys[c])
                                for x in out_cols:
                                    X[x] = None
                            else:
                                out_cols = self.keys[c]
                            if X.loc[:, c].ndim == 1:
                                X[out_cols] = super().transform(X.loc[:, c].to_frame())
                            else:
                                X[out_cols] = super().transform(X.loc[:, c])
            else:
                if all(isinstance(x, int) for x in self.keys):
                    X.loc[:, self.keys] = super().transform(X.iloc[:, self.keys])
                else:   
                    X.loc[:, self.keys] = super().transform(X.loc[:, self.keys])       
        return X

class KeysQuantileTransformer(FilterMixin, QuantileTransformer):
    # In normal Python programming, we might be able to avoid an __init__ method that calls super() 
    # as we are doing below. However, sklearn transformers do not like *args and their member variables 
    # need to be specified explicitly. We echo the arguments of our "Core" object.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FilterFunctionTransformer(DataFrameMixin, FilterMixin, FunctionTransformer):
    """Allow for sklearn's [FunctionTransformer]
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
    functionaltiy, but permit filtering and ensure that the outputs are a DataFrame.

    The following example shows several elements:

    ```python
    df = pd.DataFrame({
        'first_name': ['Anne', 'Bob', 'Charlie', 'Bob'],
        'last_name': ['Bancroft', 'Dylan', 'Chaplin', 'Marley'],
        'age': [20, 21, 22, 23],
    })

    def strcat_cols(df, fill=''):
        return [f'{fill}'.join(row[1:]) for row in df.itertuples()]

    concat_tx = FilterFunctionTransformer(
        strcat_cols,
        keys={('first_name', 'last_name'): 'concat'},
        feature_names_out=FilterFunctionTransformer.calling_feature_names_out,
        kw_args=dict(fill=' ')
    )
    concat_tx.called_feature_names_out_ = ['age', 'concat']
    concat_tx.transform(df.copy())
    ```

    **keys:** In the example above, `keys` is a single-element dict. Filtering works to take the keys as input
    and the values as output. In this case, with a tuple as the key, 'first_name' and 'last_name' 
    serve as the inputs to the function and 'concat' is the name of the output.

    **feature_names_out:** In this particular example, we specify the output from this function to be
    ['age', 'concat']. That is, we concatenated `first_name` and `last_name` so we do not need them anymore.
    Strangely, when the `feature_names_out()` method was implemented in Sklearn v. 1.1, it requires the
    output column names to be in the form of a callable that takes two arguments: the transformer (like self),
    and `feature_names_in`. This cannot be a lambda function because the model needs to be picklable.
    Therefore, before calling `transform()` we must set an attribute called `called_feature_names_out_` and
    then we can pass the `FilterFunctionTransformer.calling_feature_names_out` as the argument to retrieve
    the output column names desired.
    """
    # In normal Python programming, we might be able to avoid an __init__ method that calls super() 
    # as we are doing below. However, sklearn transformers do not like *args and their member variables 
    # need to be specified explicitly. We echo the arguments of our "Core" object.
    def __init__(self, func=None, inverse_func=None, **kwargs):
        super().__init__(func=func,
                         inverse_func=inverse_func,
                         **kwargs)

    @staticmethod
    def calling_feature_names_out(trans, feature_names_in):
        return trans.called_feature_names_out_
