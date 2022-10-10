import contextlib
from functools import partial
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

# Our temporary overload of Pipeline._transform() method.
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/pipeline.py
def _pipe_transform(self, X):
    Xt = X
    for _, name, transformer in self._iter():
        # If has_intermediate, then the transformer can capture intermediate results without those results
        # being in an explicit Pipeline.
        has_intermediate = hasattr(transformer, 'set_intermediate_results_dict')
        if has_intermediate:
            transformer.set_intermediate_results_dict(self.intermediate_results__, name)
            # # This runs Xt through the general pipeline twice. Once to capture subcomponents.
            # # The next to capture regularly.
            # p = transform.pipeline
            # Xt_sub = Xt
            # with intermediate_transforms(p):
            #     Xt_sub = p.transform(Xt_sub)
            #     intermediate_results_sub = p.intermediate_results__

            # for key_sub in intermediate_results_sub.keys():
            #     sub_name = name + '__' + key_sub
            #     if self.intermediate_keys__ is not None:
            #         try:
            #             self.intermediate_keys__.remove(sub_name)
            #             self.intermediate_results__[sub_name] = intermediate_results_sub[key_sub]
            #             if len(self.intermediate_keys__) == 0:
            #                 break
            #         except ValueError:
            #             pass
            #     else:
            #         self.intermediate_results__[sub_name] = intermediate_results_sub[key_sub]

        if isinstance(transformer, Pipeline):  # If a sub-pipeline, dive into it
            with intermediate_transforms(transformer):
                Xt = transformer.transform(Xt)
                intermediate_results_sub = transformer.intermediate_results__

            for key_sub in intermediate_results_sub.keys():
                sub_name = name + '__' + key_sub
                if self.intermediate_keys__ is not None:
                    try:
                        self.intermediate_keys__.remove(sub_name)
                        self.intermediate_results__[sub_name] = intermediate_results_sub[key_sub]
                        if len(self.intermediate_keys__) == 0:
                            break
                    except ValueError:
                        pass
                else:
                    self.intermediate_results__[sub_name] = intermediate_results_sub[key_sub]              
        else:
            if isinstance(transformer, FeatureUnion):  # If a FeatureUnion contains sub-pipelines, dive into them
                for t in transformer.transformer_list:
                    if isinstance(t[1], Pipeline):
                        Xt_sub = Xt#.copy()
                        with intermediate_transforms(t[1]):
                            Xt_sub = t[1].transform(Xt_sub)
                            intermediate_results_sub = t[1].intermediate_results__

                        for key_sub in intermediate_results_sub.keys():
                            sub_name = name + '__' + key_sub
                            if self.intermediate_keys__ is not None:
                                try:
                                    self.intermediate_keys__.remove(sub_name)
                                    self.intermediate_results__[sub_name] = intermediate_results_sub[key_sub]
                                    if len(self.intermediate_keys__) == 0:
                                        break
                                except ValueError:
                                    pass
                            else:
                                self.intermediate_results__[sub_name] = intermediate_results_sub[key_sub]
            
            # If a ColumnTransformer contains sub-pipelines, dive into them
            elif isinstance(transformer, ColumnTransformer):
                for name, subtransformer, cols in transformer.transformers_:
                    if subtransformer == 'passthrough' or name == 'remainder':
                        continue
                    if isinstance(subtransformer, Pipeline):
                        Xt_sub = Xt#.copy()
                        with intermediate_transforms(subtransformer):
                            Xt_sub = subtransformer.transform(Xt_sub[cols])
                            intermediate_results_sub = subtransformer.intermediate_results__

                        for key_sub in intermediate_results_sub.keys():
                            sub_name = name + '__' + key_sub
                            if self.intermediate_keys__ is not None:
                                try:
                                    self.intermediate_keys__.remove(sub_name)
                                    self.intermediate_results__[sub_name] = intermediate_results_sub[key_sub]
                                    if len(self.intermediate_keys__) == 0:
                                        break
                                except ValueError:
                                    pass
                            else:
                                self.intermediate_results__[sub_name] = intermediate_results_sub[key_sub]

            Xt = transformer.transform(Xt)
        if self.intermediate_keys__ is not None:
            try:
                self.intermediate_keys__.remove(name)
                self.intermediate_results__[name] = deepcopy(Xt)
                if len(self.intermediate_keys__) == 0:
                    break
            except ValueError:
                pass
        else:
            self.intermediate_results__[name] = deepcopy(Xt)

        if has_intermediate:
            delattr(transformer, 'intermediate_results__')
            delattr(transformer, 'intermediate_results_name__')

    return Xt

@contextlib.contextmanager
def intermediate_transforms(pipe: Pipeline, keys: list=[], bypass_list: list=[]):
    """Allows for the retrieval of all or parts of the transformations in a
    sklearn Pipeline, as well as the ability to dynamically bypass parts of
    the pipeline.

    Within the context, intermediate results are available as a dict, `pipe.intermediate_results__`
    with keys equal to the names of the transformers in the pipeline. This dict is a temporary structure
    and only available within the context.

    Args:
        pipe (Pipeline): [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) 
            object. 
        keys (list, optional): List of pipeline object to retrieve. If empty, then
            all are available in the pipe's `intermediate_results__` within this
            context. Otherwise, just the names listed are captured. Defaults to [].
        bypass_list (list, optional): List of names to bypass. When not empty, this is
            the set of names in the pipeline to "passthrough" if these names are at
            in a pipeline or "drop" if in a FeatureUnion. Defaults to [].

    Example:

        We instantiate a simple pipeline.

            import numpy as np
            import pandas as pd
            from sklearn.pipeline import Pipeline
            from sklearn.base import TransformerMixin

            df = pd.DataFrame({'name': ['Anne', 'Bob', 'Charlie', 'Bob'],
                            'age': [20, 21, 22, 23]})

            class LowercaseTransformer(TransformerMixin):
                def transform(self, X):
                    return X.apply(lambda x: x.lower())

            class UppercaseTransformer(TransformerMixin):
                def transform(self, X):
                    return X.apply(lambda x: x.upper())

            class CamelcaseTransformer(TransformerMixin):
                def transform(self, X):
                    return X.apply(lambda x: x[0].upper() + x[1:].lower())

            class ReverseTransformer(TransformerMixin):
                def transform(self, X):
                    try:
                        return X.applymap(lambda x: x[::-1])
                    except AttributeError:
                        return X.apply(lambda x: x[::-1])

            lct = LowercaseTransformer()
            uct = UppercaseTransformer()
            cct = CamelcaseTransformer()
            rt = ReverseTransformer()
            pipe = Pipeline([('lower', lct), ('upper', uct), ('reverse', rt), ('camel', cct), ('last', 'passthrough')])

        Then we can execute some contexts.

            # To retrieve all intermediate results...
            with intermediate_transforms(pipe):
                Xt = pipe.transform(df['name'])
                intermediate_results = pipe.intermediate_results__

        Outputs:

            {'lower': 0       anne
            1        bob
            2    charlie
            3        bob
            Name: name, dtype: object,
            'upper': 0       ANNE
            1        BOB
            2    CHARLIE
            3        BOB
            Name: name, dtype: object,
            'reverse': 0       ENNA
            1        BOB
            2    EILRAHC
            3        BOB
            Name: name, dtype: object,
            'camel': 0       Enna
            1        Bob
            2    Eilrahc
            3        Bob
            Name: name, dtype: object}

        To retrieve the first few steps, we can execute the following. Note that in this case, 
        the order of the keys does not matter, but the returned transform, `Xt`, will be the results 
        of the last transformer in our keys. And `intermediate_results` contains only the keys of interest.

            with intermediate_transforms(pipe, keys=['upper', 'lower']):
                Xt = pipe.transform(df['name'])
                intermediate_results = pipe.intermediate_results__

        This provides:

            {'lower': 0       anne
            1        bob
            2    charlie
            3        bob
            Name: name, dtype: object,
            'upper': 0       ANNE
            1        BOB
            2    CHARLIE
            3        BOB
            Name: name, dtype: object}

        To bypass/passthrough/drop transformers, we can execute this context. This may be useful
        in FeautureUnions to ignore some paths. However, skipping transformations is likely to give 
        unexpected final results.

            with intermediate_transforms(pipe, bypass_list=['camel']):
                Xt = pipe.transform(df['name'])
                intermediate_results = pipe.intermediate_results__

        This provides the following output:

            {'lower': 0       anne
            1        bob
            2    charlie
            3        bob
            Name: name, dtype: object,
            'upper': 0       ANNE
            1        BOB
            2    CHARLIE
            3        BOB
            Name: name, dtype: object,
            'reverse': 0       ENNA
            1        BOB
            2    EILRAHC
            3        BOB
            Name: name, dtype: object}

    """
    if not isinstance(pipe, Pipeline):
        raise ValueError(f'Variable pipe must be a Pipeline, not {pipe.__class__.__name__}')

    pipe.intermediate_results__ = {}
    pipe.intermediate_keys__ = None
    if keys:
        pipe.intermediate_keys__ = keys
    
    before_list_objs = {}
    bypass_list_objs = {}
    if bypass_list:
        params = pipe.get_params()
        for k in bypass_list:
            if k in params:
                before_list_objs[k] = params[k]
                if 0 < k.find('__') < len(k):
                    bypass_list_objs[k] = 'drop'  # FeatureUnion bypass
                else:
                    bypass_list_objs[k] = 'passthrough'  # Pipeline bypass
        if bypass_list_objs:
            pipe.set_params(**bypass_list_objs)
                          
    _transform_before = pipe.transform
    pipe.transform = partial(_pipe_transform, pipe)  # Monkey-patch our _pipe_transform method.
    yield pipe  # Release our patched object to the context
    
    # Restore
    pipe.transform = _transform_before
    if before_list_objs:
        pipe.set_params(**before_list_objs)
    delattr(pipe, 'intermediate_results__')
    delattr(pipe, 'intermediate_keys__')
