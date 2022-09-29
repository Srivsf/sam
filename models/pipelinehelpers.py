from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

from dhi.dsmatch.sklearnmodeling.models.featureuniondataframe import FeatureUnionDataFrame


class FeatureNamesPipeline(Pipeline):
    """Pipeline objects do not have a `get_feature_names_out()` method like many transformers.
    This enables Pipelines to have the function by simply taking the feature names of the
    final transformer in the pipeline.

    See: FeatureNamesMixin
    """
    def get_feature_names_out(self, input_features=None):
        idx = -1
        if self.steps[idx][1] in ['passthrough', 'drop']:
            idx -= 1
        return self.steps[idx][1].get_feature_names_out()

def findall_transformertype(root_tx, transformer_type, _already_surveyed=None, _steps_found=None):
    """Search a Transformer object for all subinstances of a particular Transformer type.
    If none are found, returns an empty list.

    This also searches sub-pipelines.

    Args:
        root_tx (BaseEstimator): sklearn BaseEstimator object
        transformer_type (Transformer): Type of Transformer to search for.

    Returns:
        list: List of (name, transformer) of the instances found or [] if nothing found.
    """
    if _already_surveyed is None:
        _already_surveyed = []
    if _steps_found is None:
        _steps_found = []

    if root_tx in ['drop', 'passthrough']:
        return _steps_found
    
    if root_tx in _already_surveyed:
        return _steps_found
    
    _already_surveyed.append(root_tx)
    
    child_transformers = [v for v in vars(root_tx).values() if isinstance(v, BaseEstimator)
                         and not any(root_tx == t[1] for t in _steps_found)]
    for tx in child_transformers:
        found_sub = findall_transformertype(tx, transformer_type, _already_surveyed)
        
        if found_sub:
            found_sub = [t for t in found_sub if not any(t[1] == t2[1] for t2 in _steps_found)]
        if found_sub:
            _steps_found.extend(found_sub)

    if isinstance(root_tx, transformer_type):
        if not any(root_tx == t[1] for t in _steps_found):
            try:
                _steps_found.append((root_tx.name, root_tx))
            except AttributeError:
                _steps_found.append((str(root_tx), root_tx))
        return _steps_found

    if isinstance(root_tx, Pipeline):
        for step in root_tx.steps:
            if isinstance(step[1], transformer_type):
                _steps_found.append(step)
            else:
                found_sub = findall_transformertype(step[1], transformer_type, _already_surveyed)
                if found_sub:
                    _steps_found.extend(found_sub)

    elif isinstance(root_tx, FeatureUnion):
        for step in root_tx.transformer_list:
            if isinstance(step[1], transformer_type):
                _steps_found.append(step)
            else:
                found_sub = findall_transformertype(step[1], transformer_type, _already_surveyed)
                if found_sub:
                    _steps_found.extend(found_sub)

    elif isinstance(root_tx, ColumnTransformer):
        for step in root_tx.transformers_:
            if isinstance(step[1], transformer_type):
                _steps_found.append(step[:2])
            else:
                found_sub = findall_transformertype(step[1], transformer_type, _already_surveyed)
                if found_sub:
                    _steps_found.extend(found_sub)

    return _steps_found

def leftjoin_pipeline(pipe_left: Pipeline, pipe_right: Pipeline, 
                      on: str=None, left_on: str=None, right_on: str=None, 
                      prefixes=('left', 'right')):
    """With two standalone pipelines that share common preprocessing steps, join them
    with a FeatureUnion at the point where they diverge, so that they share the same
    preprocessing Pipeline instance.

    Args:
        pipe_left (Pipeline): [description]
        pipe_right (Pipeline): [description]
        on (str, optional): Name of the last step in the preprocessing pipeline shared by
            `pipe_left` and `pipe_right`. Defaults to None.
        left_on (str, optional): Name of the last step in the preprocessing pipeline of `left_pipeline`.
            Defaults to None.
        right_on (str, optional): Name of the last step in the preprocessing pipeline of `right_pipeline`.
            Defaults to None.
        prefixes (tuple, optional): Prefix names for the branching. Defaults to ('left', 'right').

    Raises:
        ValueError: If the particular step cannot be found in either Pipeline.

    Returns:
        Pipeline: A conjoined pipeline that uses the left's preprocessing pipeline instances and not the
            right's.
    """

    if on is None:
        if left_on is None or right_on is None:
            raise ValueError('Specify a common value for "on" or individual values for "left_on" and "right_on".')
    if on is not None:
        left_on = right_on = on
            
    left_idx = right_idx = -1

    pipelines = findall_transformertype(pipe_left, Pipeline)
    for pipeline in pipelines:
        if isinstance(pipeline, tuple):
            pipeline = pipeline[1]
        if left_on in pipeline.named_steps:
            for left_idx, (pipe_step_name, tx) in enumerate(pipeline.steps):
                if pipe_step_name == left_on:
                    break
            break
            
    if left_idx == -1:
        raise ValueError(f'"{left_on}" not found as a step in the left pipeline.')

    pipelines = findall_transformertype(pipe_right, Pipeline)
    for pipeline in pipelines:
        if isinstance(pipeline, tuple):
            pipeline = pipeline[1]
        if right_on in pipeline.named_steps:
            for right_idx, (pipe_step_name, tx) in enumerate(pipeline.steps):
                if pipe_step_name == right_on:
                    break
            break

    if right_idx == -1:
        raise ValueError(f'"{right_on}" not found as a step in the right pipeline.')

    pre_pipe = FeatureNamesPipeline(pipe_left.steps[:left_idx+1])
    left_pipe = FeatureNamesPipeline(pipe_left.steps[left_idx+1:])
    right_pipe = FeatureNamesPipeline(pipe_right.steps[right_idx+1:])
    fu_tx = FeatureUnionDataFrame([
        (prefixes[0], left_pipe),
        (prefixes[1], right_pipe),
    ])

    pipe_left = FeatureNamesPipeline([
        ('pre', pre_pipe),
        ('branch', fu_tx)
    ])
    
    return pipe_left
