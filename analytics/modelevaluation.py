import time
from io import StringIO
from typing import List, Tuple

import pandas as pd
import numpy as np

def labeled_xtab(df: pd.DataFrame, pred_col: str='pred', labeled_col: str='match_score', 
        rownames: list=None, colnames: list=None, sorted_rows: bool=True, sorted_cols: bool=True) -> pd.DataFrame:
    """Retrieve the cross tabulation of predicted values (rows) vs labeled values (columns)

    Args:
        df (pd.DataFrame): DataFrame that has a predicted column as well as a labeled column.
        pred_col (str, optional): Name of the predicted column. Defaults to 'pred'.
        labeled_col (str, optional): Name of the labeled column. Defaults to 'match_score'.
        rownames (list): If set, ensure that the dataframe contains these rows. If these rows are otherwise
            absent, then an index will be inserted with zeros for all values in the row. Defaults to None.
        colnames (list): If set, ensure that the dataframe contains these column. If these columns are otherwise
            absent, then a column will be inserted with zeros for all values in the column. Defaults to None.
        sorted_rows (bool): Only used if rownames is set and new rows are inserted. Defaults to True.
        sorted_cols (bool): Only used if colnames is set and new columns are inserted. Defaults to True.

    Returns:
        pd.DataFrame: The sum of matches between predicted and labeled.
    """
    df_xtab = pd.crosstab(df[pred_col], df[labeled_col])
    try:
        excluded = set(rownames).difference(df_xtab.index)
        for r in excluded:
            df_xtab.loc[r, :] = 0
        df_xtab.sort_index(inplace=True)
    except TypeError:  # Will be thrown if rownames is None, which is default, so we ignor.
        pass

    try:
        excluded = set(colnames).difference(df_xtab.columns)
        for c in excluded:
            df_xtab.loc[:, c] = 0
        df_xtab.sort_index(axis=1, inplace=True)
    except TypeError:  # Will be thrown if rownames is None, which is default, so we ignor.
        pass
    
    return df_xtab


def aggregate_stats_from_xtab(df_xtab: pd.DataFrame):
    """
    From a crosstab DataFrame, retrieve some statistics.

    Args:
        df_xtab (pd.DataFrame): As retrieved via `labeled_xtab`.
    """
    n_correct = np.diagonal(df_xtab).sum()
    n_records = df_xtab.values.sum()
    absolute_accuracy = n_correct / n_records
    one_half_amount = n_correct
    one_half_amount += np.diagonal(df_xtab, offset=1).sum() * .5
    one_half_amount += np.diagonal(df_xtab, offset=-1).sum() * .5
    one_half_accuracy = one_half_amount / n_records
    
    gaussian_amount = n_correct
    for i in range(1, 3):
        gaussian_amount += np.diagonal(df_xtab, offset=i).sum() * np.exp(-(i**2)/2)
        gaussian_amount += np.diagonal(df_xtab, offset=-i).sum() * np.exp(-(i**2)/2)
    gaussian_accuracy = gaussian_amount / n_records

    return dict(
        n_records=n_records,
        n_correct=n_correct,
        absolute_accuracy=absolute_accuracy,
        one_half_accuracy=one_half_accuracy,
        gaussian_accuracy=gaussian_accuracy
    )

def print_aggregate_stats(stats_dict: dict):
    """Given a dict from `aggregate_stats_from_xtab()`, print it out in a nicely formatted way.

    Args:
        stats_dict (dict): As acquired from `aggregate_stats_from_xtab()`
    """
    print(f'Total number of records: {stats_dict["n_records"]}')
    print(f'Total exact matches: {stats_dict["n_correct"]}')
    print(f'Percent exact: {stats_dict["absolute_accuracy"]*100:.1f}%')
    print(f'Percent one-half 1 off: {stats_dict["one_half_accuracy"]*100:.1f}%')
    print(f'Percent Gaussian rolloff: {stats_dict["gaussian_accuracy"]*100:.1f}%')


def evaluate_model(df: pd.DataFrame, model, labeled_col: str='match_score') -> Tuple[pd.DataFrame, dict]:
    """Run a DataFrame through a model by calling its `predict()` method, placing the results
    of the prediction in a new column called "pred".

    Args:
        df (pd.DataFrame): Test DataFrame to compare with labeled data.
        model (model): Object with a `predict()` method that takes the DataFrame as input.
        labeled_col (str, optional): Name of the column with hand labels. Defaults to 'match_score'.

    Returns:
        tuple: A cross tabulated DataFrame of prediction labels against hand labels, and a dict with stats
            of the cross tabulation. It also updates the input DataFrame with a "pred" column and may
            update the state of the model if it is modified when calling `predict()`.
    """
    df['pred'] = model.predict(df)
    df_xtab = labeled_xtab(df, labeled_col=labeled_col)
    d_stats = aggregate_stats_from_xtab(df_xtab)
    print_aggregate_stats(d_stats)
    
    return df_xtab, d_stats

def profile_predict(df: pd.DataFrame, model) -> List[float]:
    """Utility function for iterating through each row of the DataFrame and calling the model on that row.

    Args:
        df (pd.DataFrame): DataFrame from which to iterate.
        model (model): Object with a `predict()` method that takes the row (cast as a DataFrame) as input.

    Returns:
        list: List of durations in nanoseconds.
    """
    times = []
    for i, row in df.iterrows():
        v = row.to_frame().T
        start = time.process_time()
        model.predict(v)
        times.append(time.process_time()-start)
    return times

def profile_transform(df: pd.DataFrame, model) -> List[float]:
    """Utility function for iterating through each row of the DataFrame and calling the model on that row.

    Args:
        df (pd.DataFrame): DataFrame from which to iterate.
        model (model): Object with a `transform()` method that takes the row (cast as a DataFrame) as input.

    Returns:
        list: List of durations in nanoseconds.
    """
    times = []
    for i, row in df.iterrows():
        v = row.to_frame().T
        start = time.process_time()
        model.transform(v)
        times.append(time.process_time()-start)
    return times

def print_timing_performance(df: pd.DataFrame, model, profile_func=profile_predict):
    """Run `profile_predict()` and print the duration statistics.

    Args:
        df (pd.DataFrame): DataFrame from which to iterate.
        model (model): Object with a `predict()` method that takes the row (cast as a DataFrame) as input.
    """
    times = profile_func(df, model)
    s = f'Timing performance for individual requests:\n'
    s += f'mean: {np.mean(times)*1000:.2f} ms, '
    s += f'median: {np.median(times)*1000:.2f} ms, '
    s += f'min: {np.min(times)*1000:.2f} ms, '
    s += f'max: {np.max(times)*1000:.2f} ms'
    print(s)

def profile(df: pd.DataFrame, model, output_path: str, functions=[], modules=[], profile_func=profile_predict):
    """Execute `profile_func` through the line profiler (lprun) and write the results to a file.

    Args:
        df (pd.DataFrame): DataFrame from which to iterate.
        model (model): Object with a `predict()` method that takes the row (cast as a DataFrame) as input.
        output_path (str): Path to a .txt file. This should be in some data location that the model typically writes to.
        functions (list, optional): List of specific functions to put in scope. Either `modules` or this list 
            should not be empty for useful results. (Both can contain values.) Defaults to empty.
        modules (list, optional): List of imported modules to put in scope. Either `functions` or this list 
            should not be empty for useful results. (Both can contain values.) Defaults to empty.

    Example:
        The following example looks at the cleaning and stemming transformer and targets that module.

            from dhi.dsmatch.sklearnmodeling.models import cleanstemtransformer
            from dhi.dsmatch.analytics.modelevaluation import profile, profile_transform

            model = cleanstemtransformer.make_cleanstem_pipeline()
            df = <ACQUIRED SOME WAY>
            profile(df, model, './profile.txt', modules=[cleanstemtransformer], profile_func=profile_transform)

    
    Example 2:
        Similar to the above, but calling explicit functions instead of the module.
        
            from dhi.dsmatch.sklearnmodeling.models.cleanstemtransformer import make_cleanstem_pipeline
            from dhi.dsmatch.preprocess.clean import clean, stem
            from dhi.dsmatch.analytics.modelevaluation import profile, profile_transform

            model = make_cleanstem_pipeline()
            df = <ACQUIRED SOME WAY>
            profile(df, model, './profile.txt', functions=[clean, stem], profile_func=profile_transform)

    """
    # https://github.com/rkern/line_profiler/blob/master/line_profiler.py
    # From the commandline, this would be
    # timing_file = f'{model_name}_timing.txt'
    # %lprun -T $timing_file -m $m.__module__ -s profile_func(df, model)
    from line_profiler import LineProfiler
    from IPython.core.page import page

    lp = LineProfiler()
    for f in functions:
        lp.add_function(f)
    for module in modules:
        lp.add_module(module)
    lp_wrapper = lp(profile_func)
    lp_wrapper(df, model)

    # Trap text output.
    stdout_trap = StringIO()
    lp.print_stats(stdout_trap, stripzeros=True)
    output = stdout_trap.getvalue()
    output = output.rstrip()
    page(output)
    pfile = open(output_path, 'w')
    pfile.write(output)
    pfile.close()
