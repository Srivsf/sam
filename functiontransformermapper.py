"""
Generic utility methods to be used in FunctionTransformers.

These methods try to execute arbitrary "transforming" functions on data and permit various responses
in the case of an error. If there is an error, the original object is returned.

Note: When possible, call FunctionTransformer with a given function and kwargs over these functions.
For example, `FunctionTransformer(np.median)` is much more succinct and efficient than wrapping with some
of these methods.
"""
import functools
import numpy as np
import pandas as pd

def applymap(df: pd.DataFrame, func, output_cols: list=None, concat_columns=False, **fkwargs) -> pd.DataFrame:
    """Performs [pd.applymap](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.applymap.html)
    but does so without executing the first row twice.

    Args:
        df (DataFrame): DataFrame to apply a function to.
        func (function): Function that is called on each element applied
        output_cols (list, optional): List of output columns. If None, then the existing column names are used. 
            Defaults to None.
        concat_columns (bool, optional): If True, then the outputs are concatenated as additional columns onto X.
            One should probably use `output_cols` in this case to avoid duplicate column names. Defaults to False.
            See also FilterApplyTransformer.
        fkwargs (dict, optional): Optional kwarg arguments to go to the applied function.

    Returns:
        DataFrame: DataFrame where each element in cols is applied.
    """
    if not isinstance(df, pd.DataFrame):
        if output_cols is not None and len(output_cols) == df.shape[1]:
            df = pd.DataFrame(df, columns=output_cols)
        else:
            df = pd.DataFrame(df)
    recs = {row[0]: [func(x, **fkwargs) for x in row[1:]] 
            for row in df.itertuples()}
    if output_cols is None:
        output_cols = df.columns
    ret = pd.DataFrame.from_dict(recs, orient='index', columns=output_cols)
    if concat_columns:
        ret = pd.concat([df, ret], axis=1)
    return ret

def applyrows(df: pd.DataFrame, func, **fkwargs) -> pd.DataFrame:
    """Calls `df.apply(func, axis=1)`, but amenable to parallelization.

    Args:
        df (DataFrame): DataFrame to apply a function to.
        func (function): Function that is called on each row.
        fkwargs (dict, optional): Optional kwarg arguments to go to the applied function.

    Returns:
        DataFrame: DataFrame with the output from the applied function over each row.
    """
    return df.apply(func, **fkwargs, axis=1)
    
def np_apply(X, func, op_dtypes=None):
    """This iterator functions much like applymap. It uses Numpy's nditer to iterate
    over each item, and returns an object of the same shape. If using a Pandas DataFrame,
    `applymap()` is probably the better choice.

    For dtypes that are strings, set the corresponding `op_dtypes` to "object"

    Args:
        X (Array, DataFrame, Series, dict, list, sparse_matrix): Input to apply the function to.
        func (function): Function applied to each item in X.
        op_dtypes (list, optional): List of 2 items corresponding the input and output types. 
            If operating on arbitrary objects, set both to "object". Defaults to None.

    Returns:
        numpy.array: Array where elements are the result of running func on each input element.
    """
    with np.nditer([X, None], 
                   flags=['refs_ok'],
                   op_flags=[['readonly'], ['writeonly', 'allocate']],
                   op_dtypes=op_dtypes
                  ) as it:
        for x, y in it:
            y[...] = func(x.item())
        if op_dtypes[1] == 'object':
            return it.operands[1].astype('object')
        return it.operands[1]

def try_member_func(X, member_func='func', on_error='warn', fkwargs=None):
    """Try to execute a member function of X.

    Args:
        X (object): A Python object. In the case of FunctionTransformers, this should most
            likely be a list, dict, Numpy object, or Pandas DataFrame or Series.
        func (str): Name of the member function to execute.
        fkwargs (dict, optional): Optional kwarg arguments to go to the member function.
        on_error (str, optional): Behavior to execute on an error. Options are:

            * `warn` -- print the error and return X without transformation.
            * `ignore` -- return X without transformation.
            * `raise` -- raise the error.

            Defaults to 'warn'.

    Raises:
        err: AttributeError
        berr: BaseException -- any other exception.

    Returns:
        Transformation of X if possible, and original X if an exception occurred and we warn or ignore errors.

    Example:
        The following example shows how we can call a Pandas.DataFrame method and specify kwargs if necessary.
    
            import pandas as pd
            from sklearn.preprocessing import FunctionTransformer

            df = pd.DataFrame({'name': ['Anne', 'Bob', 'Charlie', 'Bob'],
                            'age': [20, 21, 22, 23]})
            print(df)
            tx = FunctionTransformer(try_member_func, kw_args=dict(func='drop_duplicates', fkwargs=dict(subset='name')))
            print()
            print('Transformed:')
            print(tx.transform(df))

        Outputs:

                name  age
            0     Anne   20
            1      Bob   21
            2  Charlie   22
            3      Bob   23

            Transformed:
                name  age
            0     Anne   20
            1      Bob   21
            2  Charlie   22
    """
    try:
        f = getattr(X, member_func)
        return f(**fkwargs)
    except AttributeError as err:
        if on_error == 'ignore':
            pass
        elif on_error == 'warn':
            print(f'"{member_func}" does not exist as a member function of {type(X)}.')
        else:
            raise err
    except BaseException as berr:
        if on_error == 'ignore':
            pass
        elif on_error == 'warn':
            print('Warning:', berr)
        else:
            raise berr
    return X

def try_vectorize(X, func, on_error='warn', **fkwargs):
    """Try to call `np.vectorize(func)(X)`.

    Args:
        X (object): A Python object. In the case of FunctionTransformers, this should most
            likely be a list, dict, Numpy object, or Pandas DataFrame or Series.
        func (function): function to execute.
        fkwargs (dict, optional): Optional kwarg arguments for np.vectorize, such as `signature`.
        on_error (str, optional): Behavior to execute on an error. Options are:

            * `warn` -- print the error and return X without transformation.
            * `ignore` -- return X without transformation.
            * `raise` -- raise the error.
            
            Defaults to 'warn'.

    Note: This appears to call the function twice on the first element, much like `Pandas.applymap()`.
        As such, it may not be amenable to transformations where single elements are submitted.

    Raises:
        err: AttributeError
        berr: BaseException -- any other exception.

    Returns:
        Transformation of X if possible, and original X if an exception occurred and we warn or ignore errors.

    Example:
        The following example transforms ascii values to their character representations.
    
            import numpy as np

            X = np.arange(ord('a'), ord('a')+3*4).reshape(3,4)
            print(X)

            tx = FunctionTransformer(try_vectorize, kw_args=dict(func=chr))
            print()
            print('Transformed:')
            print(tx.transform(X))

        Outputs:

            [[ 97  98  99 100]
            [101 102 103 104]
            [105 106 107 108]]

            Transformed:
            [['a' 'b' 'c' 'd']
            ['e' 'f' 'g' 'h']
            ['i' 'j' 'k' 'l']]

    """
    try:
        return np.vectorize(func)(X)
    except BaseException as err:
        if on_error == 'ignore':
            pass
        elif on_error == 'warn':
            print('Warning:', err)
        else:
            raise err
    return X

def try_map(X, func, on_error='warn', **fkwargs):
    """Try to call `list(map(func, X))`.

    Args:
        X (object): A Python object. In the case of FunctionTransformers, this should most
            likely be a list, dict, Numpy object, or Pandas DataFrame or Series.
        func (function): function to execute.
        fkwargs (dict, optional): Optional kwarg arguments to go to the function.
        on_error (str, optional): Behavior to execute on an error. Options are:

            * `warn` -- print the error and return X without transformation.
            * `ignore` -- return X without transformation.
            * `raise` -- raise the error.
            
            Defaults to 'warn'.

    Raises:
        err: AttributeError
        berr: BaseException -- any other exception.

    Returns:
        Transformation of X if possible, and original X if an exception occurred and we warn or ignore errors.

    Example:
        The following example shows a simple iterable that allows for custom functions that take kwargs.
    
            import numpy as np

            def add_y(x, y):
                return x + y

            X = np.arange(3*4).reshape(3,4)
            print(X)

            tx = FunctionTransformer(try_map, kw_args=dict(func=add_y, fkwargs=dict(y=3)))
            print()
            print('Transformed:')
            print(tx.transform(X))

        Outputs:

            [[ 0  1  2  3]
            [ 4  5  6  7]
            [ 8  9 10 11]]

            Transformed:
            [array([3, 4, 5, 6]), array([ 7,  8,  9, 10]), array([11, 12, 13, 14])]

    """
    try:
        return list(map(functools.partial(func, **fkwargs), X))
    except BaseException as err:
        if on_error == 'ignore':
            pass
        elif on_error == 'warn':
            print('Warning:', err)
        else:
            raise err
    return X

def try_func(X, func, on_error='warn', **fkwargs):
    """Try to call `func(X, **kwargs)`.

    Args:
        X (object): A Python object. In the case of FunctionTransformers, this should most
            likely be a list, dict, Numpy object, or Pandas DataFrame or Series.
        func (function): function to execute.
        fkwargs (dict, optional): Optional kwarg arguments to go to the function.
        on_error (str, optional): Behavior to execute on an error. Options are:

            * `warn` -- print the error and return X without transformation.
            * `ignore` -- return X without transformation.
            * `raise` -- raise the error.

            Defaults to 'warn'.

    Raises:
        err: AttributeError
        berr: BaseException -- any other exception.

    Returns:
        Transformation of X if possible, and original X if an exception occurred and we warn or ignore errors.
    """
    try:
        return func(X, **fkwargs)
    except BaseException as err:
        if on_error == 'ignore':
            pass
        elif on_error == 'warn':
            print('Warning:', err)
        else:
            raise err
    return X
