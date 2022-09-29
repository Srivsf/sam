import pandas as pd
from sklearn.pipeline import FeatureUnion

from dhi.dsmatch.sklearnmodeling.models.mixins import DataFrameMixin

class FeatureUnionDataFrame(DataFrameMixin, FeatureUnion):
    """This is a normal FeatureUnion, but when outputs of all internal transformers are DataFrames,
    this concatenates them into DataFrames.
    Model outputs provide column names as:
    
    ```
     <transformer_name_1>      |     <transformer_name_n>
    <column_1>, <column_2>, ...|... <column_1>, <column_2>
    ```

    Example:

            df = pd.DataFrame({'first_name': ['Anne', 'Bob', 'Charlie', 'Bob'],
                            'last_name': ['Bancroft', 'Dylan', 'Chaplain', 'Marley']})
            print(df)
            lowercase_tx = ApplyTransformer(applymap, str.lower)
            uppercase_tx = ApplyTransformer(applymap, str.upper)
            fu_tx = FeatureUnionDataFrame([('lower', lowercase_tx), ('upper', uppercase_tx)])
            Xt = fu_tx.transform(df)
            print(Xt)

        This provides the following output:

            first_name last_name
            0       Anne  Bancroft
            1        Bob     Dylan
            2    Charlie  Chaplain
            3        Bob    Marley

                lower                upper          
            first_name last_name first_name last_name
            0       anne  bancroft       ANNE  BANCROFT
            1        bob     dylan        BOB     DYLAN
            2    charlie  chaplain    CHARLIE  CHAPLAIN
            3        bob    marley        BOB    MARLEY

    """
    _version = '1.0.0'

    ##########
    # THE CODE BELOW WORKS WHEN WE DEFINE THE OBJECT AS
    # class FeatureUnionDataFrame(FeatureUnion): ...
    ##########
    # def _hstack(self, Xs):
    #     """Given a list of DataFrames (probably following `transform()` or `fit_transform()`),
    #     keep their columns and ascribe the model names as a hierarchical column name.

    #     Args:
    #         Xs (list): List of DataFrames (probably following `transform()` or `fit_transform()`).

    #     Returns:
    #         pd.DataFrame: single, concatenated DataFrame. If an error, it will return a numpy array.
    #     """
    #     try:
    #         if all(map(lambda x: isinstance(x, pd.DataFrame), Xs)):
    #             # When we concatenate Pandas DataFrames, they must all have the same depth of
    #             # multi-indexed columns. This loop largely adjusts for that, padding `''`` in the
    #             # top levels of the hierarchy if it needs to.
    #             valid_names = [n for (n, t) in self.transformer_list if t != 'drop']
    #             max_depth = 0
    #             for df_ in Xs:
    #                 try:
    #                     depth = len(df_.columns.levels)
    #                 except AttributeError:
    #                     depth = 1
    #                 if depth > max_depth:
    #                     max_depth = depth
    #             max_depth += 1
    #             for n, df_ in zip(valid_names, Xs):
    #                 new_cols = []
    #                 for c in df_.columns:
    #                     if isinstance(c, tuple):
    #                         new_c = [n, *c]
    #                     else:
    #                         new_c = [n, c]
    #                     new_cols.append(tuple([''] * (max_depth - len(new_c)) + new_c))
    #                 df_.columns = pd.MultiIndex.from_tuples(new_cols)
    #             return pd.concat(Xs, axis=1)
    #     except:
    #         pass
    #     return super()._hstack(Xs)
