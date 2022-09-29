
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display import HTML


def plot_hist(data, title_text='', thresholds=[], resolution=101, xlim=None):
    """Plot a histogram with threshold demarcations if specified.

    Args:
        data (Numeric array): Array of numeric values to get a histogram of.
        title_text (str, optional): String to provide as a title. Defaults to ''.
        thresholds (list, optional): List of demarcation values. If specified, then red lines will overlay
            the histogram at these locations. Defaults to [].
        resolution (int): Number of bins.
        xlim (tuple): limit of the horizontal axis.

    Returns:
        axes: Matplotlib.axes object.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.linspace(0, 1, resolution)
    ax.hist(data, bins=bins, density=True);
    if xlim:
        ax.set_xlim(xlim)
    for t in thresholds:
        ax.plot([t, t], [0, ax.get_ylim()[1]], color='red')
    ax.set_title(title_text)

    return ax

def pretty_value_counts(s: pd.Series, head_rows: int=None, show_rank: bool=True, 
                        vc_kwargs: dict={}, html_kwargs: dict={}) -> pd.DataFrame:
    """Perform a `pd.Series.value_counts()`, and display as a DataFrame.

    Note: This function largely only works in Jupyter notebooks.

    Args:
        s (pd.Series): Series to perform a `value_counts()` on.
        head_rows (int, optional): Number of top rows to display. Defaults to -1, which is everything.
        show_rank (bool, optional): If True, index is 1 to the number of head_rows. If False, then DataFrame is 0-based.
            Defaults to True.
        vc_kwargs (dict, optional): keyword arguments for `pd.Series.value_counts()`. Defaults to {}.
        html_kwargs (dict, optional): keyword arguments for `pd.DataFrame.to_html()`. Defaults to {}.

    Returns:
        pd.DataFrame: The full value_counts converted to a DataFrame, ignoring head_rows.
    """
    display(HTML(f'<h4>{s.name}</h4>'))
    if 'normalize' in vc_kwargs:
        freq_col = 'fraction'
    else:
        freq_col = 'count'
    df = s.value_counts(**vc_kwargs).reset_index().rename(columns={'index': 'value', s.name: freq_col})
    if show_rank:
        df.index += 1
        df.index.name = 'rank'
    if head_rows is None:
        display(HTML(df.to_html(**html_kwargs)))
    else:
        display(HTML(df.head(head_rows).to_html(**html_kwargs)))
    return df
