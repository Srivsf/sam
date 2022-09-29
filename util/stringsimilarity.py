"""
Functions for performing edit distance and comparisons between strings.
"""
import sys
import os
import multiprocessing as mp
import concurrent.futures
from itertools import combinations, zip_longest, chain, product
import random
from math import ceil
from difflib import SequenceMatcher

import pandas as pd
import numpy as np

def levenshtein(seq1, seq2): 
    """Perform the Levenshtein distance metric between two sequences.
    The Levenshtein distance is the smallest number of edits to transform
    one string into another.
    
    Parameters
    ----------
    seq1 : str
        First string to compare
    seq2 : str
        Second string to compare
    
    Returns
    -------
    int
        The smallest number of edits to transform one string into another.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    val = matrix[size_x - 1, size_y - 1]
    return val

def levenshtein_factor(s1: str, s2: str):
    """Perform a Levenshtein distance between two strings, but then normalize
    by the length of the longest string.
    
    Parameters
    ----------
    s1 : str
    s2 : str
    
    Returns
    -------
    float
        A number between 0 and 1. 0 means no similarity. 1 means duplicate,
        and each string is of the same length. 
        0.5 means 50% of the characters in the longest string are present and in
        the correct order in the shorter string.
    """
    max_len = max([len(s1), len(s2)])
    return (max_len-levenshtein(s1, s2))/max_len

def levenshtein_factor_mp(kwargs):
    s1, s2 = kwargs
    return levenshtein_factor(s1, s2)

def all_pairs_levenshtein_factor(strs: list, max_pairs: int=0, n_processes: int=-1):
    """Perform Levenshtein Factor comparison across all pairs of strings.
    
    Note: This runs at about 5000 pairs per second on 8-processor MacBook Pro.
    Runtime is (len(strs)**2)/2 when `max_pairs` == 0.
    For example, 3M pairs is ~2500 strings and takes about 10 mins.

    Parameters
    ----------
    strs : list
        List of strings to perform pairwise comparisons on.
    max_pairs : int, optional
        Total number of pairwise comparisons, by default 0, which is all pairs.
        Note: Number of unique strings is ~= sqrt(max_pairs) * 2
    n_processes : int, optional
        Number of processors to use. If -1, it will use all available, by default -1.

    Returns
    -------
        DataFrame
            A nxn DataFrame with the pairwise levenshtein factors.
            Column/Rows are the strings.
    """
    results = []
    if n_processes == -1:
        n_processes = mp.cpu_count()
    
    if max_pairs > 0:
        n_samples = int(np.ceil(np.sqrt(max_pairs*2))) - 1
        strs = random.sample(strs, k=n_samples)
    
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map_async(levenshtein_factor_mp, combinations(strs, 2)).get()
    
    # We create a square matrix
    a = np.zeros((len(strs), len(strs)))
    iu = np.triu_indices(len(strs), k=1)   # Upper triangular
    il = np.tril_indices(len(strs), k=-1)  # Lower triangular
    a[iu] = results
    a[il] = a.T[il]
    
    df_sim = pd.DataFrame(a, index=strs, columns=strs)

    return df_sim

def longest_substr(s1: str, s2: str, with_size: bool=False):
    """Get the longest common substring between string a and string b.
    
    Parameters
    ----------
    a : str
        First string
    b : str
        Second string
    with_size : bool, optional
        If true, return a tuple(longest_substr, len(longest_substr)).
        Otherwise return the longest substring. By default False
    
    Returns
    -------
    str
        The longest common substring between strings a and b. If with_size=True,
        return a tuple(longest_substr, len(longest_substr)).
    """
    # initialize SequenceMatcher object with  
    # input string 
    seqMatch = SequenceMatcher(None, s1, s2) 

    # find match of longest sub-string 
    # output will be like Match(a=0, b=0, size=5) 
    match = seqMatch.find_longest_match(0, len(s1), 0, len(s2)) 

    # print longest substring 
    return s1[match.a: match.a + match.size], match.size

def longest_substr_factor(s1: str, s2: str):
    """Perform a longest common substring between two strings, but then normalize
    by the length of the shortest string.
    
    Parameters
    ----------
    s1 : str
    s2 : str
    
    Returns
    -------
    float
        A number between 0 and 1. 0 means no similarity. 1 means one string is completely
        encapsulated in the other. 
        0.5 means 50% of the characters in the shortest string are a common substring of both strings.
    """
    min_len = min([len(s1), len(s2)])
    return longest_substr(s1, s2)[1]/min_len

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def longest_substr_mp_chunk(chunk):
    data = []
    keys = []
    try:
        for (s1, s2) in zip(chunk):
            data.append(longest_substr(s1, s2, True))
            keys.append((s1, s2))
    except TypeError:  # Some values might be none. Just ignore.
        pass

    df = pd.DataFrame(keys, columns=['a', 'b'])
    df = pd.concat([df, pd.DataFrame(data, columns=['substr', 'substr_len'])], axis=1)

    return df

def all_pairs_longest_substr(strs, max_pairs: int=0, n_processes: int=-1):
    """Perform an all-pairs longest common substring across all pairs of strings.
    
    Note: This runs at about 135000 pairs per second on 8-processor MacBook Pro.
    Runtime is (len(strs)**2)/2 when `max_pairs` == 0.
    For example, 10M pairs is ~4472 strings and takes about 1:15 mins.

    Parameters
    ----------
    strs : list
        List of strings to perform pairwise comparisons on.
    max_pairs : int, optional
        Total number of pairwise comparisons, by default 0, which is all pairs.
        Note: Number of unique strings is ~= sqrt(max_pairs) * 2
    n_processes : int, optional
        Number of processors to use. If -1, it will use all available, by default -1.

    Returns
    -------
        DataFrame
            A DataFrame of up to max_pairs rows with the longest common substring between pairs.
    """
    results = []
    if n_processes == -1:
        n_processes = mp.cpu_count()

    if max_pairs > 0:
        n_samples = int(np.ceil(np.sqrt(max_pairs*2))) - 1
        try:
            strs = random.sample(strs, k=n_samples)
        except ValueError:  # Assume that population is already larger and proceed.
            pass

    n_combos = int(len(strs)*(len(strs)-1)/ 2)
    n_elems_per_group = ceil(n_combos/n_processes)

    with concurrent.futures.ProcessPoolExecutor() as pool:
        results = list(pool.map(longest_substr_mp_chunk, zip(
             grouper(combinations(strs, 2), n_elems_per_group))))
    # with mp.Pool(processes=n_processes) as pool:
    #     results = pool.map(longest_substr_mp_chunk, zip(
    #         grouper(combinations(strs, 2), n_elems_per_group)))
    
    return pd.concat(results)

def all_to_all_longest_substr_pairs(strs_a: list, strs_b: list, max_pairs: int=0, n_processes: int=-1):
    """Perform an all-pairs longest common substring across all pairs of strings.
    
    Note: This runs at about 135000 pairs per second on 8-processor MacBook Pro.
    Runtime is (len(strs)**2)/2 when `max_pairs` == 0.
    For example, 10M pairs is ~4472 strings and takes about 1:15 mins.

    Parameters
    ----------
    strs_a : list
        List of strings to perform pairwise comparisons with `strs_b`.
    strs_b : list
        List of strings to perform pairwise comparisons with `strs_a`.
    max_pairs : int, optional
        Total number of pairwise comparisons, by default 0, 
        which is all elements of `strs_a` with all elements of `strs_b`.
        Note: Number of unique strings is ~= sqrt(max_pairs) * 2
    n_processes : int, optional
        Number of processors to use. If -1, it will use all available, by default -1.

    Returns
    -------
        DataFrame
            A DataFrame of up to max_pairs rows with the longest common substring between pairs.
    """
    results = []
    if n_processes == -1:
        n_processes = mp.cpu_count()

    n_combos = len(strs_a) * len(strs_b)

    if max_pairs > 0 and max_pairs < n_combos:
        n_samples = int(np.floor(np.sqrt(max_pairs)))
        try:
            strs_a = random.sample(strs_a, k=n_samples)
            strs_b = random.sample(strs_b, k=n_samples)
        except ValueError:  # Assume that population is already larger and proceed.
            pass

    n_combos = len(strs_a) * len(strs_b)
    n_elems_per_group = ceil(n_combos / n_processes)

    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(longest_substr_mp_chunk, zip(
            grouper(product(strs_a, strs_b), n_elems_per_group)))

    return pd.concat(results)

def top_n_substrings(df: pd.DataFrame, min_substr_len: int=1, top_n=100):
    """Get the top *n* most common substrings from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have a `substr_len` column and `substr` column as can be made
        from running `all_to_all_longest_substr_pairs()`.
    min_substr_len : int, optional
        Filter for the minimal substring length to allow, by default 1, which is all strings.
    top_n : int, optional
        Top *n* most repeated substrings, by default 100.
    
    Returns
    -------
    pd.DataFrame
        Sorted list of most commonly-repeated substrings from the DataFrame passed in.
    """
    df_ = df[df.substr_len >= min_substr_len]
    df_ = df_.substr.value_counts().head(top_n)
    df_ = df_.reset_index().rename(columns={'substr': 'substr_freq', 'index': 'substr'})
    
    return df_
