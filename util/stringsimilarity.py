Jupyter Notebook
stringsimilarity.py
06/07/2021
Python
File
Edit
View
Language
1
"""
2
Functions for performing edit distance and comparisons between strings.
3
"""
4
import sys
5
import os
6
import multiprocessing as mp
7
import concurrent.futures
8
from itertools import combinations, zip_longest, chain, product
9
import random
10
from math import ceil
11
from difflib import SequenceMatcher
12
​
13
import pandas as pd
14
import numpy as np
15
​
16
def levenshtein(seq1, seq2): 
17
    """Perform the Levenshtein distance metric between two sequences.
18
    The Levenshtein distance is the smallest number of edits to transform
19
    one string into another.
20
    
21
    Parameters
22
    ----------
23
    seq1 : str
24
        First string to compare
25
    seq2 : str
26
        Second string to compare
27
    
28
    Returns
29
    -------
30
    int
31
        The smallest number of edits to transform one string into another.
32
    """
