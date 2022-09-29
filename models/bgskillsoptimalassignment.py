Jupyter Notebook
bgskillsoptimalassignment.py
06/07/2021
Python
File
Edit
View
Language
1
from itertools import chain
2
​
3
from tqdm.auto import tqdm
4
import pandas as pd
5
import numpy as np
6
from scipy.optimize import linear_sum_assignment
7
from sklearn.metrics.pairwise import cosine_similarity
8
from sklearn.preprocessing import QuantileTransformer
9
​
10
from dhi.dsmatch.sklearnmodeling.models.quantilehelpers import QuantileConfidenceTransformer, QuantilePredictMixin
11
from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer
12
from dhi.dsmatch.sklearnmodeling.functiontransformermapper import try_func
13
​
14
class BGSkills(QuantileConfidenceTransformer):
15
    _version = '1.0.0'
16
    name = 'efc_bgskills_pairwise_optimal_assignment'
17
​
18
    @staticmethod
19
    def lowercase_list(x):
20
        try:
21
            return list(map(str.lower, x))
22
        except:
23
            pass
24
        return []
25
            
26
    @staticmethod
27
    def optimal_assignment(jd_skills, r_skills, df_lookup, maximize=True, return_artifacts=False, thresh=None):
28
        """Given a list of job description skills, a list of resume skills, and a lookup table
29
        of the match of skills to each other, run the optimal assignment algorithm to obtain
30
        an optimal mapping and return the mean of those edge weights.
31
​
