"""
Functions to get ngrams and ngramming models.         
"""
from itertools import chain

from gensim.models.phrases import Phrases, Phraser

def build_ngrams_model(words: list, n: int=3):
    """Use gensim's Phrases to make bigrams and trigrams.
    
    Args:
        words (list): list or list of lists of words. 
        n (int): 2 == bigrams, 3 = bigrams and trigrams
        
    Returns:
        Gensim Phraser object
    """
    assert(n == 2 or n == 3)
    phrases = Phrases(chain(*words))  # bigrams
    if n == 3:
        phrases = Phrases(phrases[chain(*words)])  # trigrams
    ngrams_model = Phraser(phrases)
    return ngrams_model

def apply_ngrams(sentences, ngrams_model):
    """
    Given a list of lists of sentences, concatenate bigrams into one word of 
    the format "first_end" and trigrams into "first_middle_end" format.
    
    Returns:
        List of lists where outer lists are sentences and inner lists are individual
        words or n-grams when reasonable.
    """
    ret = []
    for sent in sentences:
        ret.append(ngrams_model[sent])
    return ret
