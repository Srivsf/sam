"""
Functions to clean and stem text.         
"""

import string
import unicodedata

import contractions
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

from bs4 import BeautifulSoup

import pandas as pd
from tqdm.auto import tqdm

from dhi.dsmatch.util.regexfuncs import *

stop_words = set(stopwords.words('english'))

punctuation_table = str.maketrans({key: None for key in string.punctuation if key != '_'})  # This removes all punctuation
punctuation_table[ord('–')] = None  # Add the hyphen
sno = SnowballStemmer('english')

# tqdm.pandas(mininterval=2)  # Update status bars every 2 seconds

def clean(raw_text: str, for_stemming: bool=False, **kwargs) -> str:
    """Clean text through a variety of regex functions. See comments for transformations.

    This function is geared for job descriptions and resumes that are in a text format already,
    perhaps HTML, but processed more for analytic rather than display purposes. For example,
    all items in a bulleted list are simply converted into sentences.

    Args:
        raw_text (str): Text to be transformed.
        for_stemming (bool): When True, then convert some expressions into formats conducive
            to stemming. For example, "x@y" becomes "x_at_y" and "www.abc.com" becomes
            "www_dot_abc_dot_com". When False, then these transformations do not take place.

    Returns:
        str: Transformed text.
    """
    try:
        t = contractions.fix(raw_text)  # "Don't" -> "do not"
    except IndexError:
        t = raw_text
    
    t = re.sub(r'</?p>', '. ', t)  # Paragraphs made into sentences.
    t = re.sub(f'<br\s*/>', '\r\n', t)  # HTML line breaks into regular line breaks
    t = t.replace('[E|e].g.', 'for example')
    t = t.replace('[I|i].e.', 'that is')
    t = all_caps_semi(t)  # Convert every "ABC:" to ". ABC:".
    t = html_items_to_sentences(t)
    t = divs_to_periods(t)

    if for_stemming:
        t = periods_before_li_tags(t)   # Line items get made into sentences
        t = replace_at(t)               # "x@y" -> "x_at_y"
        t = replace_dot(t)              # ".net" -> "dot_net", "www.abc.com" -> "www_dot_abc_dot_com"
        t = concat_bounding_ampersand(t) # "C & H" -> "C_and_H"
        t = concat_bounding_slash(t)    # "A / B" -> "A_slash_B"
        t = replace_IT(t)               # "IT" -> "information_technology"
        t = t.replace('++', 'plusplus')
        t = re.sub(r'C\s*#', 'C_sharp', t)

    soup = BeautifulSoup(t, "html.parser")  # Next 2 lines removes xml/html tags
    t = soup.get_text()             
    t = unicodedata.normalize("NFKD", t)  # Get rid of \xa0 -- extra whitespace
    # For unicode chars, see https://www.compart.com/en/unicode
    t = re.sub(r'[\u00A7\u00B0\u00B6\u00B7\u2022\u2023\u25A0-\u25FF\uE000-\uF8FF\u2043\u2219]\s*', '. ', t)  # Bullets and geometric chars
    t = re.sub(r'[\uFEFF\u200b]\s', '', t)  # Non-breaking space
    t = left_paren_then_not_lower(t)
    t = return_camel(t)
    t = re.sub(r'\s+', ' ', t)  # Multiple spaces become 1 space only.
    t = re.sub(r'(.)\1{2,}', '. ', t)  # 3 or more repeated characters are converted into a ". "
    t = re.sub(r'\r\n\s*', '. ', t)  # Line breaks are made into sentences.
    t = re.sub(r'\s*\.', '.', t)     # "    ." becomes "."
    t = re.sub(r'\s\d+(.|\))\s+', ' ', t)  # Ignore " 1. ", " 2. ", etc.  " 1) ", " 2) ", too.
    t = re.sub(r'\.+', '.', t)  # Repeated periods, such as ...., becomes .s

    return t.strip()

def clean_for_stemming(raw_text: str, for_stemming: bool=False, **kwargs) -> str:
    return clean(raw_text=raw_text, for_stemming=True, **kwargs)

def stem(raw_text, **kwargs):
    """Apply stemming to the raw (or previously cleaned) text.

    Args:
        raw_text (str): Document-level text.
        
    Returns:
        Sentences that have been tokenized.
    """
    sentences = sent_tokenize(raw_text)
    sentences = [word_tokenize(s) for s in sentences]
    new_sentences = ''
    for sent in sentences:
        stemmed = [sno.stem(word.lower()) for word in sent if word.lower() not in stop_words]    
        no_punctuation = [s.translate(punctuation_table) for s in stemmed]
        no_punctuation = [s for s in no_punctuation if s != '']
        new_sentences += ' '.join(no_punctuation) + '. '
    return new_sentences

def clean_and_stem(raw_text, **kwargs):
    """Apply the clean() function and then the stem() function and return the result.

    Args:
        raw_text (str): Raw text that is to be cleaned and stemmed.
    
    Returns:
        List of lists with sentences as the outer lists and pruned, 
        and stemmed words as the inner list.
        
        May return None if filtered to nothing.
   """
    try:
        return stem(clean(raw_text))
    except:
        return None

def df_apply_clean_and_stem(df, col):
    """
    Apply stemming to the column of interest, adding a `<col>_stemmed` column to the dataframe.
    
    Args:
        df (DataFrame): DataFrame to modify.
        col (str): Column name where each row is a string document that will be cleaned and stemmed.
    """
    tqdm.pandas(mininterval=2, desc=f'Stemming {col}')  # Update status bars every 2 seconds
    df[col + '_stemmed'] = df[col].progress_apply(clean_and_stem)
    tqdm.pandas(mininterval=.1, desc='')  # Reset TQDM.


