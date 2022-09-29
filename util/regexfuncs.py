"""
Various Regex functions.

Most functions follow a particular pattern for find/replace:

    def query(astr: str):
        '''Find and replace pattern if it exists.'''

        pat = re.compile('''
        (?P<group1>A)          # Multilines 
        (?P<group2>MULTILINE)  # permits comments
        (?P<group3>REGEX)      # to make pattern
        (?P<group4>FORMULA)    # formulas clearer.
        ''', re.VERBOSE)

        def match_func(match_obj):
            '''Internal function that only gets called there is a match.
            If we used named groups in the pattern, we can reference them in the groupdict.'''
            d = match_obj.groupdict()
            return do_something_with_dict_and_return_string(d)

        return pat.sub(match_func, astr)
            
"""

import re

def periods_before_li_tags(astr: str):
    """
    Make each `<li>` tag end with a period if they do not already have a period so that they operate as sentences.
    """
    pat = re.compile(r"""
    (\.)?  # Swallow period if it already exists
    \s*    # optional whitespace
    </li>  # li closing tag
    """, re.VERBOSE)
    
    return pat.sub('.</li>', astr)

def concat_bounding_ampersand(astr: str):
    """
    Convert every "A & B" to "A_and_B".
    """
    
    pat = re.compile(r"""
    (?P<pre>\w+)  # contiguous letters
    \s*           # optional whitespace
    &(amp;)?      # ampersand char possibly with "amp;", too
    \s*           # optional whitespace
    (?P<post>\w+) # contiguous letters
    """, re.VERBOSE)
    
    def match_func(match_obj):
        d = match_obj.groupdict()
        return d['pre'] + '_and_' + d['post']
        
    return pat.sub(match_func, astr)

def replace_IT(astr: str):
    """
    Make each "IT" string "information_technology".
    """
    
    pat = re.compile(r"""
        (?P<pre>\w)?  # optional contiguous letters
        I\.*T\.*
        (?P<post>\w)? # optional contiguous letters
        """, re.VERBOSE)

    def match_func(match_obj):
        d = match_obj.groupdict()
        if d['pre'] is None and d['post'] is None:  # If no pre and post, then IT stands as a word.
            return 'information_technology'
        retstr = ''
        if d['pre']:
            retstr += d['pre']
        retstr += 'IT'
        if d['post']:
            retstr += d['post']
        return retstr

    return pat.sub(match_func, astr)

def concat_bounding_slash(astr):
    """
    Make each "a / b" string "a_slash_b".
    """
    pat = re.compile(r"""
        (?P<pre>\w+)  # contiguous letters
        \s*           # optional whitespace
        /             # slash char
        \s*           # optional whitespace
        (?P<post>\w+) # contiguous letters
        """, re.VERBOSE)

    def match_func(match_obj):
        d = match_obj.groupdict()
        return d['pre'] + '_slash_' + d['post']
    
    while True:
        newstr = pat.sub(match_func, astr)
        if newstr == astr:
            break
        astr = newstr
    return astr

def replace_dot(astr):
    """
    Make each "[a].b" string "a_dot_b".
    """
    pat = re.compile(r"""
        (?P<pre>\w+)?  # contiguous letters
        \.             # dot char
        (?P<post>\w+)  # contiguous letters
        """, re.VERBOSE)
    
    def match_func(match_obj):
        d = match_obj.groupdict()
        retstr = ''
        if d['pre']:
            retstr += d['pre'] + '_'
        retstr += 'dot_' + d['post']
        return retstr

    return pat.sub(match_func, astr)

def replace_uri(astr):
    """
    Make each "[a].b" string "a_dot_b".
    """
    pat = re.compile(r"""
        (?P<mode>\w+://)?  # contiguous letters
        (?P<pre>\w+)?  # contiguous letters
        \.             # dot char
        (?P<post>\w+(\.)?)  # contiguous letters
        """, re.VERBOSE)

    def match_func(match_obj):
        d = match_obj.groupdict()
        retstr = ''
    #     if d['mode']:
    #         retstr += d['mode'] + '_'
        if d['pre']:
            retstr += d['pre'] + '_'
        retstr += 'dot_' + d['post']
        return retstr

    while True:
        newstr = pat.sub(match_func, astr)
        if newstr == astr:
            break
        astr = newstr
    return astr

def replace_at(astr):
    """
    Make each "[a]@b" string "a_at_b".
    """
    pat = re.compile(r"""
        (?P<mode>\w+://)?   # Optional header xxx:// 
        (?P<pre>\w+)?       # contiguous letters
        @                   # at char
        (?P<post>\w+(\.)?)  # contiguous letters
    """, re.VERBOSE)
    
    def match_func(match_obj):
        d = match_obj.groupdict()
        retstr = ''
        if d['pre']:
            retstr += d['pre'] + '_'
        retstr += 'at_' + d['post']
        return retstr

    while True:
        newstr = pat.sub(match_func, astr)
        if newstr == astr:
            break
        astr = newstr
    return astr

def all_caps_semi(astr: str):
    """
    Convert every "ABC:" to ". ABC:".
    """
    
    pat = re.compile(r"""
    (?P<caps>[A-Z\s]+:)  # CAPS WITH POSSIBLE SPACES:
    """, re.VERBOSE)
    
    def match_func(match_obj):
        d = match_obj.groupdict()
        return '. ' + d['caps']
        
    return pat.sub(match_func, astr)

def left_paren_then_not_lower(astr: str):
    """
    Convert every ")Camel" to "). Camel". Or any mark ") - my str" to "). - my str"
    """
    
    pat = re.compile(r"""
    (?P<paren>\)\s*)
    (?P<c>[^a-z])  # Any
    """, re.VERBOSE)
    
    def match_func(match_obj):
        d = match_obj.groupdict()
        return '). ' + d['c']
        
    return pat.sub(match_func, astr)

def return_camel(astr: str):
    """
    Convert every "\nCamel" to ". Camel".
    """
    
    pat = re.compile(r"""
    (?P<ret>\n)
    (?P<c>[A-Z])  # Capital Letter
    """, re.VERBOSE)
    
    def match_func(match_obj):
        d = match_obj.groupdict()
        return '. ' + d['c']
        
    return pat.sub(match_func, astr)

def html_items_to_sentences(astr: str):
    """
    Make sure every "<li blah BLAH blah /li>" ends as a sentence "<li blah BLAH blah.  /li>"
    """
    pat = re.compile(r"""
    (?P<right><\s*/(li|ul|LI|UL)\s*>)  # "/li>"
    """, re.VERBOSE)
    
    def match_func(match_obj):
        d = match_obj.groupdict()
        return '. ' + d['right']
        
    return pat.sub(match_func, astr)

def divs_to_periods(astr: str):
    """
    Make sure every "</div>" is ". </div>"
    """
    pat = re.compile(r"""
    (?P<div><\s*/\s*div\s*>)  # "/li>"
    """, re.VERBOSE)
    
    def match_func(match_obj):
        d = match_obj.groupdict()
        return '. ' + d['div']
        
    return pat.sub(match_func, astr)
