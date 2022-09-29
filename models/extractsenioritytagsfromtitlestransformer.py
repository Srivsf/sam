import re
from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer
from dhi.dsmatch.sklearnmodeling.models.custombase import CustomTransformer
from dhi.dsmatch.sklearnmodeling.functiontransformermapper import applymap


class ExtractSeniorityTagsFromTitlesTransformer(CustomTransformer):   
    """
    The class extracts the seniority tags from the job title, we sent a DataFrame[col] and it returns columns of a list with one
    or multiple tags for eg: Sr Java Lead will return [senior, lead].
    Arg:
       d[col]: DataFrame with column name (for eg job_title)
    Returns:
       d[col]: DataFrame with same column name but values will be a list of string/strings 
    """

    pats = {}
    pats['senior'] = r'(?P<tag>\bsenior\b|\bsr\b|\bsnr\b)(?P<ignore>)'
    pats['junior'] = r'(?P<tag>(\bjunior\b|\bjr\b|\bjnr\b))'
    pats['intern'] = r'(?P<tag>\bintern(ship)?\b|\btrainee\b|\apprentice(ship)?)'
    pats['level'] = r"""
    (?P<tag>\b(level|lvl|tier)\b)
    (?P<intermediate>\s+)  # Any arbitrary string between "level" and number of ii...
    (?P<level>(\b[1-8]\b)|(one|two|three|iii|ii|iv|viii|vii|vi|v|i))
    """
    pats['l_number'] = r'(?P<tag>\bl)(?P<level>(i+|v|[0-9])\b)'
    pats['lev_number'] = r'(?P<level>\b(i+|v|iv|vi+)\b)'
    pats['assistant'] = r'(?P<tag>\bassistant\b(?!\W+vice\b))'
    pats['associate'] = r'(?P<tag>\bassoc.*\b)'
    pats['entry_level'] = r'(?P<tag>\bentry.*level\b|\bbeginner\b|\bfresh.*\b|\bcollege grad.*\b)'
    pats['master'] = r'(?P<tag>(?<!scrum\s)(?<!web\s)master(?![data management]))'
    pats['principal'] = r'(?P<tag>\bprincipal\b)'
    pats['head'] = r'(?P<tag>\bhead\b)'
    pats['fellow'] = r'(?P<tag>\bfellow\b)'
    pats['advanced'] = r'(?P<tag>\badvanced\b)'
    pats['post_doctoral'] = r'(?P<tag>\bpost.*doctoral\b)'
    pats['president'] = r'(?P<tag>\b(?<!vice\s)president\b)'
    pats['professional'] = r'(?P<tag>\bprofessional\b)'
    pats['expert'] = r'(?P<tag>\bexpert\b)'
    pats['midlevel'] = r'(?P<tag>(mid level|\bmid\b))'
    pats['chief'] = r'(?P<tag>\bchief\b)'
    pats['officer'] = r'(?P<tag>\bofficer\b)'
    pats['certified'] = r'(?P<tag>\bcertif.*\b)'
    pats['graduate'] = r'(?P<tag>\bgraduate\b)'
    pats['degree'] = r'(?P<tag>\bgraduate\b)'
    pats['staff'] = r'(?P<tag>\bstaff\b)'

    def _map_level(self, astr):
        astr = astr.lower()
        try:
            return int(astr)
        except ValueError:
            pass
        if astr == 'i':
            return 1
        if astr == 'ii':
            return 2
        if astr == 'iii':
            return 3
        if astr == 'iv':
            return 4
        if astr == 'v':
            return 5
        if astr == 'vi':
            return 6
        if astr == 'vii':
            return 7
        if astr == 'one':
            return 1
        if astr == 'two':
            return 2
        if astr == 'three':
            return 3
        return 0
    
    def senior_patterns(self, astr):
        ret = []
        for k, v in self.pats.items():
            for m in re.finditer(v, astr, re.VERBOSE|re.IGNORECASE):
                if not m:
                    continue
                d = m.groupdict()
                if k in ['level', 'l_number', 'lev_number']:
                    d['level'] = self._map_level(d['level'])
                    ret.append({'tag': f"level {d['level']}"})
                else:
                    if 'tag' in d:
                        ret.append({'tag': k})

                    else:
                        ret.append(k, d)
        return ret


    def extracted_values(self, x):
        tags=[]
        for d_ in x:

            for item in d_.values():
                tags.append(item)
        tags = list(set(tags)) 
        return tags

                                
    def transform(self, X, **kwargs):
        # seniority extraction
        tx = ApplyTransformer(applymap, self.senior_patterns)
        X =  tx.transform(X)
        tx = ApplyTransformer(applymap, self.extracted_values)
        X =  tx.transform(X)
        return X
                                
