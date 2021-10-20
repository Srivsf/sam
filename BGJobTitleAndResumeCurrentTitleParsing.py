import pandas as pd
import json
import re
import itertools
from itertools import combinations
from jsonpath_ng import parse
from sklearn.pipeline import FeatureUnion
from jsonpath_ng import parse
from dhi.dsmatch.sklearnmodeling.models.featureuniondataframe import  FeatureUnionDataFrame
from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer
from dhi.dsmatch.sklearnmodeling.models.quantilehelpers import QuantileConfidenceTransformer
from dhi.dsmatch.sklearnmodeling.functiontransformermapper import applymap,try_func
from dhi.dsmatch.sklearnmodeling.functiontransformermapper import applymap,try_func
class BGJobTitleAndResumeCurrentTitleParsing(QuantileConfidenceTransformer):
    def extract_title_jd(self,d: dict,canon) -> list:
        """Extract the canonical skill names from a Burning Glass parsed dict.

        Args:
            d (dict): Burning Glass LENS structure, parsed via json.loads if originally a string.

        Returns:
            list: List of canonical skill names found.
        """

        jsonpath_expr = parse(f'$..{canon}[*]')
        d =[match.value for match in jsonpath_expr.find(d)]
        if len(d)>0:
            d = d[0]
        return d
    def job_extract(self, dd):
        df = dd[['description_bg_parse']]
        df['description_bg_parse'] =df['description_bg_parse'].apply(json.loads)
        bgt_occ_tx = ApplyTransformer(applymap, self.extract_title_jd, fkwargs=dict(canon= 'bgt_occ'))
        clean_job_title_tx = ApplyTransformer(applymap, self.extract_title_jd, fkwargs=dict(canon= 'clean_job_title'))
        standard_title_tx = ApplyTransformer(applymap, self.extract_title_jd, fkwargs=dict(canon= 'standard_title'))
        consolidated_title_tx =ApplyTransformer(applymap, self.extract_title_jd, fkwargs=dict(canon= 'consolidated_title'))
        fu_tx = FeatureUnionDataFrame([('bgt_occ', bgt_occ_tx), ('clean_job_title', clean_job_title_tx),
                                        ('standard_title', standard_title_tx), ('consolidated_title', consolidated_title_tx)
                                       ])       
        Xt = fu_tx.transform(df)
        #Xt['bgt_occ'] = Xt['bgt_occ'].str.replace('NA',np.nan)
        Xt.columns = Xt.columns.str.replace('__description_bg_parse','')
        Xt.columns = ['bg_j_'+x for x in Xt.columns]
        
        for c in [x for x in Xt.columns]:
            Xt[c] = Xt[c].where(Xt[c].str.len() > 0, '')
        Xt['bg_j_bgt_occ'] = Xt['bg_j_bgt_occ'].str.replace('NA','')
        return Xt
    def extract_title_resume(self,d: dict,canon):
        bg_clean = parse(f'$..{canon}[*]')
        c =[match.value for match in bg_clean.find(d)]
        if len(c)>0:
            c = c[0]
        return c
    def resume_extract(self,dd):
        df = dd[['resume_bg_parse']]
        df['resume_bg_parse'] =df['resume_bg_parse'].apply(json.loads)
        bgt_occ_tx = ApplyTransformer(applymap, self.extract_title_resume, fkwargs=dict(canon= '@bgtocc'))
        clean_job_title_tx = ApplyTransformer(applymap, self.extract_title_resume, fkwargs=dict(canon= '@clean'))
        standard_title_tx = ApplyTransformer(applymap, self.extract_title_resume, fkwargs=dict(canon= '@std'))
        consolidated_title_tx =ApplyTransformer(applymap, self.extract_title_resume, fkwargs=dict(canon= '@consolidated'))
        fu_tx = FeatureUnionDataFrame([('bgt_occ', bgt_occ_tx), ('clean_job_title', clean_job_title_tx),
                                        ('standard_title', standard_title_tx), ('consolidated_title', consolidated_title_tx)
                                       ])       
        Xt = fu_tx.transform(df)
        Xt.columns = Xt.columns.str.replace('__resume_bg_parse','')
        Xt.columns = ['bg_r_'+x for x in Xt.columns]
        for c in [x for x in Xt.columns]:
            Xt[c] = Xt[c].where(Xt[c].str.len() > 0, '')
        Xt['bg_r_bgt_occ'] = Xt['bg_r_bgt_occ'].str.replace('NA','')
        return Xt
    def transform(self,X, **kwargs):
        r = self.resume_extract(X)
        j = self.job_extract(X)
        X = pd.concat([X,r,j],axis=1)
        return X
