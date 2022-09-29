import pandas as pd
import re
import itertools
from dhi.dsmatch.sklearnmodeling.models.custombase import CustomTransformer
from dhi.dsmatch.preprocess.clean import stem #.apply(stem)
from itertools import chain, accumulate
from dhi.dsmatch.preprocess.clean import clean_for_stemming, stem 
from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer
from dhi.dsmatch.sklearnmodeling.functiontransformermapper import applymap,try_func
from itertools import chain, accumulate

class MappingFileTransformer(CustomTransformer):
    
  
    def _stemmed_lowered(self,d,col) -> pd.DataFrame:
        """Converting the mapping column into 2 additional columns lowred and stemmed which contains 
        lowercase and steeming of original text."""
        d= d.fillna('')
        d_ =d
        d= d.reset_index().rename(columns={'index':'row_id'})
        d['lowered'] = d[col].str.lower()
        d['stemmed'] = d[col].apply(clean_for_stemming)
        d['stemmed'] = d['stemmed'].apply(stem)
        d['stemmed'] =d['stemmed'].apply(lambda x: x[:-2])
        d = d[['lowered','stemmed']]
        d = pd.concat([d,d_],axis=1)
        return d
    
    
   
    def transform(self,X, **kwargs):
        X = self._stemmed_lowered(X,'dhiTitles')
        X = X.fillna('')
        df_lowered = X[['lowered','dhiTitles']]
        df_lowered.set_index('lowered',inplace=True)
        df_lowered.sort_index(inplace=True)
        
        df_stemmed = X[['stemmed','dhiTitles']]
        df_stemmed.set_index('stemmed',inplace=True)
        df_stemmed.sort_index(inplace=True)
        return df_lowered, df_stemmed

class FindingMappingTransformer(CustomTransformer):
    def __init__(self,df_lowered,df_stemmed):
        self.df_lowered = df_lowered
        self.df_stemmed = df_stemmed
    def _geting_ready_previous_title(self,s):
        s = s.split(':: ')
        return s 

         
    def p_title(self,df):
        c = 'previous_title'
        df = df[['x_id',c]]
        cc = ApplyTransformer(applymap, self._geting_ready_previous_title,keys={c:f'{c}'})
        cc.transform(df)
        df = df.explode('previous_title')
        return df
    def mapping(self,title):
        title = re.sub(r'[()]', '', title)#Removing parathese from the string 
        title = title.replace('/',' / ')
        lowered_strs = []
        stemmed_strs = []
        lower_splitted = list(map(lambda x: x + ' ', title.lower().split()))
        stemmed_splitted = list(map(lambda x: x + ' ', clean_for_stemming(title).split()))
        # ['senior ', 'java ', 'programmer ']
        for i in range(len(lower_splitted)):
            # With accumulate, lowered_strs becomes: 
            # ['senior', 'senior java', 'senior java programmer', 'java', 'java programmer', 'programmer']
            # and stemmed_strs is the stemmed versions of those combinations.
            for p in accumulate(lower_splitted[i:]):
                w = p[:-1]
                if w:
                    lowered_strs.append(w)

        for i in range(len(stemmed_splitted)):
            # With accumulate, lowered_strs becomes: 
            # ['senior', 'senior java', 'senior java programmer', 'java', 'java programmer', 'programmer']
            # and stemmed_strs is the stemmed versions of those combinations.
            for p in accumulate(stemmed_splitted[i:]):

                w = stem(p[:-1])[:-2]  # :-2 to remove '. ' that comes from stemmer
                if w:
                    stemmed_strs.append(w)

        lowered_strs = list(set(lowered_strs))
        stemmed_strs = list(set(stemmed_strs))
        if (len(lowered_strs)==0 ) and (len(stemmed_strs)==0 ):
            return {}
        if (len(stemmed_strs)==0 ):
            df_=None
        else:
            # Find the intersection of the stemmed_strs with df_stemmed
            df_ = pd.DataFrame(stemmed_strs).set_index(0).join(self.df_stemmed, how='inner')
            df_['src'] ='stemmed'
        if (len(lowered_strs)>0 ):
            # Find the intersection of the lowered_strs with df_lowered
            df_2 = pd.DataFrame(lowered_strs).set_index(0).join(self.df_lowered, how='inner')
            df_2['src'] ='lowered'
            # Conjoin the two indersections
            df_ = pd.concat([df_, df_2])
        # Drop any duplicated indices and skills.
        df_ = df_.reset_index().drop_duplicates().set_index('index')
        # Filter for overlaps. If we find "java", "programming", and "java programming" as individual skills,
        # we drop "java" and "programming."
        df_['splitted'] = df_.index.str.split()
        df_['idx_num'] = df_['splitted'].apply(len)
        df_['idx_len'] = df_.index.str.len()
        df_.sort_values(by=['idx_num', 'idx_len'], ascending=True, inplace=True)
        to_drop = []
        idx_splitted = df_.splitted.values.tolist()
        for i in range(len(idx_splitted)):
            if len(set(idx_splitted[i]).difference(set(chain(*idx_splitted[i+1:])))) == 0:
                to_drop.append(i)
        if to_drop:
            df_ = df_.loc[df_.index.delete(to_drop)]

        return df_.dhiTitles.unique()
    def previous_d_f(self,df):
        df = df.fillna('')
        pt = self.p_title(df).reset_index(drop=True)
        
        c = 'previous_title'
        
        cc = ApplyTransformer(applymap, self.mapping, keys={c:f'{c}_match'})
        cc.transform(pt)
        pt = pt.groupby('x_id').apply(lambda x: [list(x['previous_title'])
                                                     ,list(x['previous_title_match'])
                                                
                                                ]).apply(pd.Series).reset_index()
        pt.columns = ['x_id','previous_title','previous_title_match']
        fc = 'previous_title_match'
        pt[fc] = pt[fc].apply(lambda x: list(itertools.chain(*x)))         
        pt[fc] = pt[fc].apply(lambda x: [s for s in x if s!='' and s!='Other'])
        pt = pt.reset_index(drop = True)
        return pt[['x_id','previous_title_match']]
    def transform(self,X, **kwargs):
        X = X.fillna('')
        X = X.reset_index().rename(columns={'index':'x_id'})
       
        pt = self.previous_d_f(X)
       
        for c in ['job_title','desired_title']:
            cc = ApplyTransformer(applymap, self.mapping, keys={c:f'{c}_match'})
            cc.transform(X)
        X = pd.merge(X,pt, on='x_id',how='left')
        return X
