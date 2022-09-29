import pandas as pd
from math import e
from itertools import chain

from dhi.dsmatch.sklearnmodeling.models.custombase import CustomTransformer
from dhi.dsmatch.sklearnmodeling.models.quantilehelpers import QuantileConfidenceTransformer

class SkillsFindingsTransformer(QuantileConfidenceTransformer,CustomTransformer):
    """Finding the match and than scoring accoring to the cluster genration.
    for eg:
    {'problem solving': 2, 'snmp': 2, 'planning': 2} , the problem solving job skill found in the 2nd geration in the profile 
    skills
    Args:df_bgt_skills_crosswalk(DataFrame)
    
    Returns:DataFrame['found_on_job','not_found_job','not_found_res','not_found_job_cluster_family']
    """
    _version = '1.0.0'
    name = 'generation_skills'
    domain = 'skills'
    def __init__(self,dd_,**kwarg):
        self.dd_ = dd_
#         print(self.dd_.columns)
    def found_skills(self,row): 
        # finding the skills if resume contains the matching skills from job
        row['found_on_job']= set(row['job_title_and_skills']).intersection(set(row['profile_title_and_skills']))
        return row
    def not_found_skills(self,row):
        # list out those job skills which do not present in resume
        row['not_found_job']= set(row['job_title_and_skills'])-set(row['profile_title_and_skills'])
        return row
    def not_found_in_low_clust_res(self,row):
        # taking out the skills from the resume which was already found a match in job skills, which will help us in not double 
        #weighing the skills.
        if len(row['not_found_job']) != 0:
            row['not_found_res'] =set(row['profile_title_and_skills']) - set(row['found_on_job'])
        else:
            row['not_found_res'] =set()
        return row
    def mapping_family(self,df,map_df,col):
        # mapping the not found skills on resume to its clusters. 
        df = df.reset_index().rename(columns={'index':'x_id'})
        d = df[['x_id',col]]
        d  = d.explode(col)
#         print(d.columns, map_df.columns)
        d  = pd.merge(d,map_df, left_on=col, right_on='skill',how='left' )
        d= d.drop(columns='skill')
        d = d.fillna('')
        d = d.groupby('x_id').agg(list)
        for c in [col, 'cluster']:
            d[c] = d[c].apply(lambda x: [ s for s in x if s!=''])
        d= d.drop(columns=col)
        d = d['cluster'].to_list()
        return d
    def transform(self,X, y=None, **kwargs):
  
        X = X.apply(self.found_skills,axis = 1)
        X = X.apply(self.not_found_skills,axis = 1)
        X = X.apply(self.not_found_in_low_clust_res,axis = 1)
#         print(X.columns, self.dd_.columns)
        X['not_found_job_cluster_family'] = self.mapping_family(X,self.dd_,'not_found_job')
        X['not_found_res_cluster_family'] = self.mapping_family(X,self.dd_,'not_found_res')
        return X
        
        

class GenerationScoreCalculationTransformer(CustomTransformer):
    """Scoring the skills and its clusters.
    - Full credict to one-to-one skills i.e exact match found in the profile skills comparing with job skills.
    - using exponential decay rate to give lesser credit with progression of titles.
    for example:
    job skills has 10 skills 
    one-to-one match = 8
    and partial score using decay funct = 0.19
    total cluster and skill weight is 8 + 0.19 = 8.19
    final score = 8.19/10 = 0.89
    
    Arg: DataFrame['found_on_job','not_found_job','not_found_res','not_found_job_cluster_family']
    
    Returns: DataFrame['one_to_one_score','partial_score','final_score']
    """
    def __init__(self, decay_rate, **kwarg):
        self.decay_rate = decay_rate
        
    def member_tier(self, member, nfcf):   
        i =0
        for m in member:
            i+=1
            if m in set(chain(*nfcf)):

                found_skill = m
                found_index = i
                break 
            else:
                found_skill = 0
                found_index = 0

        return [found_skill, found_index]

    def skill_generation(self, parent, nfcf):
        d ={}
        skill = []
        sk_gen = []
        per=[]
        for p in parent:
            per.append(p[0])
        nfj = per
        for mem in parent:
            m = self.member_tier(mem, nfcf)
            skill.append(m[0])
            sk_gen.append(m[1])
        d['skill_clust'] =skill
        d['sk_gen'] =sk_gen
        d['org_skill'] =nfj
        d = pd.DataFrame(d)
        return dict(zip(d['org_skill'], d['sk_gen']))

    def decay_func(self, P:int, t:float)->float:
        return P*pow(e, (-(self.decay_rate)*t))
    
    def calc_weight(self,row:float)->float:
        row['one_to_one_score'] = len(row['found_on_job'])
        partial_score = 0
        if len(row['cluster_found'])>0:
            generation = row['cluster_found'].values()

            for gen in generation:
                partial_score += self.decay_func(1, gen)
        row['partial_score'] = partial_score
        row['len_job_skills'] = len(row['job_title_and_skills'])
        if len(row['job_title_and_skills']) >0:# if job_title_and_skills is not empty return a score
            row['final_score'] = (row['one_to_one_score'] + row['partial_score'])/ row['len_job_skills']
        else:#return a zero score
            row['final_score']=0
        return row
    
    def transform(self,X,**kwargs):
        X['cluster_found'] = X.apply(lambda x : self.skill_generation(x['not_found_job_cluster_family'], x['not_found_res_cluster_family']), axis=1) 
        X = X.apply(lambda x : self.calc_weight(x),axis=1)
        X['final_score'] = X['final_score'].values.reshape(-1,1)
        X['confidence'] = 1.0# we are keeping confidence as 1 this time
        return X
