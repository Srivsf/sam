import os
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import multiprocessing
import json
import itertools
from gensim.models import Word2Vec
from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer
from dhi.dsmatch.sklearnmodeling.functiontransformermapper import applymap, applyrows
from dhi.dsmatch.preprocess.clean import clean_for_stemming, stem
from dhi.dsmatch.util.misc import assert_dataframe, assert_isstr


from dhi.dsmatch.sklearnmodeling.models.custombase import CustomTransformer
from dhi.dsmatch import s3_ds_bucket
import re
from tqdm.auto import tqdm
from dhi.dsmatch import local_bucket
from dhi.dsmatch.util.io import read_csv
from sklearn.preprocessing import QuantileTransformer
from dhi.dsmatch import s3_ds_bucket, local_bucket
from dhi.dsmatch.sklearnmodeling.models.quantilehelpers import QuantileConfidenceTransformer
from dhi.dsmatch.sklearnmodeling.models.quantilehelpers import QuantilePredictMixin
from typing import List
from dhi.dsmatch.sklearnmodeling.models.pipelinehelpers import FeatureNamesPipeline
from dhi.dsmatch.preprocess.bgprocessing import extract_canonical_skill_names
from sklearn.preprocessing import FunctionTransformer

class ProfileSkillsWord2VecTransformerCore(QuantileConfidenceTransformer):
    """This class is to build the profile skills word2vec model, An unsupervised model which takes the columns(['previous_title', 'current_title', 'profile_skills', 'job_skills', 'desired_title', 
     'job_title', 'job_data_bg_skills', 'resume_data_bg_skills']). 
     Step 1:preprocess the data
     Step 2:Build a word2vec model from genesim
     Step 3:1 to 1 match score of each skill
     Step 4 : Linear sum assignemnet applied on the match scores of each apply event
     step 5 : quantile_transformer is applied on those scores 
          
     We will calculate quantile score (qtile), confidence score.
     It Build the quantile score and confidence score from training data. 

    Args:
            df_dhi_skills_broader_transitive (DataFrame): Use this DHI file to map to its family clusters for example 
                                                           .net---->dot_net,applic_develop,softwar_develop .net will be mapped to all the 
                                                           skills present.
            
            df_dhi_job_titles_to_related_skills (DataFrame): Use this DHI file to map the titles to its associated skills for example 
                                                                .net application developer---> dot_net_applic_develop,visual_basic_dot_net,
                                                                 microsoft_technolog,asp_dot_net,html,softwar_develop,dot_net,
                                                                 c_sharp,agil,softwar'
                                                                 
            df_crosswalk(DataFrame): Combination of DHI and BG skills cluster of a perticular skill.
            All the above mapping files have 2 columns keys and value. Keys is the string to replace with the value strings. Value strings 
            already have the key in it. In that case we are not going to lose the key skills. The value column is alreay cleaned ans 
            stemmed. All Mapping skills, profile/job skills mentioned by human and Bg parsed skills from resume and job description cleaned 
            stemmed and passed to build the model. 
            This object can take the already trained model if any. It does not provide any label.We are keeping
            confidence score as 1.0.
      
    """  
    
    
    def __init__(self, df_dhi_skills_broader_transitive, df_dhi_job_titles_to_related_skills, df_crosswalk,q,
                 w2v_model_=None,quantile_transformer_=None,transform_cols=None,**kwargs):
        self.df_dhi_skills_broader_transitive = df_dhi_skills_broader_transitive
        self.df_dhi_job_titles_to_related_skills = df_dhi_job_titles_to_related_skills
        self.df_crosswalk = df_crosswalk
        self.q = q
        self.w2v_model_=w2v_model_
        self.quantile_transformer_=quantile_transformer_
        self.transform_cols = transform_cols
        prediction_thresholds = kwargs.pop('prediction_thresholds', None)
        QuantilePredictMixin.__init__(self, prediction_thresholds)
        self.fitting=False
        
    def clean_split(self, text:str) -> List[str]:
        # lowercase input text and split on ':: ' then remove leading/trailing whitespace
        """We will use this function to convert strings of columns(privious_title, job_skills and profile_skills) into list 
        and make it ready to use the skills for mapping

         Args:
            x (string): strings

        Returns:
            list: list of strings
        """  
        items = [w.strip() for w in text.lower().split(':: ')]

        # filter out empty strings from items
        return list(filter(None, items))
    def current_title(self,x) -> List[str]:
        """We will use this function to get ready all the other titles present like , 
        desired title, current tiles and  and job titles. Removing all the prefix which starts with sr, senior, jr, junior , This will
        make it ready to use the skills for mapping

         Args:
            x (string): [string of muliple titles]

        Returns:
            list: list of Titles
        """  

        for rem in ['sr.','sr ','senior','junior','jr.']:
            x=x.lower().lstrip()
            x = re.sub(r'[^\w\s]','',x)
            x=x.split('~')
            x[:] = [item for item in x if item != '']         
            return x

    def bg_skills(self,x)-> List[str]: 
        """Using this function to get ready the burning glass parsed skills for job description 
           and resume. If it is empty convert into empty list [], else lowercse, take out abbreviation out from the pranthese like hypertext preprocessor (PHP), that will
        convert into 2 skills and convet into list of skills. 

         Args:
            x ([string]): [string of muliple titles]

        Returns:
            [list]: list of Titles
        """      

        x = list(itertools.chain(*[s.lower().split('(') for s in json.loads(x)]))#removing words in prestheses and making a spearte 
                                                                            #skills for example :File Transfer Protocol (FTP)
                                                                            # will become File Transfer Protocol,FTP
        x = [s.replace(')','').strip() for s in x]
        return x
    def bg_pre_process(self,X):
        assert_dataframe_tx = FunctionTransformer(assert_dataframe)
        assert_isstr_tx = FunctionTransformer(assert_isstr)
        jsonloads_tx = ApplyTransformer(applymap, json.loads)
        bgextractskills_tx = ApplyTransformer(applymap, extract_canonical_skill_names)
        pipeline_bg = FeatureNamesPipeline([
                                        ('jsonloads', jsonloads_tx)
                                        ,('bgextractskills', bgextractskills_tx)])
        return pipeline_bg.transform(X)

    
    
    def _util(self,df):
        """This function is used to process all above titles functions

        Args:
            df ([dataframe]): rows of strings

        Returns:
            [dataframe]: rows of list 
        """    

        df['previous_title']=df['previous_title'].apply(self.clean_split)
        df['current_title']=df['current_title'].apply(self.current_title)
        df['desired_title']=df['desired_title'].apply(self.current_title)
        df['job_title']=df['job_title'].apply(self.current_title)
        df['job_data_bg_skills']=self.bg_pre_process(df['description_bg_parse'])
        df['resume_data_bg_skills']=self.bg_pre_process(df['resume_bg_parse'])
        df['job_skills']=df['job_skills'].apply(self.clean_split)
        df['profile_skills']=df['profile_skills'].apply(self.clean_split)
        return df

    def tokenize(self,x) -> List[str]:
        """Tokenising the skill phrase and convert them into words for example java devwlopment will become java and development, convert them into list after removing
        duplicatesif any.

        Args:
            x ([list]): [list of skill phrases]

        Returns:
            [list]: [list of tokens]
        """    

        if type(x)== str:
            x=x.split(' ')
            x=[s for s in x if s!='']        
        elif type(x) == list:
            x=[s for s in x if s!='']
            x=[s.split(' ') for s in x]        
            x=list(itertools.chain(*x))
            x=list(set(x))  
        #x[:] = [item for item in x if item != '']
        return x 

    def mapping(self,df,col,map_df):#df is the original dataframe, map_df is the datframe have mapped values as key and value
        """This function is used to map the skills to its cluster.
        all the mapping files are  converted into 2 columns keys and value.
        We will find the key and map to its family cluster. The functio will edit the skills on the original dataframe its advisable to copy 
        the datafrme to other variable and than concat to the original datframe

        Args:
            df (dataframe):
            col (type): column whose skills rows  needs to be mapped to its clustes
            map_df (dataframe): Mapping file it can be croass walk file or DHI file. 
        """    
        c=[x for x in map_df.columns]
        sam=dict(zip(map_df[c[0]], map_df[c[1]]))#converting into list
        for index, profile_row in enumerate(tqdm(df[col],total=df.shape[0])):# loop over each rpw
            profile_keys = set([item.lower().strip() for item in profile_row])#removing the duplicate 
            new_val=[value for key, value in sam.items() if key in profile_keys]#print value from the dict if key found in dict
    #         new_val=list(set(new_val))
            new_val[:] = [item for item in new_val if item != '']#removing empty items if exixt
            df.at[index, col] = new_val

    def extract_skill_from_titles(self,df,col,map_df):# Finding the titles or skills from the dhi-title-to-skills file and map it with its associated skills 
        """this function is to map the titles and skills to its associated skills, we are going to use dhi tilte to skill file for mapping.
        We will find the key and map to its family cluster. The functio will edit the skills on the original dataframe its advisable to copy 
        the datafrme to other variable and than concat to the original datframe

        Args:
            df ([dataframe]):
            col ([type]): column whose skills rows  needs to be mapped to its clustes
            map_df ([dataframe]): Mapping file it can be croass walk file or DHI file. 
        """ 

        c=[x for x in map_df.columns]
        sam=dict(zip(map_df[c[0]], map_df[c[1]]))  # converting mapping file into a dict
        for index, profile_row in enumerate (df[col]):#enumrates to each row
            profile_row=[x for x in profile_row if len(x)>0]        
            profile_keys = set([item.lower().strip() for item in profile_row])#remove duplicates
            new_key=[key for key, value in sam.items() if key in profile_keys]
            new_val=[value for key, value in sam.items() if key in profile_keys]#print value from the dict of  if key found in dict
            new_val[:] = [item for item in new_val if item != '']# removing empty items if exixt
            df.at[index, col] = new_val


    def _mapping_skills_from_title(self,df):
        """Finding the titles from the dhi-titles-skill file and map to its associated cluster through
        #extract_skill_from_titles function than adding the all profile related skills into map_job and job related skills into map_job
        #and concating the mapped skills column into main data column

        Args:
            df (dataframe): 
        Returns:
            dataframe with 2 added columns map_job and map_prof ]: with added skills in map_job and map_prof columns which is mapped by 
            crosswalk file.    
            """    

        df_=df[['previous_title', 'current_title', 'desired_title', 'job_title']]#will pick only thsese columns for extraction.
        #dhi_t=self.df_dhi_job_titles_to_related_skills#pd.read_csv(os.path.join(local_bucket, 'data','dice','mapping-data', 'dhi_t.csv'))
        for c in tqdm(df_.columns, total=len(df_.columns)):
            self.extract_skill_from_titles(df_,c,self.df_dhi_job_titles_to_related_skills)

        job=['job_title']#adding the all job related skills
        prof=['previous_title', 'current_title', 'desired_title']#adding the all profile related skills

        df_['map_job']=df_[job]
        df_['map_prof']=df_[prof[0]]+df_[prof[1]]+df_[prof[2]]
        col=['map_job','map_prof']
        df_=df_[col]
        df=pd.concat([df,df_],axis=1)# concat to the original dataframe
        return df

    def _mapping_skills_phrase(self,data):# tokenizing the skill phrases than find the skill cluster and add its family clusters in the skills 
        """This function is to map tokenised skill phrases and than map it to its family cluster

        Args:
            data (dataframe): [description]
            cw (dataframe , Crosswalk file): Mapping File where skills cluster is present
          
        Returns:
            [dataframe]: with added skills in map_job and map_prof columns which is mapped by crosswalk file.
        """    

        df_=data[['previous_title', 'current_title', 'desired_title', 'job_title','profile_skills','job_skills']]# creating a new dataframe df_ with only processing column
        #cw=pd.read_csv(os.path.join(local_bucket, 'data','dice','mapping-data', 'cw.csv'))#This cross wlk file saved in local bucket
        for t_col in df_.columns:
            df_[t_col]=df_[t_col].apply(self.tokenize)#tokenising the columns
            self.extract_skill_from_titles(df_,t_col,self.df_crosswalk)#using extract_skill_from_titles function to map the skills    
        df_['jj']=df_['job_title']+df_['job_skills']#Concating all the skills mapped from job title and skills
        df_['pp']=df_['previous_title']+df_['current_title']+df_['desired_title']+df_['profile_skills']#Concating all the skills mapped from profile titles and skills
        df_=df_[['jj','pp']]
        data=pd.concat([data,df_],axis=1)#Concating df_ and original dataframe
        data['map_job']=data['jj']+data['map_job']# combing current mapped skills with previous mapped skills of job which was present in map_job column
        data['map_prof']=data['pp']+data['map_prof']# combing current mapped skills with previous mapped skills of profilewhich was present in map_prof column
        data=data.drop(columns=['jj','pp'])# droping these column , beacuse its skills is already being added to the map_job and map_prof columns
        return data
    
    def _mapping_cw_dhi(self,data):
        """mapping skills with crosswalk and dhi files both together with this function

        Args:
            data (dataframe): 

        Returns:
            data([dataframe]): dataframe with added skills in map_job and map_prof columns (extracted from crosswalk and dhi )  
        """    

        df_=data[['ex_id','profile_skills','job_skills','resume_data_bg_skills','job_data_bg_skills']]
        #cw=pd.read_csv(os.path.join(local_bucket, 'data','dice','mapping-data', 'cw.csv'))
        #dhi=pd.read_csv(os.path.join(local_bucket, 'data','dice','mapping-data', 'dhi.csv'))
        for c in df_.columns[1:]:
            self.mapping(df_,c,self.df_crosswalk)
            self.mapping(df_,c,self.df_dhi_skills_broader_transitive)
        df_.columns=df_.columns.map(lambda x : 'dh_cw_' +x if x !='ex_id' else x)
        df_['prof']=(df_['dh_cw_resume_data_bg_skills']+df_['dh_cw_profile_skills']).apply(set).apply(list)
        df_['jobs']=(df_['dh_cw_job_data_bg_skills']+df_['dh_cw_job_skills']).apply(set).apply(list)
        df_=df_[['prof','jobs']]
        data=pd.concat([data,df_],axis=1) 
        data['map_prof']=data['map_prof'] +data['prof']
        data['map_job']=data['map_job'] +data['jobs']
        for c in ['map_prof','map_job']:
            data[c]=data[c].apply(set).apply(list)
            data[c]=data[c].apply(lambda x: [item for item in x if item != ''])
            data[c]=data[c].apply(lambda x:  ','.join([str(elem) for elem in x]))
        #data=data.drop(columns=['prof','jobs'])
        return data

    def _stemming_data(self,df,c):
        """stemming skills phrases and combined it with an underscore and converting it into a string from list

        Args:
            df (dataframe): dataframe
            

        Returns:
            dataframe: with 1 added columns containing stemmed skills
        """    
        df_job=df[['ex_id',c]]
        df_job=df_job.explode(c).reset_index()# exploding the list into multiple row.
        df_job
        df_job=df_job.rename(columns={'index': 'index_val',c:'stem_value'})
        df_job=df_job.fillna('')
        clean_tx = ApplyTransformer(applymap, clean_for_stemming)
        stem_tx = ApplyTransformer(applymap, stem)
#         cleaner_stemmer = make_cleanstem_pipeline(['stem_value'])# this cleaning process in dsmatch
#         clean_cols = [f'{c}_clean' for c in ['stem_value']]
        j = clean_tx.transform(df_job[['stem_value']])# cleaning
        j = stem_tx.transform(j)# stemming 
        j=j.rename(columns={'stem_value':f'stemmed_{c}'})
        j[f'stemmed_{c}']=j[f'stemmed_{c}'].str.rstrip('. ')
        j[f'stemmed_{c}']=j[f'stemmed_{c}'].str.replace(' ','_')
        df_job=pd.concat([df_job,j],axis=1)# concating cleaned columns with original exploded dataframe.
        df_job=df_job.groupby(['index_val','ex_id'])[f'stemmed_{c}'].apply(','.join).reset_index()# group by index_value and ex_id
        df_job[f'stemmed_{c}']=df_job[f'stemmed_{c}'].str.split(',')
        df_job[f'stemmed_{c}']=df_job[f'stemmed_{c}'].apply(set).apply(list)
        df_job[f'stemmed_{c}']=df_job[f'stemmed_{c}'].apply(lambda x: [item for item in x if item != ''])
        df_job[f'stemmed_{c}']=df_job[f'stemmed_{c}'].apply(lambda x:  ','.join([str(elem) for elem in x]))
        df_job.drop('index_val', axis=1, inplace=True) 
        df_job.drop('ex_id', axis=1, inplace=True) 
        df=pd.concat([df,df_job],axis=1)
        return df

    def _final_data(self,data):
        """Finally adding all the cleaned and stemmed skills and map mapped skills into one pot for each profile and job.

        Args:
            data (Dataframe): description

        Returns:
            dataframe: with 2 added columns final_profile_skills and final_job_skills which contains all the skills including mapping  and stemming.
        """    

        data['final_profile_skills']=data['stemmed_resume_data_bg_skills'] + ',' + data['map_prof']+ ',' + data['stemmed_profile_skills']#com
        data['final_job_skills']=data['stemmed_job_data_bg_skills'] + ',' + data['map_job']+ ',' + data['stemmed_job_skills']
        return data

    def _duplicate_removal(self,k): 
        """Remove duplicate which are next to each, we will get the final string in a string format,
         beause of mappoing there are lot of words come next to each other for example if string is 'research,bilingu,bilingu,communic,natur_languag,bilingu,bilingu'
        this function will convert it into 'research,bilingu,communic,natur_languag,bilingu, it will not remove all the duplicates just removing the duplicate skill next to each other.

        Args:
            k (string):string of skills 
        Returns:
            string: strings after removing duplicates next to each other
        """     
        k=k.split(',')
        z=[]
        for x,y in zip(k,k[1:]):
            if (x!=y) & (x!=''):
                z.append(x)
        z=list(set(z))
        z=','.join([str(elem) for elem in z])
        return z

    def preprocess_data(self,data):
        """This function is  combination of all the preprocessing steps and mapping the cluster. 

        Args:
            data (dataframe): dataframe with raw skills and titles

        Returns:
            dataframe: final dataframre with processed skills in string fromat
        """    
        data=data.fillna('')
        data=data.reset_index()
        data=data.rename(columns={'index':'ex_id'})
        data=self._util(data)# calling '_util' function for all the skills and titles columns  to use in the below functions 
        data=self._mapping_skills_from_title(data)# calling mapping_skills_from_title funtion for title mapping
        data=self._mapping_skills_phrase(data)# calling mapping_skills_phrase funtion for tokens mapping after tokenising the skill phrase
        col=['job_data_bg_skills','resume_data_bg_skills','profile_skills','job_skills']# stemming the skills of these columns
        for x in col:
            data=self._stemming_data(data,x)# calling stemming_data function for stemming the above columns
        data=self._mapping_cw_dhi(data)#calling mapping_cw_dhi to map the skills with cross walk and dhi cluster
        data=self._final_data(data)#calling final_data to concat all the above processed data into one
        data['final_profile_skills']=data['final_profile_skills'].apply(self._duplicate_removal)#removing duplicate string next to each other
        data['final_job_skills']=data['final_job_skills'].apply(self._duplicate_removal)     
        return data

    def exploding_columns(self,df):
        """This function we will use after we have preprocessed test data, we will explode it for skill to skill scoing

        Args:
            df (dataframe): description
            profile (column): profile column for exploding skills
            job (column): job column for exploding skills

        Returns:
            datframe: dataframe with exploded skills for 1:1 mapping
        """
        job='final_job_skills'
        profile='final_profile_skills'
        for x in tqdm([profile,job]):# taking job and profile skills after preprocessing 
            df[x]=df[x].str.split(',')# Split them will a comma
        df=df.drop_duplicates(subset=['ex_id'])#we are droping duplicates, test set is lablled by 3 labeller, to avoide problem in 1:1 mapping on each ex_id .
        p=df[['ex_id',profile]]#creating a dataframe for profile with ex_id
        p=p.explode(profile)#exploding the profile skills
        j=df[['ex_id',job]]#creating a dataframe for job with ex_id
        j=j.explode(job)#exploding the job skills
        df_=pd.merge(j,p,how='outer',left_on=['ex_id'],right_on=['ex_id'])#mergig the j and p dataframe on ex _id
        df_=df_.fillna('')
#         df_=df_[df_[job]!='']#Removing any empty job row
#         df_=df_[df_[profile]!='']#Removing any empty profile row
#         df_=df_.reset_index(drop=True)
        return df_

    def scoring(self,df):
        """This function we will use for geting the similrity score for each job skill with profile skill. This will work only if we have exploded the model and have 1 
        skill in each row.  This is only for word2vec model , after exploding each skill on ex_id getting a similarity score by  
        wv.similarity. 

        Args:
            df (exploded dataframe): dataframe must go trough exploding_columns function
            model (type): model for scoring the skills, can be word2vec
        Returns:
            [Dataframe]: with added word2vec_score column, which contains the scores of maching between job and profile skill
        """    

        sc=[]
        job='final_job_skills'
        profile='final_profile_skills'
        for jd_sent,res_sent in tqdm(zip(df[job],df[profile]),total=df.shape[0]):#zipping job and profile skills on index fro 1:1 scoring
            if (jd_sent != '') & (res_sent != ''):# if skill row in profile and job are not empty
                try:
                    score=self.w2v_model_.wv.similarity(jd_sent, res_sent)#get a similary score
                    sc.append(score)
                except Exception  as e:
                    sc.append(0.0)#else append 0
            else:
                sc.append(0.0)

        df['word2vec_score']=sc
        df['word2vec_score']=df['word2vec_score'].fillna(0.0)#if no score present or skill not present in the vocabulary give it a 0 score
        return df


    def linear_sum_assign(self,df_):
        """After getting a similarity scores apply optimal assignment, get  a mean score or quantile score and than convert 
        into label between 1 to 5, this function will work after using scoring function. When we have the similarity  scores of each job and profile skills, linear sum algorithm will help us t
        to pick the appropriate one

        Args:
            df_ ([dataframe]): exploded datframe with scoring word2vec_score columns
            prof ([column]): profile columns with exploded skill
            job ([column]): job columns with exploded skill
            data ([dataframe]): original datframe where these scores added on ex_id

        Returns:
            [orignal dataframe]: with 1 added word2vec_map_score columns containd list of match scores picked by
            linear sum assignment
        """    
        job='final_job_skills'
        prof='final_profile_skills'
        by_row = df_.groupby('ex_id')#groupby ex_id
        matches=[]
        scores = {}
        for name, g in tqdm(by_row):
            g[job]=g[job].replace('','None')
            g[prof]=g[prof].replace('','None')
            g_ = g.pivot_table(index=['ex_id',job], columns=prof, values='word2vec_score',dropna=False) # converting job skill on rows and profile skills as column
            g_=g_.fillna(0)
            row_matches, col_matches=linear_sum_assignment(g_.values,maximize=True) #applying optimal assignment
            f = []
            s = []
            for r, c in zip(row_matches, col_matches):
                f.append((g_.index[r] , g_.columns[c], g_.iloc[r, c]))
                s.append(g_.iloc[r, c])
            matches.append(f)
            scores[name] = {'rows': row_matches, 'cols': col_matches, 'vals': s}
        sc= pd.DataFrame.from_dict(scores, orient='index')
        sc['matches']=matches
        return sc

    def quant(self,lst_test,preproc_test):
        """Here we will apply quantile .80 on each apply event.

        Args:
            lst_test (dataframe): Dataframe with match scores of pairwise comparision of skilla after optimal
                                assignment.
            preproc_test (dataframe): original dataframe to merge lst_test
            q(quantile value): initilized in _init_

        Returns:
            preproc_test(dataframe): With all the columns and lst_test scores after applying quantile.
        """

        col='vals'
        fin=[]
        for ls in lst_test[col]:
            if len(ls)>0:
                f=np.quantile(ls,self.q)
                fin.append(f)
            else:
                fin.append(0)
        lst_test['q_val']=fin
        lst_test=lst_test.reset_index()
        lst_test=lst_test.rename(columns={'index':'ex_id'})
        lst_test=lst_test[['ex_id',col,'q_val']]
        preproc_test=preproc_test.merge(lst_test,  on ='ex_id', how='left')#merging the lst_test dataframe to the
                                                                          #original dataframe
        return preproc_test


    def quantile_score(self,data):
        """After getting the mean score of optimal assignment, get  a mean score or quantile score, train it if
        not present, else used the trained quantile.
        Args:
            data (type): raw mean scores generated after optimal assignment

        Returns:
            type(int): quantile score
        """ 
        if self.fitting:
            self.quantile_transformer_ = QuantileTransformer(n_quantiles=1000, random_state=0)
            self.quantile_transformer_.fit_transform(data['q_val'].values.reshape(-1, 1))
        data['qtile'] = self.quantile_transformer_.transform(data['q_val'].values.reshape(-1, 1))#getting a quantile score
        data['qtile']=data['qtile'].fillna(0.0)
        data['confidence']=1.0#keeping the confidence as 1
        return data
    

    def vocab_w2b(self,train):
        """Building vocabulary 

        Args:
            train (type): Combined job and profile skills mapped from DHI and CW files cleaned and stemmed.

        Returns:
            type: word2vec gensim vocabulary model
        """ 
        l=(train['final_profile_skills']+' '+train['final_job_skills'])
        l=pd.DataFrame(l)
        l[0]=l[0].fillna('')
        l[0]=l[0].str.split(',')
        l[0]=l[0].apply(lambda x :[ s for s in x if s!=''])
        corpus=l[0].to_list()
        cores = multiprocessing.cpu_count()
        model = Word2Vec(corpus, min_count = 1 ,vector_size=500, window =25 , sg = 1, workers=cores ,sample=1e-5, max_vocab_size=None, alpha=0.01,negative=20)
        return model
    
    def fit (self, X, y=None, **kwargs):
        """Traing the model with all above functions. It takes raw data from the columns mentioned above train a word2vec model if not 
        trained before, get 1:1 similarity scores of all the skills on each apply event. and train a quanltile transformer on the final 
        match score calculated. 

        Args:
            X (Dataframe): ['previous_title', 'current_title', 'profile_skills', 'job_skills', 'desired_title', 
     'job_title', 'job_data_bg_skills', 'resume_data_bg_skills'].
            y :  Defaults to None.
        Returns:
            Dataframe: qtile and confidence score  and trained word2vec model
        """    
        
        self.fitting=True
        X = self.preprocess_data(X)        
        if self.w2v_model_ is None:
            self.w2v_model_=self.vocab_w2b(X)          
        df_ = self.exploding_columns(X)
        df_ = self.scoring(df_)
        df_=self.linear_sum_assign(df_)
        X = self.quant(df_,X)
        X=self.quantile_score(X)
        self.transform_cols = X.columns
        self.fitting=False        
        return self
    


    def transform(self, X, y=None, **kwargs):
        """The columns ['previous_title', 'current_title', 'profile_skills', 'job_skills', 'desired_title',   
           'job_title', 'job_data_bg_skills', 'resume_data_bg_skills'] contains a list of skills, these lists are mapped to the DHI and Cw 
           files for cluster skills and then exploded for pairwise comparison using the word2vec model. Applied Linear sum assignment on 
           the scores to get the best mapping between the skills of the two lists. Take the threshold scores out from the list and apply
           the mean. Then apply a quantizer to calculate the quantile score.
        Args:
            X (dataframe): ['previous_title', 'current_title', 'profile_skills', 'job_skills', 'desired_title', 
             'job_title', 'job_data_bg_skills', 'resume_data_bg_skills']
            y :  Defaults to None.

        Returns:
            dataframe: qtile and confidence
        """    
        X = self.preprocess_data(X)        
        df_ = self.exploding_columns(X)
        df_ = self.scoring(df_)
        df_ = self.linear_sum_assign(df_)
        X = self.quant(df_,X)
        X = self.quantile_score(X)
        self.transform_cols = X.columns
        return X
from dhi.dsmatch.sklearnmodeling.models.mixins import FeatureNamesMixin
class ProfileSkillsWord2VecTransformer(FeatureNamesMixin, ProfileSkillsWord2VecTransformerCore):
    _version = '0.1'
    domain = 'skills'
