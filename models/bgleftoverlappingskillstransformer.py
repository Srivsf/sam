import json
from itertools import chain, accumulate

import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

from dhi.dsmatch.sklearnmodeling.models.custombase import CustomTransformer
from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer
from dhi.dsmatch.sklearnmodeling.models.leftoverlappingsetstransformer import LeftOverlappingSetsTransformer
from dhi.dsmatch.sklearnmodeling.models.pipelinehelpers import FeatureNamesPipeline
from dhi.dsmatch.sklearnmodeling.models.mixins import FeatureNamesMixin, FilterFunctionTransformer
from dhi.dsmatch.sklearnmodeling.models.quantilehelpers import QuantilePredictMixin
from dhi.dsmatch.sklearnmodeling.functiontransformermapper import applymap
from dhi.dsmatch.util.misc import assert_dataframe, assert_isstr
from dhi.dsmatch.preprocess.bgprocessing import extract_canonical_skill_names, extract_occ_codes
from dhi.dsmatch.preprocess.clean import stem


class BGLeftOverlappingSkillsTransformerCore(QuantilePredictMixin, CustomTransformer):
    _version = '1.4.3'
    name = 'overlapping_skills'
    domain = 'skills'
    
    @staticmethod
    def skills_to_cluster(s, obj=None) -> pd.DataFrame:
        """Map the Burning Glass canonical names to their skillClusterLabel.

        Args:
            s (pd.Series): Single column, where each entry is a list of BG canonical skill names.
            obj (BGLeftOverlappingSkillsTransformerCore): Essentially "self", but allows for this method to
                be called as a staticmethod.

        Returns:
            pd.DataFrame: Single column, where each entry is a list of BG skillClusterLabels, corresponding to the inputs.
        """
        if isinstance(s, pd.DataFrame):
            s = pd.Series(s.iloc[:, 0])
        df_ = s.explode()
        df_ = df_.reset_index().rename(columns={'index':'row_id', df_.name: 'skillLabel'})
        df_ = pd.merge(df_, obj.df_taxonomy_skills, on=['skillLabel'], how='left').fillna('')
        df_ = df_.groupby('row_id')['skillClusterLabel'].agg(list).apply(lambda x: [s for s in x if s != ''])
        return df_

    @staticmethod
    def bgt_to_dhi_skills(s, obj=None) -> pd.DataFrame:
        """Map the Burning Glass canonical names to corresponding DHI skills.

        Args:
            s (pd.Series): Single column, where each entry is a list of BG canonical skill names.
            obj (BGLeftOverlappingSkillsTransformerCore): Essentially "self", but allows for this method to
                be called as a staticmethod.

        Returns:
            pd.DataFrame: Single column, where each entry is a list of DHI skills, corresponding to the BG inputs.
        """
        if isinstance(s, pd.DataFrame):
            s = pd.Series(s.iloc[:, 0])
        df_ = s.explode()
        df_ = df_.reset_index().rename(columns={'index':'row_id', df_.name: 'bgtSkill'})
        df_ = pd.merge(df_, obj.df_bgt_skills_crosswalk, how='left').fillna('')
        df_ = df_.groupby('row_id')['dhiSkills'].agg(BGLeftOverlappingSkillsTransformerCore.extend_lists)
        return df_
    
    @staticmethod
    def map_lowered(s, obj=None) -> pd.DataFrame:
        """Map bgtSkill names to the DHI names if there is a mapping by their lowercase index. This removes
        duplicates such as "Data Science" and "Data science" (the last phrase has a lowercase "s"). This
        adopts the DHI name when possible.
        
        Args:
            s (pd.Series): Single column, where each entry is a list of BG canonical skill names.
            obj (BGLeftOverlappingSkillsTransformerCore): Essentially "self", but allows for this method to
                be called as a staticmethod.

        Returns:
            pd.DataFrame: Single column of lists, where each list contains DHI skills if the item is mappable 
                from Burning Glass to DHI, otherwise, the item will be the Burning Glass skill name.
        """
        if isinstance(s, pd.DataFrame):
            s = pd.Series(s.iloc[:, 0])

        df_ = s.explode()
        df_ = df_.reset_index().rename(columns={'index':'row_id', df_.name: 'bgtSkill'})
        df_.fillna('', inplace=True)
        df_['lowered'] = df_.bgtSkill.apply(str.lower)
        df_.set_index('lowered', inplace=True)
        df_ = df_.join(obj.df_lowered, how='left')#.fillna('')
        df_.skill.fillna(df_.bgtSkill, inplace=True)
        df_ = df_.groupby('row_id')['skill'].agg(BGLeftOverlappingSkillsTransformerCore.extend_lists)
        return df_
    
    @staticmethod
    def one_hot(x):
        """Note that this needs to be compatible with csr_matrix objects."""
        x[x > 1e-3] = 1
        return x
    
    @staticmethod
    def double_sqrt(x):
        """Note that this needs to be compatible with csr_matrix objects."""
        return np.sqrt(np.sqrt(x))

    @staticmethod
    def echo(x):
        """Note that this needs to be compatible with csr_matrix objects."""
        return x

    @staticmethod
    def randx(x):
        x[x.nonzero()] = np.random.random(x.getnnz())
        return x
    
    @staticmethod
    def tolist(x):
        return [x]

    @staticmethod
    def extend_lists(g):
        alist = []
        for row in g:
            if isinstance(row, list):
                alist.extend(row)
            else:
                alist.append(row)
        return alist

    @staticmethod
    def occs_to_skills(s, obj=None) -> pd.DataFrame:
        if isinstance(s, pd.DataFrame):
            s = pd.Series(s.iloc[:, 0])
        df_ = s.explode()
        df_ = df_.reset_index().rename(columns={'index':'row_id', df_.name: 'title'})
        df_ = pd.merge(df_, obj.df_title_to_skills, how='left').fillna('')
        df_ = df_.groupby('row_id')['skills'].agg(BGLeftOverlappingSkillsTransformerCore.extend_lists)
        return df_
    
    @staticmethod
    def _concat_lists(row):
        return list(chain(*row))

    @staticmethod
    def concat_lists(df):
        return df.apply(BGLeftOverlappingSkillsTransformerCore._concat_lists, axis=1).to_frame()

    def _build_stemmed_and_lowered(self):
        """Build the `self.df_stemmed` and `self.df_lowered` DataFrames. These DataFrames are indexed either
        by stemmed skill names or lowercased skill names, respectively, and those indices map to a standardized
        skill name.
        """
        skill_list = self.df_bgt_skills_crosswalk.skill
        skill_list = skill_list[skill_list.notnull()]
        skill_list = skill_list.unique()
        skill_list = skill_list.tolist()
        s2 = self.df_bgt_skills_crosswalk.explode('dhiSkills')['dhiSkills']
        s2 = s2[s2.notnull()]
        s2 = s2.unique()
        s2 = s2.tolist()
        skill_list.extend(s2)
        skill_list = list(set(skill_list))
        df_skills = pd.DataFrame(skill_list, columns=['skill'])
        df_skills['lowered'] = df_skills.skill.apply(str.lower)
        df_skills['stemmed'] = df_skills.lowered.apply(stem)
        df_skills['stemmed'] = df_skills['stemmed'].apply(lambda x: x[:-2])  # Remove '. '
        df_skills.set_index('stemmed', inplace=True)
        df_skills.sort_index(inplace=True)

        df_dhi_skills = self.df_dhi_skills_crosswalk[['dhiSkill', 'dhiSkillAliases']].copy()
        df_dhi_skills.rename(columns={'dhiSkill': 'skill'}, inplace=True)
        df_dhi_skills['dhiSkillAliases'].fillna('', inplace=True)
        df_dhi_skills = df_dhi_skills[df_dhi_skills['dhiSkillAliases'] != '']
        df_dhi_skills['dhiSkillAliases'] = df_dhi_skills['skill'] + ';' + df_dhi_skills['dhiSkillAliases']
        df_dhi_skills['dhiSkillAliases'] = df_dhi_skills['dhiSkillAliases'].apply(lambda x: x.split(';'))
        df_dhi_skills = df_dhi_skills.explode('dhiSkillAliases')
        df_dhi_skills = df_dhi_skills[~((df_dhi_skills['dhiSkillAliases'].duplicated(keep=False))&
                                        (df_dhi_skills['skill']!=df_dhi_skills['dhiSkillAliases']))]
        df_dhi_skills['lowered'] = df_dhi_skills['dhiSkillAliases'].apply(str.lower)
        df_dhi_skills['stemmed'] = df_dhi_skills.lowered.apply(stem)
        df_dhi_skills['stemmed'] = df_dhi_skills['stemmed'].apply(lambda x: x[:-2])  # Remove '. '
        df_dhi_skills.drop(['dhiSkillAliases'], axis=1, inplace=True)

        df_stemmed = df_dhi_skills[['skill', 'stemmed']]
        df_stemmed = df_stemmed[df_stemmed['stemmed'] != '']
        df_stemmed = df_stemmed.drop_duplicates(subset='stemmed')
        df_stemmed.set_index('stemmed', inplace=True)
        df_stemmed.sort_index(inplace=True)

        df_stemmed = pd.concat([df_stemmed, df_skills[['skill']]])
        df_stemmed.sort_index(inplace=True)
        idxs = df_stemmed.index.duplicated()
        df_stemmed = df_stemmed.loc[~idxs]

        df_lowered = df_dhi_skills[['skill', 'lowered']]
        df_lowered = df_lowered[df_lowered['lowered'] != '']
        df_lowered = df_lowered.drop_duplicates(subset='lowered')
        df_lowered.set_index('lowered', inplace=True)
        df_lowered.sort_index(inplace=True)
        
        df_lowered = pd.concat([df_lowered, df_skills.reset_index().set_index('lowered')[['skill']]])
        df_lowered.sort_index(inplace=True)
        idxs = df_lowered.index.duplicated()
        df_lowered = df_lowered.loc[~idxs]

        self.df_stemmed = df_stemmed
        self.df_lowered = df_lowered
                
    @staticmethod
    def extract_skills_from_title(title: str, obj=None) -> list:
        """With a given job title, extract any embedded skills. For example, "Senior Java Programmer" should
        find "Java" and "Programming" as skills, or if "Java Programming" is its own skill, get that.

        Args:
            title (str): Job title
            obj (BGLeftOverlappingSkillsTransformerCore): Essentially "self", but allows for this method to
                be called as a staticmethod.

        Returns:
            list: Any mapped skill
        """
        lowered_strs = []
        stemmed_strs = []
        splitted = list(map(lambda x: x + ' ', title.lower().split()))
        # ['senior ', 'java ', 'programmer ']
        for i in range(len(splitted)):
            # With accumulate, lowered_strs becomes: 
            # ['senior', 'senior java', 'senior java programmer', 'java', 'java programmer', 'programmer']
            # and stemmed_strs is the stemmed versions of those combinations.
            for p in accumulate(splitted[i:]):
                w = p[:-1]
                if w:
                    lowered_strs.append(w)
                w = stem(p[:-1])[:-2]  # :-2 to remove '. ' that comes from stemmer
                if w:
                    stemmed_strs.append(w)

        lowered_strs = list(set(lowered_strs))
        stemmed_strs = list(set(stemmed_strs))

        # Find the intersection of the stemmed_strs with df_stemmed
        df_ = pd.DataFrame(stemmed_strs).set_index(0).join(obj.df_stemmed, how='inner')
        # Find the intersection of the lowered_strs with df_lowered
        df_2 = pd.DataFrame(lowered_strs).set_index(0).join(obj.df_lowered, how='inner')
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
        #     print(f'diff: {idx_splitted[i]}, {set(idx_splitted[i]).difference(set(chain(*idx_splitted[i+1:])))}')
            if len(set(idx_splitted[i]).difference(set(chain(*idx_splitted[i+1:])))) == 0:
                to_drop.append(i)
        if to_drop:
            df_ = df_.loc[df_.index.delete(to_drop)]
        return df_.skill.unique().tolist()

    def __init__(self, 
                 df_taxonomy_skills=None, 
                 df_bgt_skills_crosswalk=None, 
                 df_dhi_skills_crosswalk=None,
                 skills_from_titles = False,
                 df_title_to_skills=None,
                 duplicate_scaling_func=np.sqrt, 
                 preprocessing=False,
                 **kwargs):

        if df_title_to_skills is not None and df_bgt_skills_crosswalk is None:
            raise ValueError('df_bgt_skills_crosswalk must be specified when using df_title_to_skills.')
            
        self.df_taxonomy_skills = df_taxonomy_skills
        self.df_bgt_skills_crosswalk = df_bgt_skills_crosswalk
        self.df_dhi_skills_crosswalk = df_dhi_skills_crosswalk
        self.skills_from_titles = skills_from_titles
        self.df_title_to_skills = df_title_to_skills
        self.duplicate_scaling_func = duplicate_scaling_func
        self.preprocessing = preprocessing
        
        self.df_stemmed = None
        self.df_lowered = None
        if (df_bgt_skills_crosswalk is not None) and (df_dhi_skills_crosswalk is not None):
            self._build_stemmed_and_lowered()
        
        description_cols = ['description_skills']
        resume_cols = ['resume_skills'] 
        
        jsonloads_tx = ApplyTransformer(applymap, json.loads, keys=['description_bg_parse', 'resume_bg_parse'])

        steps = [('jsonloads', jsonloads_tx)]
        
        if self.skills_from_titles:
            skillsfromtitles_tx = ApplyTransformer(applymap,
                func=BGLeftOverlappingSkillsTransformerCore.extract_skills_from_title, 
                keys={'job_title': 'description_titleskills'},
                fkwargs=dict(obj=self))
            steps.append(('skillsfromtitle', skillsfromtitles_tx))
            description_cols.append('description_titleskills')
        
        if df_title_to_skills is not None:
            get_titles_tx = ApplyTransformer(applymap, extract_occ_codes, 
                                             keys={'description_bg_parse': 'description_occ',
                                                   'resume_bg_parse': 'resume_occ'})
            titletoskills_tx = FilterFunctionTransformer(func=BGLeftOverlappingSkillsTransformerCore.occs_to_skills, 
                                                         keys={'description_occ': 'description_occ_skills', 
                                                               'resume_occ': 'resume_occ_skills'},
                                                         kw_args=dict(obj=self))
            steps.append(('gettitles', get_titles_tx))
            steps.append(('titletoskills', titletoskills_tx))
            description_cols.append('description_occ_skills')
            resume_cols.append('resume_occ_skills')
            
        names_out = []
        names_out.extend(description_cols)
        names_out.extend(resume_cols)
        bgextractskills_tx = ApplyTransformer(applymap, extract_canonical_skill_names,
                keys={'description_bg_parse': 'description_skills', 
                      'resume_bg_parse': 'resume_skills'},
                feature_names_out=names_out)
        steps.append(('bgextractskills', bgextractskills_tx))

        if df_taxonomy_skills is not None:
            skills_to_cluster_tx = FilterFunctionTransformer(
                func=BGLeftOverlappingSkillsTransformerCore.skills_to_cluster,
                keys={'description_skills': 'description_cluster_skills', 
                      'resume_skills': 'resume_cluster_skills'},
                kw_args=dict(obj=self))
            steps.append(('skillstocluster', skills_to_cluster_tx))
            description_cols.append('description_cluster_skills')
            resume_cols.append('resume_cluster_skills')

        if df_bgt_skills_crosswalk is not None:
            bgt_to_dhi_tx = FilterFunctionTransformer(
                func=BGLeftOverlappingSkillsTransformerCore.bgt_to_dhi_skills,
                keys={'description_skills': 'description_dhi_skills', 
                      'resume_skills': 'resume_dhi_skills'},
                kw_args=dict(obj=self))
            steps.append(('bgttodhi', bgt_to_dhi_tx))
            description_cols.append('description_dhi_skills')
            resume_cols.append('resume_dhi_skills')
            
        if self.df_lowered is not None:
            lowermap_tx = FilterFunctionTransformer( 
                func=BGLeftOverlappingSkillsTransformerCore.map_lowered, 
                keys={'description_skills': 'description_skills', 
                      'resume_skills': 'resume_skills'},
                kw_args=dict(obj=self))
            steps.append(('lowermap', lowermap_tx))

        if preprocessing is False:
            # Concatenate the "description" columns into one column called "description_skills" and
            # concatenate the "resume" columns into another column called "resume_skills".
            # Then pass those to the LeftOverlappingSetsTransformer.
            concatlists_tx = FilterFunctionTransformer(
                func=BGLeftOverlappingSkillsTransformerCore.concat_lists,
                keys={
                    tuple(description_cols): 'description_skills',
                    tuple(resume_cols): 'resume_skills',
                },
                feature_names_out=FilterFunctionTransformer.calling_feature_names_out
            )
            concatlists_tx.called_feature_names_out_ = ['description_skills', 'resume_skills']
            steps.append(('concatlists', concatlists_tx))
            lols_tx = LeftOverlappingSetsTransformer(duplicate_scaling_func=duplicate_scaling_func)
            lols_tx.domain = 'skills'
            lols_tx.name = 'overlapping_skills'
            steps.append(('overlapping_skills', lols_tx))
            
        steps.append(('last', 'passthrough'))

        self.pipeline = FeatureNamesPipeline(steps=steps)
        
        prediction_thresholds = kwargs.pop('prediction_thresholds', None)
        QuantilePredictMixin.__init__(self, prediction_thresholds)
        super().__init__(**kwargs) # In case there are other features.
        
    def fit(self, X, y=None, **kwargs):
        assert_dataframe(X)
        assert_isstr(X)
        self.pipeline.fit(X, y=y, **kwargs)
        return self

    def fit_transform(self, X, y=None, **kwargs):
        assert_dataframe(X)
        assert_isstr(X)
        return self.pipeline.fit_transform(X)

    def transform(self, X):
        check_is_fitted(self)
        assert_dataframe(X)
        assert_isstr(X)
        return self.pipeline.transform(X)


class BGLeftOverlappingSkillsTransformer(FeatureNamesMixin, BGLeftOverlappingSkillsTransformerCore):
    _version = '1.4.3'
    domain = 'skills'
    
    # In normal Python programming, we might be able to avoid an __init__ method that calls super() 
    # as we are doing below. However, sklearn transformers do not like *args and their member variables 
    # need to be specified explicitly. We echo the arguments of our "Core" object.
    def __init__(self, df_taxonomy_skills=None, 
                 df_bgt_skills_crosswalk=None, 
                 df_dhi_skills_crosswalk=None, 
                 skills_from_titles = False,
                 df_title_to_skills=None,
                 duplicate_scaling_func=np.sqrt, 
                 preprocessing=False, 
                 **kwargs):
        super().__init__(df_taxonomy_skills=df_taxonomy_skills,
                         df_bgt_skills_crosswalk=df_bgt_skills_crosswalk,
                         df_dhi_skills_crosswalk=df_dhi_skills_crosswalk,
                         skills_from_titles=skills_from_titles,
                         df_title_to_skills=df_title_to_skills,
                         duplicate_scaling_func=duplicate_scaling_func,
                         preprocessing=preprocessing,
                         **kwargs)
