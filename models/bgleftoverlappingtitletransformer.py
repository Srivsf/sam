import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')
import numpy as np
import json
import itertools
from sklearn.pipeline import Pipeline
from dhi.dsmatch.sklearnmodeling.models.leftoverlappingsetstransformer import LeftOverlappingSetsTransformer
from dhi.dsmatch.preprocess.bgjobtitleandresumetitleparsing import BGJobTitleAndResumeTitleParsing
from dhi.dsmatch.sklearnmodeling.models.crosswalkmapping import MappingFileTransformer, FindingMappingTransformer
from dhi.dsmatch.sklearnmodeling.models.cwclustermappingdhitobgtransformer import CWClusterMappingDHItoBGTransformer
from dhi.dsmatch.sklearnmodeling.models.cwclustermappingdhitobgtransformer import DecisionMakingTransformer
from dhi.dsmatch.sklearnmodeling.models.cwclustermappingbgtodhitransformer import CWClusterMappingBGtoDHITransformer
from dhi.dsmatch.sklearnmodeling.models.custombase import CustomTransformer
from dhi.dsmatch.sklearnmodeling.models.pipelinehelpers import FeatureNamesPipeline
from dhi.dsmatch.sklearnmodeling.models.mixins import FeatureNamesMixin
from dhi.dsmatch.sklearnmodeling.models.quantilehelpers import QuantilePredictMixin
from dhi.dsmatch.sklearnmodeling.models.applytransformer import ApplyTransformer
from dhi.dsmatch.sklearnmodeling.functiontransformermapper import applymap

class BGLeftOverlappingTitleTransformerCore(QuantilePredictMixin, CustomTransformer):
    #final fix
    _version = '1.1.2'
    name = 'overlapping_titles'
    domain = 'titles'
    @staticmethod
    def merge_bg_occ_and_sub_occ(bg_occ,bg_subocc):
        """Adding suboccupations on bgoccupations into bg_occ from sub_occ """
        bg_subocc =bg_subocc.rename(columns={'occ':'bgtOcc'})
        bg_subocc = bg_subocc[['bgtOcc','subOcc']]
        bg_occ = pd.merge(bg_occ,bg_subocc, on='bgtOcc', how='outer')
        return bg_occ
    @staticmethod
    def _rename_dhi_bg(bg_occ,dhi_bg):
        dhi_bg =dhi_bg.rename(columns={'dhiTitle':'dhiTitles', 
                                       'dhiSubCategory':'dhiSubCategories',
                                       'dhiMainCategory':'dhiMainCategories',
                                       'bgtOccs':'bgtOcc',
                                      'occGroups':'occGroup',
                                      'subOccs':'subOcc'})
        return dhi_bg

    @staticmethod
    def _prep_cw(df_):
        """Combining all columns from mbg_occ on bgtOccCode for because we have multiple cluster for
        1 occupation code"""
        col = df_.columns
        df_ = df_.fillna('')
        colm = [x for x in df_.columns]

        if 'bgtOccCode' not in colm:
            df_['concatenated_titles'] = df_[[x for x in df_.columns]].values.tolist()
            df_['concatenated_titles'] = df_['concatenated_titles'].apply(lambda x: list(set(x)))#remove duplicate in clusters
            df_= df_[['dhiTitles','concatenated_titles']]

        if 'bgtOccCode' in colm:
            df_['concatenated_titles'] = df_[[x for x in df_.columns if x!='bgtOccCode']].values.tolist()
            df_ = df_[['bgtOccCode','concatenated_titles']]
            df_ = df_.groupby(['bgtOccCode']).apply(lambda x: [list(x['concatenated_titles'])
                                                      ]).apply(pd.Series).reset_index()
            df_.columns = ['bgtOcc','concatenated_titles']
            df_['concatenated_titles'] = df_['concatenated_titles'].apply(lambda x: list(set(itertools.chain(*x))))
            df_= df_[['bgtOcc','concatenated_titles']]
        df_['concatenated_titles'] = df_['concatenated_titles'].apply(lambda x: [s for s in x if s!=''])
        df_['concatenated_titles'] = df_['concatenated_titles'].apply(lambda x: [s for s in x if                                                                               s!='Other'])
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
    
    def __init__(self
                 ,bg_occ=None
                 ,bg_subocc=None
                 ,dhi_bg=None
                 ,keep_first=None
                 ,keep_second=None
                 ,duplicate_scaling_func = np.sqrt
                 ,**kwargs):
        self.keep_first = keep_first
        self.keep_second = keep_second
        self.duplicate_scaling_func= duplicate_scaling_func
        #Mapping files
        self.bg_subocc = bg_subocc
        self.bg_occ = BGLeftOverlappingTitleTransformer._prep_cw(BGLeftOverlappingTitleTransformer.merge_bg_occ_and_sub_occ(bg_occ,self.bg_subocc))
        self.dhi_bg = BGLeftOverlappingTitleTransformer._prep_cw(BGLeftOverlappingTitleTransformer._rename_dhi_bg(bg_occ,dhi_bg))
        prediction_thresholds = kwargs.pop('prediction_thresholds', None)
        QuantilePredictMixin.__init__(self, prediction_thresholds)
        # Mapping file dhi_bg convert into two dataframe lowered and stemmed to find the possible match
        mapp = MappingFileTransformer()
        df_ = mapp.transform(self.dhi_bg)
        self.df_lowered = df_[0]
        self.df_stemmed = df_[1]
        #pipeline steps
        steps=[]
        lols_tx = LeftOverlappingSetsTransformer(duplicate_scaling_func= self.duplicate_scaling_func)
        lols_tx.domain = 'titles'
        lols_tx.name = 'overlapping_titles'
        jsonloads_tx = ApplyTransformer(applymap, json.loads, keys=['description_bg_parse', 'resume_bg_parse'])
        steps.append(('jsonloads', jsonloads_tx))
        steps.append(('bg_occcode_parsing', BGJobTitleAndResumeTitleParsing()))
        steps.append(('mapping_cluster', CWClusterMappingBGtoDHITransformer(self.bg_occ)))
        steps.append(('Finding_combination_for_mapping', FindingMappingTransformer(df_lowered=self.df_lowered,df_stemmed =self.df_stemmed)))
        steps.append(('current_title_mapping_cluster', CWClusterMappingDHItoBGTransformer(self.dhi_bg)))
        steps.append(('decisionmaking', DecisionMakingTransformer(self.keep_first,self.keep_second)))
        steps.append(('overlapping_titles',lols_tx ))
        steps.append(('last', 'passthrough'))
        self.pipeline = FeatureNamesPipeline(steps=steps)
    def _preprocess(self,X):
        X['desired_title'] = np.where(X['desired_title'] == '', X['current_title'], X['desired_title'])
        X = X.fillna('')
        return X

    def fit(self,X, y=None, **kwarg):
        X = self._preprocess(X)
        self.pipeline.fit(X, y=y, **kwarg)
        return self
        
    def fit_transform(self,X, y=None, **kwarg):
        X = self._preprocess(X)
        return self.pipeline.fit_transform(X)
   
    def transform(self,X,y=None, **kwarg ):
        X = self._preprocess(X)
        return self.pipeline.transform(X )

class BGLeftOverlappingTitleTransformer(FeatureNamesMixin, BGLeftOverlappingTitleTransformerCore):
    _version = '1.1.2'
    domain = 'titles'
    

    def __init__(self
                 ,bg_occ=None
                 ,bg_subocc = None
                 ,dhi_bg=None
                 ,keep_first=None
                 ,keep_second=None
                 ,duplicate_scaling_func=np.sqrt
                 ,**kwargs):
        super().__init__(bg_occ = bg_occ
                 ,bg_subocc = bg_subocc
                 ,dhi_bg = dhi_bg
                 ,keep_first = keep_first
                 ,keep_second = keep_second
                 ,duplicate_scaling_func= duplicate_scaling_func
                 ,**kwargs)
        

        
