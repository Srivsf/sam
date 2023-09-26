import subprocess
import time
import unicodedata
import sys
import logging
from datetime import datetime
import json
import pickle
import re
import numpy as np
import traceback
import os
import zipfile
from collections.abc import Iterable
import joblib
import sklearn
import pandas as pd
from nltk import ngrams

logging.basicConfig(filename="./run_cognitive.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
# Set the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

################################################
# Download the models and data assets from WML
################################################
####CODE for WKC
'''
## commented for local testing
from ibm_watson_machine_learning import APIClient

wml_credentials = {
    "url": "https://mariadev.ebiz.verizon.com",
    "username": "dasan55",
    "token": os.environ["USER_ACCESS_TOKEN"],
    "instance_id": "wml_local",
    "version" : "4.0"         
}
space_id =  asset_details["space_id"]

client = APIClient(wml_credentials)
client.set.default_space(space_id)

base_download_path = "/opt/ibm/scoring/python/workdir"

#models_dir = f'{base_download_path}/models'
models_dir = f'{base_download_path}/models/models'

#if  os.path.exists(base_download_path):
#    logging.info(f"Custom log msg0306. The {base_download_path} exists")

data_dir = f'{base_download_path}/data'
#data_dir = f'{base_download_path}'

os.mkdir(data_dir)


# Download the models zip
models_file_name = f'{base_download_path}/models.zip'
client.repository.download(asset_details["models"]["asset_id"], filename=models_file_name)
# Extract contents
with zipfile.ZipFile(models_file_name, 'r') as zip_ref:
        zip_ref.extractall(base_download_path)

# Download data files
# Lookup
client.data_assets.download(asset_details["data"]["Lookup_latest"]["asset_id"],
                            f'{data_dir}/{asset_details["data"]["Lookup_latest"]["file_name"]}'
                           )
# BT_GUID_MAPPING
client.data_assets.download(asset_details["data"]["BT_GUID_MAPPING"]["asset_id"],
                            f'{data_dir}/{asset_details["data"]["BT_GUID_MAPPING"]["file_name"]}'
                           )  

# Regex Rules
client.data_assets.download(asset_details["data"]["regex_rules"]["asset_id"],
                            f'{data_dir}/{asset_details["data"]["regex_rules"]["file_name"]}'
                           )     

#     # Standard Tokens
#     client.data_assets.download(asset_details["data"]["std_tokens_to_submit"]["asset_id"],
#                                 f'{data_dir}/{asset_details["data"]["std_tokens_to_submit"]["file_name"]}'
#                                )       

#     # Input Data
#     client.data_assets.download(asset_details["data"]["LabelledData_GCP_01_26_23"]["asset_id"],
#                                 f'{data_dir}/{asset_details["data"]["LabelledData_GCP_01_26_23"]["file_name"]}'
#                                ) 


################################################
# Load all the models 
################################################     
bayes_model_file = f'{models_dir}/{asset_details["models"]["file_names"]["Bayes"]}'
bayes_label_file = f'{models_dir}/{asset_details["models"]["file_names"]["Bayes_Labels"]}'
bm25_model_file = f'{models_dir}/{asset_details["models"]["file_names"]["BM25"]}'
bm25_label_file = f'{models_dir}/{asset_details["models"]["file_names"]["BM25_Labels"]}'
knn_model_file = f'{models_dir}/{asset_details["models"]["file_names"]["NearNghbr"]}'
knn_label_file = f'{models_dir}/{asset_details["models"]["file_names"]["NearNghbr_Labels"]}'

trainingNBModel = joblib.load(bayes_model_file)
label_nb = joblib.load(bayes_label_file)
bm25 = joblib.load(bm25_model_file)
label_bm25 = joblib.load(bm25_label_file)
NN1_Model = joblib.load(knn_model_file)
label_1nn = joblib.load(knn_label_file)
'''
####CODE for WKC

####CODE for LOCAL
################################################
# Load all the models
################################################
bayes_model_file = 'data/wkc_depl_files/models/Bayes.pkl'
bayes_label_file = 'data/wkc_depl_files/models/Bayes_Labels.pkl'
bm25_model_file = 'data/wkc_depl_files/models/BM25.pkl'
bm25_label_file = 'data/wkc_depl_files/models/BM25_Labels.pkl'
knn_model_file = 'data/wkc_depl_files/models/NearNghbr.pkl'
knn_label_file = 'data/wkc_depl_files/models/NearNghbr_Labels.pkl'

# -------Future Release-----------------------------
dt_sbert_model_file = 'data/wkc_depl_files/models/DT_SBERT.pkl'
dt_sbert_labels_file = 'data/wkc_depl_files/models/DT_SBERT_labels.pkl'
dt_model_file = 'data/wkc_depl_files/models/DT.pkl'
dt_labels_file = 'data/wkc_depl_files/models/DT_labels.pkl'
nn_sbert_labels_file = 'data/wkc_depl_files/models/NearNghbr_SBERT_Labels.pkl'
nn_sbert_model_file = 'data/wkc_depl_files/models/NearNghbr_SBERT.pkl'
# -------Future Release-----------------------------
trainingNBModel = joblib.load(bayes_model_file)
label_nb = joblib.load(bayes_label_file)
bm25 = joblib.load(bm25_model_file)
label_bm25 = joblib.load(bm25_label_file)
NN1_Model = joblib.load(knn_model_file)
label_1nn = joblib.load(knn_label_file)


# -------Future Release-----------------------------
trainingDTSBERTModel = joblib.load(dt_sbert_model_file)
trainingKNNSBERTModel = joblib.load(nn_sbert_model_file)
label_knn_sbert = joblib.load(nn_sbert_labels_file)
trainingDTModel = joblib.load(dt_model_file)
# -------Future Release-----------------------------



####CODE for LOCAL


########################################################################
## Following code has been taken from Standardization.stdz_data.py
########################################################################
def updateWithStndzdValue(feature_val, lookUp):
    """
    update the value with standardized value
    :param feature_val:
    :param lookUp:
    :return:
    """
    try:
        words = feature_val.split(" ")
        stdzd_feature_val = ""
        for w in words:
            wrd = w.strip().upper()
            stdzd = lookUp.get(wrd)  # lookup[wrd]
            if stdzd:
                val = lookUp.get(wrd).get('value')
                stdzd_feature_val = stdzd_feature_val + " " + str(val)
            else:
                stdzd_feature_val = stdzd_feature_val + " " + w.strip()
    except Exception as e:
        logger.error('Error in updateWithStndzdValue Fucntion: ', e)
        raise
    return stdzd_feature_val.strip()

def cleanup_standardize(data, col_list, col_name, col_name_stdzd, label, latest_lookup):
    try:
        logger.info("in cleanup_standardize(data)")
        if label in col_list:
            col_list.remove(label)
        else:
            pass
        data[col_list] = data[col_list].applymap(
            lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', errors='ignore').decode('utf-8'))

        data[col_list] = data[col_list].replace('[#,@,&,:,,!,%,-,<,>,/]',
                            '', regex=True)
        data[col_list] = data[col_list].replace('\(', '', regex=True)
        data[col_list] = data[col_list].replace('\)', '', regex=True)
        data[col_list] = data[col_list].replace('\{', '', regex=True)
        data[col_list] = data[col_list].replace(r"\}", '', regex=True)
        data[col_list] = data[col_list].replace(r"\|", ' ', regex=True)
        data[col_list] = data[col_list].replace(r'\"', '', regex=True)
        data[col_list] = data[col_list].replace(r'\[', '', regex=True)
        data[col_list] = data[col_list].replace(r'\]', '', regex=True)
        data[col_list] = data[col_list].replace(r'\.', ' ', regex=True)
        data[col_list] = data[col_list].replace(r'\. ', ' ', regex=True)
        data[col_list] = data[col_list].replace(r"\'", ' ', regex=True)

        data[col_name_stdzd] = data[col_name]
        data[col_name_stdzd] = data[col_name_stdzd].str.replace("_", " ")
        data[col_name_stdzd] = data[col_name_stdzd].str.replace("\-", "", regex=True)
        data[col_name_stdzd] = data[col_name_stdzd].str.replace("\+", "", regex=True)
        logger.info("********************************************")
        logger.info(data.tail(5))

        data[col_name_stdzd] = data[col_name_stdzd].replace(r'\b[0-9][0-9.,-]*\b', '', regex=True)
        data[col_name_stdzd] = data[col_name_stdzd].replace(r"\b[a-zA-Z]{1}\b", '', regex=True)
        data[col_name_stdzd] = data[col_name_stdzd].str.strip()
        data[col_name_stdzd] = data[col_name_stdzd].str.lower()
        logger.info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        logger.info(data.tail(5))

        logger.info(
            f"number of NA records in {col_name_stdzd} = {len(data[data[col_name_stdzd].isna()])}")
        logger.info(f" columnnames = {data[data[col_name_stdzd].isna()][col_name].unique()}")
        data = data.dropna(subset=[col_name_stdzd])
        logger.info(
            f"After removing NA --> number of NA records in {col_name_stdzd} = {len(data[data[col_name_stdzd].isna()])}")

        #lookup = pd.read_csv(r'C:\CC\NextGen_CC\ClassifyRules\Lookup_latest_v10.csv')

        # df_column_names_tokenized = pd.read_csv('data/column_names_tokenized.csv')
        # logger.info(f"df.head() =={df_column_names_tokenized.head()}")
        #
        # df_column_names_tokenized['key'] = df_column_names_tokenized['key'].str.upper()
        # column_names_tokenized_dict = df_column_names_tokenized.set_index('key').T.to_dict()
        #
        # data[col_name_stdzd] = data[col_name_stdzd].apply(lambda row: updateWithStndzdValue(row,column_names_tokenized_dict))
        #
        # df = pd.read_csv('data/std_tokens_to_submit.csv')
        # logger.info(f"df.head() =={df.head()}")
        #
        # df['key'] = df['key'].str.upper()
        # lookUp_dictnry = df.set_index('key').T.to_dict()
        #
        # data[col_name_stdzd] = data[col_name_stdzd].apply(lambda row: updateWithStndzdValue(row,lookUp_dictnry))

        df = pd.read_csv(latest_lookup)
        logger.info(f"df.head() =={df.head()}")

        df['key'] = df['key'].str.upper()
        lookUp_dictnry = df.set_index('key').T.to_dict()

        data[col_name_stdzd] = data[col_name_stdzd].apply(lambda row: updateWithStndzdValue(row, lookUp_dictnry))
    except Exception as e:
        print('Error in Cleanup Standardize Fucntion: ', e)
        raise

    return data

def standardization_main(ip_data_df, col_name, col_name_stdzd, label, latest_lookup):
    try:
        ip_data_df = ip_data_df.astype('string')
        # ip_data_df = ip_data_df[:2000]
        col_list = ip_data_df.columns.values.tolist()
        op_data_df = cleanup_standardize(ip_data_df, col_list, col_name, col_name_stdzd, label, latest_lookup)
        op_data_df = op_data_df.dropna()
        op_data_df = op_data_df.astype('string')
        return op_data_df
        # op_data_df['final_element_id'] = op_data_df['final_element_id'].str.replace('.0','')
    except Exception as e:
        print('Error in Standardization Main Fucntion: ', e)
        raise



###############################################################################
## Following code has been taken from RegexClassifier.regexmodeltransformer.py
###############################################################################
class RegExModelTransformer:
    """
    Arg:
        This class is used to predict the Regex model classes.
    Input:
        The column need to be standerdised first
        ruletable-> This is the regex pattern rule table (Data Frame) where column 1 is the policytagid and column2 is regex pattern.
        pd.Series-> pass a Series for example we have to get the prediction of COLUMN_NAME_STDZD column then pass df['COLUMN_NAME_STDZD']

    return[pd.DataFrame]:
        It returns the prediction, extracted strings , regex pattern (used for extraction)into datafrane format.
        if result_type is test then it retyrn p
    """
    _version = 1.1
    _domain = 'RegEx'
    _mapping_files = 'rule_table'

    def __init__(self, rule_table: pd.DataFrame, result_type=None, **kwargs) -> pd.Series:
        self.rule_table = rule_table
        #         self.joint_words_table = joint_words_table
        #         self.lookup_table = lookup_table
        self.result_type = result_type

    def extract_from_regex_rule_table(self, x: str) -> str:
        self.rule_table = self.rule_table.fillna('')
        col1 = self.rule_table.columns[0]  # policytagid column
        col2 = self.rule_table.columns[1]  # regex pattern column

        for c1, c2 in zip(self.rule_table[col1], self.rule_table[col2]):

            try:
                extracted_str = re.search(c2, x)[0]  # extracted result
                policytagid = c1  # its polycytag id
                regex_patt = c2

                return [policytagid, regex_patt]  # combined in a  list
            except Exception as e:
                with open('regexlog.txt', 'a') as f:
                    f.write(str(e))
                    f.write(traceback.format_exc())
                pass

    def fit(self, y=None, **kwargs) -> pd.Series:
        X = pd.Series(['model score', 'user id'])  ## this is just a place holder
        X = X.apply(lambda st: self.extract_from_regex_rule_table(X))  # calling extract_from_regex_rule_table
        X = X.fillna(0)
        # for testing result_type is'test

        if self.result_type == 'debug':
            return X
        # else predict only policitagid
        else:
            X.loc[X != 0] = X.loc[X != 0].apply(lambda x: x[0])
            X.loc[X.isnull()] = X.loc[X.isnull()].apply(lambda x: 0)
            return self

    def transform(self, X: pd.Series, y=None, **kwargs) -> pd.Series:  # X is the column_name

        X = X.apply(lambda st: self.extract_from_regex_rule_table(st))  # calling extract_from_regex_rule_table
        X = X.fillna(0)
        # for testing result_type is'test

        if self.result_type == 'debug':
            return X
        # else predict only policitagid
        else:
            X.loc[X != 0] = X.loc[X != 0].apply(lambda x: x[0])
            X.loc[X.isnull()] = X.loc[X.isnull()].apply(lambda x: 0)
            return X

#################################################
# Following code is from NaiveBayesPredict.py
#################################################
# the predictDataOutput will provide out put with all columns in input file
def predictDataOutput(ds_tobePrecdicted, model, feature_name, le_model, outrange, save_pred_output):
    """
    predictDataOutput will take the data and predict using the models that is trained with labelled data
    :param dataset:
    :return:
    """
    # =============================================================================
    # Prediction on Test Data
    # =============================================================================
    dataset = ds_tobePrecdicted.copy()
    dataset = dataset.reset_index(drop=True)
    logger.info(f"dataset to be predicted count()={dataset.count()}")

    probs = model.predict_proba(dataset[feature_name])
    probs_df = pd.DataFrame(probs, columns=le_model.classes_)
    logger.info(f"probs_df = {probs_df.head(4)}")
    classifier_id = '999'
    logger.info(f"Outrange value: {str(outrange)}")
    class_df = pd.DataFrame(le_model.classes_)
    class_df['seq'] = range(0, class_df.shape[0], 1)
    class_df.columns = ['label', 'seq']
    df_cols = class_df['label']

    tmp_dict = list()
    resp_end = dict()
    resp_end['classifier_id'] = classifier_id
    # num_rws = dataset_prob_df.shape[0]
    num_rws = probs_df.shape[0]
    for rw in range(num_rws):
        each_row = probs_df.iloc[rw]
        tmp_dict2 = dict()
        # prob_list = each_row
        probabilities = [(index + 0, item) for (index, item) in
                         enumerate(each_row)]
        index_labels = sorted(probabilities,
                              key=lambda x: x[1],
                              reverse=True)[:int(outrange)]
        tmp_dict2['prediction'] = df_cols[index_labels[0][0]]
        tmp_dict2['result'] = index_labels[0][1]
        tmp_dict2['probabilities'] = dict()
        for r in index_labels:
            tmp_dict2['probabilities'][df_cols[r[0]]] = r[1]
        tmp_dict.append(tmp_dict2)
    resp_end['predictions'] = tmp_dict
    df_preds = pd.DataFrame.from_dict(resp_end['predictions'])
    df_preds_final = df_preds.iloc[:, [2, 1, 0]]
    df_preds_final.columns = ['NB_RESULT', 'NB_CONFIDENCE', 'NB_TOP_CLASS']
    df_preds_final['NB_CONFIDENCE'] = pd.to_numeric(df_preds_final['NB_CONFIDENCE'], downcast='float')

    logger.info("df_preds_final details-->")
    logger.info(f"df_preds_final.columns-->{df_preds_final.columns}")
    logger.info(f"df_preds_final count-->{len(df_preds_final)}")
    logger.info(f"df_preds_final head-->{df_preds_final.head(3)}")
    ##combine the original dataset with predicted result
    df_preds_with_all_cols = pd.concat([dataset, df_preds_final], axis=1)

    logger.info("df_preds_with_all_cols details-->")
    logger.info(f"df_preds_with_all_cols.columns-->{df_preds_with_all_cols.columns}")
    logger.info(f"df_preds_with_all_cols count-->{len(df_preds_with_all_cols)}")
    logger.info(f"df_preds_with_all_cols head-->{df_preds_with_all_cols.head(3)}")
    logger.info(f"df_preds_with_all_cols count ={len(df_preds_with_all_cols)} ")
    # res = "./RESULTS"
    # tags = f"records_with_min_250_noMaxLimit_samples_STDZD_11_14_{split_num}"
    # if save_pred_output:
    #     fn = res + "//" + tags + "_" + '_classified.csv'
    #     df_preds_with_all_cols.to_csv(fn, index=None)
    return df_preds_with_all_cols

def predictDataOutputNN_SBERT(ds_tobePrecdicted, model, le_model, outrange, save_pred_output, sbert_columns_pred):
    try:
        dataset = ds_tobePrecdicted.copy()
        dataset = dataset.reset_index(drop=True)
        logger.info(f"dataset to be predicted count()={dataset.count()}")
        dataset_columns_list = dataset.columns.tolist()
        # dataset['embeddings'] = dataset.apply(lambda x: eval(x['embeddings']), axis=1)
        # dataset = dataset.embeddings.apply(pd.Series).merge(dataset, right_index=True,
        #                                                                   left_index=True)
        # predict_dataset_cols= [elem for elem in dataset.columns.tolist() if elem not in dataset_columns_list]
        probs = model.predict_proba(dataset[sbert_columns_pred])
        probs_df = pd.DataFrame(probs, columns=le_model.classes_)
        logger.info(f"probs_df = {probs_df.head(4)}")
        classifier_id = '999'
        logger.info(f"Outrange value: {str(outrange)}")
        class_df = pd.DataFrame(le_model.classes_)
        class_df['seq'] = range(0, class_df.shape[0], 1)
        class_df.columns = ['label', 'seq']
        df_cols = class_df['label']

        tmp_dict = list()
        resp_end = dict()
        resp_end['classifier_id'] = classifier_id
        # num_rws = dataset_prob_df.shape[0]
        num_rws = probs_df.shape[0]
        for rw in range(num_rws):
            each_row = probs_df.iloc[rw]
            tmp_dict2 = dict()
            # prob_list = each_row
            probabilities = [(index + 0, item) for (index, item) in
                             enumerate(each_row)]
            index_labels = sorted(probabilities,
                                  key=lambda x: x[1],
                                  reverse=True)[:int(outrange)]
            tmp_dict2['prediction'] = df_cols[index_labels[0][0]]
            tmp_dict2['result'] = index_labels[0][1]
            tmp_dict2['probabilities'] = dict()
            for r in index_labels:
                tmp_dict2['probabilities'][df_cols[r[0]]] = r[1]
            tmp_dict.append(tmp_dict2)
        resp_end['predictions'] = tmp_dict
        df_preds = pd.DataFrame.from_dict(resp_end['predictions'])
        df_preds_final = df_preds.iloc[:, [2, 1, 0]]
        df_preds_final.columns = ['KNN_SBERT_RESULT', 'KNN_SBERT_CONFIDENCE', 'KNN_SBERT_TOP_CLASS']
        df_preds_final['KNN_SBERT_CONFIDENCE'] = pd.to_numeric(df_preds_final['KNN_SBERT_CONFIDENCE'],
                                                               downcast='float')

        logger.info("df_preds_final details-->")
        logger.info(f"df_preds_final.columns-->{df_preds_final.columns}")
        logger.info(f"df_preds_final count-->{len(df_preds_final)}")
        logger.info(f"df_preds_final head-->{df_preds_final.head(3)}")
        ##combine the original dataset with predicted result
        df_preds_with_all_cols = pd.concat([dataset, df_preds_final], axis=1)
        final_columns_list = dataset_columns_list + ['KNN_SBERT_RESULT', 'KNN_SBERT_CONFIDENCE',
                                                     'KNN_SBERT_TOP_CLASS']

        # df_preds_with_all_cols_test = df_preds_with_all_cols.copy()
        # df_preds_with_all_cols_test['embeddings_after'] = df_preds_with_all_cols_test[sbert_columns_pred].values.tolist()
        # df_preds_with_all_cols_test['embeddings_check'] = df_preds_with_all_cols_test['embeddings_after'] == df_preds_with_all_cols_test['embeddings']
        # logger.info(f"Embeddings check Before and After Prediction : {df_preds_with_all_cols_test['embeddings_check'].unique()}-->")

        df_preds_with_all_cols = df_preds_with_all_cols[final_columns_list]
        logger.info("df_preds_with_all_cols details-->")
        logger.info(f"df_preds_with_all_cols.columns-->{df_preds_with_all_cols.columns}")
        logger.info(f"df_preds_with_all_cols count-->{len(df_preds_with_all_cols)}")
        logger.info(f"df_preds_with_all_cols head-->{df_preds_with_all_cols.head(3)}")
        logger.info(f"df_preds_with_all_cols count ={len(df_preds_with_all_cols)} ")
        res = "./RESULTS"
        if save_pred_output:
            tags = f"dataset_min250_max5000_12_02_KNN_SBERT"
            fn = res + "//" + tags + "_" + '_classified.csv'
            df_preds_with_all_cols.to_csv(fn, index=None)
    except Exception as e:
        logger.error(f'Error in Predicition using Nearest Neighbor SBERT : {e}')
        raise
    return df_preds_with_all_cols

def predictDataOutputDT_SBERT(ds_tobePrecdicted, model, outrange, save_pred_output, sbert_columns_pred):
    try:
        dataset = ds_tobePrecdicted.copy()
        dataset = dataset.reset_index(drop=True)
        logger.info(f"dataset to be predicted count()={dataset.count()}")
        dataset_columns_list = dataset.columns.tolist()
        dataset['DT_SBERT_TOP_CLASS'] = model.predict(dataset[sbert_columns_pred])

        final_columns_list = dataset_columns_list + [
            'DT_SBERT_TOP_CLASS']  # 'DT_SBERT_RESULT', 'DT_SBERT_CONFIDENCE',

        df_preds_with_all_cols = dataset.copy()
        df_preds_with_all_cols = df_preds_with_all_cols[final_columns_list]
        df_preds_with_all_cols['DT_SBERT_CONFIDENCE'] = 1
        logger.info("df_preds_with_all_cols details-->")
        logger.info(f"df_preds_with_all_cols.columns-->{df_preds_with_all_cols.columns}")
        logger.info(f"df_preds_with_all_cols count-->{len(df_preds_with_all_cols)}")
        logger.info(f"df_preds_with_all_cols head-->{df_preds_with_all_cols.head(3)}")
        logger.info(f"df_preds_with_all_cols count ={len(df_preds_with_all_cols)} ")
        res = "./RESULTS"
        if save_pred_output:
            tags = f"dataset_min250_max5000_12_02_NN_"
            fn = res + "//" + tags + "_" + '_classified.csv'
            df_preds_with_all_cols.to_csv(fn, index=None)
    except Exception as e:
        logger.error(f'Error in Predicition using Decision Tree SBERT: {e}')
        raise
    return df_preds_with_all_cols

def predictDataOutputDT(ds_tobePrecdicted, model, feature_name, outrange, save_pred_output):
    try:
        dataset = ds_tobePrecdicted.copy()
        dataset = dataset.reset_index(drop=True)
        logger.info(f"dataset to be predicted count()={dataset.count()}")

        dataset['DT_TOP_CLASS'] = model.predict(dataset[feature_name])

        df_preds_final = dataset.copy()

        df_preds_final['DT_CONFIDENCE'] = 1

        logger.info("df_preds_final details-->")
        logger.info(f"df_preds_final.columns-->{df_preds_final.columns}")
        logger.info(f"df_preds_final count-->{len(df_preds_final)}")
        logger.info(f"df_preds_final head-->{df_preds_final.head(3)}")
        df_preds_with_all_cols = df_preds_final.copy()
        logger.info("df_preds_with_all_cols details-->")
        logger.info(f"df_preds_with_all_cols.columns-->{df_preds_with_all_cols.columns}")
        logger.info(f"df_preds_with_all_cols count-->{len(df_preds_with_all_cols)}")
        logger.info(f"df_preds_with_all_cols head-->{df_preds_with_all_cols.head(3)}")
        logger.info(f"df_preds_with_all_cols count ={len(df_preds_with_all_cols)} ")
        res = "./RESULTS"
        if save_pred_output:
            tags = f"dataset_min250_max5000_12_02_NN_"
            fn = res + "//" + tags + "_" + '_classified.csv'
            df_preds_with_all_cols.to_csv(fn, index=None)
    except Exception as e:
        logger.error(f'Error in Predicition using Decision Tree : {e}')
        raise
    return df_preds_with_all_cols


    #############################################################

# Following code is from CCSearchEnginePredict_withNgrams.py
#############################################################
def getDocClassBasedOnScore_genrlzd(bm25, class_lst, row, feature_name):
    """
    This menthod will calculate score of question in each row
    :param bm25:
    :param class_lst:
    :param row:
    :return:
    """
    ## row has the entire row, extracting the 'Question' column with row[0]
    query = str(row[feature_name])
    len_tokens = len(query.split(" "))

    tokenized_query = []
    ## if only one token in the query string, not need of ngram split
    ## if it is a combined string like accountnumber, this need to be split as part of standadization
    if len_tokens == 1:
        tokenized_query.append(query)
    ## if >=ngram tokens, then split tokens upto ngram-1 and add them
    ## ngrams(query.split(), ngrm) returns a tuple of splits, need to extarct them using index
    ## add the tokens to a list -- example
    # row[0] is 'web number session', split into bigrams and trigrams as -->  +++++++++++ ['web number session', 'web number', 'number session']
    for ngrm in range(len_tokens, 1, -1):
        q_ngram = list(ngrams(query.split(), ngrm))
        for i in q_ngram:
            token_val = ""
            for idx in range(0, ngrm):
                token_val = token_val + " " + i[idx]
            # print("token_val =", token_val)
            tokenized_query.append(token_val.strip())

    # logger.info(f"+++++++++++{tokenized_query}")
    doc_scores = bm25.get_scores(tokenized_query)
    # logger.info(doc_scores)
    max_index = np.argmax(doc_scores)
    ##if query did not have a score, then include unigrams
    if doc_scores[max_index] == 0:
        # logger.info("doc_score is zero, adding unigrams")
        q_ngram = list(ngrams(query.split(), 1))
        for i in q_ngram:
            token_val = i[0]
            tokenized_query.append(token_val.strip())
        # logger.info(f"******** tokenized_query with unigrams as doc_score was zero without it = {tokenized_query}")
        doc_scores = bm25.get_scores(tokenized_query)
    # %%%% per tokn score to normalise the score for threshold comparison
    len_doc = len(tokenized_query)
    per_token_doc_scores = doc_scores / len_doc
    # logger.info("per_token_doc_scores ==>")
    # logger.info(per_token_doc_scores)
    max_index = np.argmax(per_token_doc_scores)
    # print(f"the best class ' {query}' belongs to is =", class_lst[max_index])
    sort_index = np.argsort(-1 * per_token_doc_scores)[:3]
    class_score = {}
    class_per_token_score = {}
    for i in sort_index:
        class_score[class_lst[i]] = doc_scores[i]
        class_per_token_score[class_lst[i]] = per_token_doc_scores[i]

    row['BM25_RESULT'] = class_lst[max_index]
    row['BM25_SCORES'] = class_score
    row['BM25_PER_TOKEN_SCORES'] = class_per_token_score
    row['BM25_TOP_CLASS'] = class_lst[max_index]
    row['BM25_CONFIDENCE'] = per_token_doc_scores[max_index]

    return row

    #############################################################

# Following code is from CCSearchEnginePredict_withNgrams.py
#############################################################
pd.options.display.float_format = '{:20,.2f}'.format
ngram = '1,2'
maxfeatures = '262144'
maxfeaturesnu = int(maxfeatures)
nvalpha = '.00000001'
logger.info("nvalpha: " + nvalpha)
logger.info(type(nvalpha))

# the predictDataOutput will provide out put with all columns in input file
def predictDataOutputNN(ds_tobePrecdicted, model, feature_name, le_model, outrange, save_pred_output):
    """
    predictDataOutput will take the data and predict using the models that is trained with labelled data
    :param dataset:
    :return:
    """
    # =============================================================================
    # Prediction on Test Data
    # =============================================================================
    dataset = ds_tobePrecdicted.copy()
    dataset = dataset.reset_index(drop=True)
    logger.info(f"dataset to be predicted count()={dataset.count()}")

    probs = model.predict_proba(dataset[feature_name])
    probs_df = pd.DataFrame(probs, columns=le_model.classes_)
    logger.info(f"probs_df = {probs_df.head(4)}")
    classifier_id = '999'
    logger.info(f"Outrange value: {str(outrange)}")
    class_df = pd.DataFrame(le_model.classes_)
    class_df['seq'] = range(0, class_df.shape[0], 1)
    class_df.columns = ['label', 'seq']
    df_cols = class_df['label']

    tmp_dict = list()
    resp_end = dict()
    resp_end['classifier_id'] = classifier_id
    # num_rws = dataset_prob_df.shape[0]
    num_rws = probs_df.shape[0]
    for rw in range(num_rws):
        each_row = probs_df.iloc[rw]
        tmp_dict2 = dict()
        # prob_list = each_row
        probabilities = [(index + 0, item) for (index, item) in
                         enumerate(each_row)]
        index_labels = sorted(probabilities,
                              key=lambda x: x[1],
                              reverse=True)[:int(outrange)]
        tmp_dict2['prediction'] = df_cols[index_labels[0][0]]
        tmp_dict2['result'] = index_labels[0][1]
        tmp_dict2['probabilities'] = dict()
        for r in index_labels:
            tmp_dict2['probabilities'][df_cols[r[0]]] = r[1]
        tmp_dict.append(tmp_dict2)
    resp_end['predictions'] = tmp_dict
    df_preds = pd.DataFrame.from_dict(resp_end['predictions'])
    df_preds_final = df_preds.iloc[:, [2, 1, 0]]
    df_preds_final.columns = ['NNBR_RESULT', 'NNBR_CONFIDENCE', 'NNBR_TOP_CLASS']
    df_preds_final['NNBR_CONFIDENCE'] = pd.to_numeric(df_preds_final['NNBR_CONFIDENCE'], downcast='float')

    logger.info("df_preds_final details-->")
    logger.info(f"df_preds_final.columns-->{df_preds_final.columns}")
    logger.info(f"df_preds_final count-->{len(df_preds_final)}")
    logger.info(f"df_preds_final head-->{df_preds_final.head(3)}")
    ##combine the original dataset with predicted result
    df_preds_with_all_cols = pd.concat([dataset, df_preds_final], axis=1)

    logger.info("df_preds_with_all_cols details-->")
    logger.info(f"df_preds_with_all_cols.columns-->{df_preds_with_all_cols.columns}")
    logger.info(f"df_preds_with_all_cols count-->{len(df_preds_with_all_cols)}")
    logger.info(f"df_preds_with_all_cols head-->{df_preds_with_all_cols.head(3)}")
    logger.info(f"df_preds_with_all_cols count ={len(df_preds_with_all_cols)} ")
    res = "./RESULTS"

    # if save_pred_output:
    #     fn = res + "//" + tags + "_" + '_classified.csv'
    #     df_preds_with_all_cols.to_csv(fn, index=None)
    return df_preds_with_all_cols

    ### From run.py

def matchCols(x, cols, required_perc_match):
    try:
        num_cols = len(cols)
        # required_match = (num_cols - 1) / num_cols
        is_match = False
        perc_match = 0
        for i in range(0, num_cols):
            matches = 0
            for j in range(0, num_cols):
                if x[i] == x[j]:
                    matches += 1
            perc_match = matches / num_cols
            if perc_match >= required_perc_match:
                is_match = True
                x['IS_MATCH'] = "Yes"
                x['SYSTEM_PREDICTION'] = x[i]
                x['SYSTEM_CONF'] = perc_match
                break

        if not is_match:
            x['IS_MATCH'] = "No"
            x['SYSTEM_PREDICTION'] = -1
            x['SYSTEM_CONF'] = perc_match
    except Exception as e:
        logger.error('Error in MatchCols function: ', e)
        raise
    return x

def round_val_dict(x):
    j = {x: np.round(y, decimals=2) for x, y in x.items()}
    return j

def json_to_df(input_json):
    try:
        columns_list = input_json['input_data'][0]['values'][0]
        column_name_list = input_json['input_data'][0]['values'][1:]
        column_name = [list[1] for list in column_name_list]
        columns_list.append(column_name)
        final_data_list = []
        final_data_list.append(columns_list)
        json_df = pd.DataFrame(final_data_list,
                               columns=['ProjectName', 'Domain', 'DatabaseName', 'TableName', 'DataClass(KDT)',
                                        'AssetDescription', 'columnName'])
        json_df = json_df.explode('columnName')
    except Exception as e:
        json_df = pd.DataFrame()
        logger.error('Error in converting json to dataframe: ', e)

    return json_df

#################################################################
# Added by WML Development: This function is required to make the prediction
# output into JSON compatianble format as the scoring response
# is a JSON data format.
#################################################################
def make_json_serizalizeable(input_iterable):
    inter_result = []
    for outer_data in input_iterable:
        if isinstance(outer_data, Iterable):
            for elem in outer_data:
                if isinstance(elem, dict):
                    for k, v in elem.copy().items():
                        if 'numpy' in str(type(k)):
                            elem.pop(k)
                            elem[k.item()] = v.item() if 'numpy' in str(type(v)) and not isinstance(v,
                                                                                                    Iterable) else v
        if isinstance(outer_data, np.ndarray):
            inter_result.append(outer_data.tolist())
        else:
            inter_result.append(outer_data)
    return inter_result

def correctQuoteJSON(s):
    rstr = ""
    escaped = False

    for c in s:

        if c == "'" and not escaped:
            c = '"'

        elif c == "'" and escaped:
            rstr = rstr[:-1]

        elif c == '"':
            c = '\\' + c

        escaped = (c == "\\")
        rstr += c

    return rstr

# Define scoring function
def score(input):
    import sys
    import os

    try:

        ####CODE for WKC
        json_string = input
        ####CODE for WKC

        ####CODE for LOCAL
        json_string = {'input_data': [{'values': [['Project Name', 'Domain', 'CREDIT_APP_V', 'CREDIT_APP_V', 'Data Class KDT', ''], ['CREDIT_APP_V', 'CREDIT_APP_NUM', 'column description'], ['CREDIT_APP_V', 'CREDIT_APP_TYPE', 'column description'], ['CREDIT_APP_V', 'ACCT_TYPE', 'column description'], ['CREDIT_APP_V', 'CREDIT_APP_STATUS', 'column description'], ['CREDIT_APP_V', 'PREV_IND', 'column description'], ['CREDIT_APP_V', 'CUST_NM', 'column description'], ['CREDIT_APP_V', 'CUST_ADDR_LINE1', 'column description'], ['CREDIT_APP_V', 'CUST_ADDR_LINE2', 'column description'], ['CREDIT_APP_V', 'CUST_ADDR_LINE3', 'column description'], ['CREDIT_APP_V', 'CUST_CITY_NM', 'column description'], ['CREDIT_APP_V', 'CUST_STATE_CD', 'column description'], ['CREDIT_APP_V', 'CUST_ZIP_CD', 'column description'], ['CREDIT_APP_V', 'CUST_CNTRY_CD', 'column description'], ['CREDIT_APP_V', 'CATS_MKT_CD', 'column description'], ['CREDIT_APP_V', 'ORDER_TYPE', 'column description'], ['CREDIT_APP_V', 'HOME_TEL_NUM', 'column description'], ['CREDIT_APP_V', 'TAX_ID', 'column description'], ['CREDIT_APP_V', 'CURR_USER_ID', 'column description'], ['CREDIT_APP_V', 'CREDIT_APP_DT', 'column description'], ['CREDIT_APP_V', 'CREDIT_APP_TM', 'column description'], ['CREDIT_APP_V', 'NUM_OF_PHONES', 'column description'], ['CREDIT_APP_V', 'AGENT_CD', 'column description'], ['CREDIT_APP_V', 'CATS_REGION_CD', 'column description'], ['CREDIT_APP_V', 'BILL_NM', 'column description'], ['CREDIT_APP_V', 'BILL_ADDR_LINE1', 'column description'], ['CREDIT_APP_V', 'BILL_ADDR_LINE2', 'column description'], ['CREDIT_APP_V', 'BILL_ADDR_LINE3', 'column description'], ['CREDIT_APP_V', 'BILL_CITY_NM', 'column description'], ['CREDIT_APP_V', 'BILL_STATE_CD', 'column description'], ['CREDIT_APP_V', 'BILL_ZIP_CD', 'column description'], ['CREDIT_APP_V', 'BILL_CNTRY_CD', 'column description'], ['CREDIT_APP_V', 'CURR_MTN', 'column description'], ['CREDIT_APP_V', 'CONV_MTN', 'column description'], ['CREDIT_APP_V', 'CUST_ID', 'column description'], ['CREDIT_APP_V', 'LOC_CD', 'column description'], ['CREDIT_APP_V', 'SLS_OUTLET_ID', 'column description'], ['CREDIT_APP_V', 'CATS_USER_NM', 'column description'], ['CREDIT_APP_V', 'BIRTH_DT', 'column description'], ['CREDIT_APP_V', 'LAST_UPD_DT', 'column description'], ['CREDIT_APP_V', 'CUST_NM_FIRST', 'column description'], ['CREDIT_APP_V', 'CUST_NM_MID_INIT', 'column description'], ['CREDIT_APP_V', 'SLS_PRSN_ID', 'column description'], ['CREDIT_APP_V', 'AREA_CD', 'column description'], ['CREDIT_APP_V', 'SSN_MASK_ID', 'column description'], ['CREDIT_APP_V', 'BIRTH_DT_MASK_ID', 'column description'], ['CREDIT_APP_V', 'BIRTH_MTH', 'column description'], ['CREDIT_APP_V', 'SHIP_TO_ADDR_LINE1', 'column description'], ['CREDIT_APP_V', 'SHIP_TO_ADDR_LINE2', 'column description'], ['CREDIT_APP_V', 'SHIP_TO_ADDR_LINE3', 'column description'], ['CREDIT_APP_V', 'SHIP_TO_CITY_NM', 'column description'], ['CREDIT_APP_V', 'SHIP_TO_STATE_CD', 'column description'], ['CREDIT_APP_V', 'SHIP_TO_ZIP_CD', 'column description'], ['CREDIT_APP_V', 'IP_ADDR', 'column description'], ['CREDIT_APP_V', 'EMAIL_ADDR', 'column description'], ['CREDIT_APP_V', 'TAX_ID_MASK_ID', 'column description'], ['CREDIT_APP_V', 'EDGE_INTERESTED_IND', 'column description'], ['CREDIT_APP_V', 'DPP_INTERESTED_IND', 'column description'], ['CREDIT_APP_V', 'SRC_SYS_CD', 'column description'], ['CREDIT_APP_V', 'ORIG_SLS_OUTLET_ID', 'column description'], ['CREDIT_APP_V', 'ORIG_SRC_SYS_CD', 'column description'], ['CREDIT_APP_V', 'ORIG_CREDIT_APP_NUM', 'column description'], ['CREDIT_APP_V', 'PREQUAL_IND', 'column description'], ['CREDIT_APP_V', 'UPG_TYP_IND', 'column description'], ['CREDIT_APP_V', 'INDIV_KEY', 'column description'], ['CREDIT_APP_V', 'HOUSEHOLD_KEY', 'column description'], ['CREDIT_APP_V', 'ADDR_KEY', 'column description'], ['CREDIT_APP_V', 'CNX_CONFIDENCE_CD', 'column description'], ['CREDIT_APP_V', 'BLENDED_IND', 'column description'], ['CREDIT_APP_V', 'BLENDED_NM_LAST', 'column description'], ['CREDIT_APP_V', 'BLENDED_NM_FIRST', 'column description'], ['CREDIT_APP_V', 'BLENDED_ADDR', 'column description'], ['CREDIT_APP_V', 'BLENDED_CITY', 'column description'], ['CREDIT_APP_V', 'BLENDED_STATE_CD', 'column description'], ['CREDIT_APP_V', 'BLENDED_ZIP_CD', 'column description'], ['CREDIT_APP_V', 'BLENDED_SIGN_TITLE', 'column description'], ['CREDIT_APP_V', 'BLENDED_SIGN_OWNERSHIP', 'column description'], ['CREDIT_APP_V', 'BLENDED_SIGN_EMAIL_ADDR', 'column description'], ['CREDIT_APP_V', 'BLENDED_SIGN_TEL_NUM', 'column description'], ['CREDIT_APP_V', 'BLENDED_SSN_MASK_ID', 'column description'], ['CREDIT_APP_V', 'BLENDED_BIRTH_DT_MASK_ID', 'column description'], ['CREDIT_APP_V', 'EFXID_KEY', 'column description'], ['CREDIT_APP_V', 'GLOBAL_KEY', 'column description'], ['CREDIT_APP_V', 'DOMESTIC_KEY', 'column description'], ['CREDIT_APP_V', 'PARENT_KEY', 'column description'], ['CREDIT_APP_V', 'EFXID_CONFIDENCE_CD', 'column description'], ['CREDIT_APP_V', 'LEGAL_ENTITY_KEY', 'column description'], ['CREDIT_APP_V', 'LEGAL_ENTITY_NM', 'column description'], ['CREDIT_APP_V', 'SSN_TOKEN_ID', 'column description'], ['CREDIT_APP_V', 'TAX_ID_TOKEN_ID', 'column description'], ['CREDIT_APP_V', 'BIRTH_DT_TOKEN_ID', 'column description'], ['CREDIT_APP_V', 'BLENDED_SSN_TOKEN_ID', 'column description'], ['CREDIT_APP_V', 'BLENDED_BIRTH_DT_TOKEN_ID', 'column description'], ['CREDIT_APP_V', 'BLENDED_BIRTH_MTH', 'column description'], ['CREDIT_APP_V', 'DECISION_PATH', 'column description'], ['CREDIT_APP_V', 'NO_CREDIT_CHECK_OPT_IND', 'column description'], ['CREDIT_APP_V', 'CCB_AUTHORIZED', 'column description']], 'fields': None}]}
        ####CODE for LOCAL

        logger.info(f"input = {json_string}")

        json_string = str(json_string)
        json_string_format = json_string.replace('None', 'null')
        json_string_new = correctQuoteJSON(json_string_format)
        logger.info(f"input after formatting = {json_string_new}")

        col_name = 'columnName'
        col_name_stdzd = 'COLUMN_NAME_STDZD'
        dest_folder = 'RESULTS'
        label = 'business_term'
        input_json = json.loads(json_string_new)

        #             if len(args) == 4:
        if True:
            # logger.info(f'file_name = {file_name}')
            logger.info(f'col_name = {col_name} ')
            logger.info(f'standardized col_name = {col_name_stdzd} ')
            logger.info(f'dest_folder = {dest_folder}')
        else:
            logger.info('Required args are not provided')
    except Exception as e:
        logger.error('Error with Arguments: ', e)
        raise

    try:
        ip_data_df = json_to_df(input_json)
        columnName_list = ip_data_df['columnName'].to_list()
        # ip_data_df.to_csv("data/ip_data_df.csv")

    except Exception as e:
        logger.error('Error with Converting Json to Dataframe: ', e)
        raise

    ### Start Standardization
    try:
        logger.info('Starting Standardization')
        ####CODE for WKC
        # latest_lookup = f'{data_dir}/{asset_details["data"]["Lookup_latest"]["file_name"]}'
        ####CODE for WKC
        ####CODE for LOCAL
        latest_lookup = f'data/wkc_depl_files/Latest_lookup_v18_2.csv'
        ####CODE for LOCAL
        df = standardization_main(ip_data_df, col_name, col_name_stdzd, label, latest_lookup)
        logger.info('Successfully Standardized')
    except Exception as e:
        logger.error('Error in Standardization: ', e)
        raise

    ### Start Regex processing
    try:
        ####CODE for WKC
        # rule_table_filename = f'{data_dir}/{asset_details["data"]["regex_rules"]["file_name"]}'
        ####CODE for WKC
        ####CODE for LOCAL
        rule_table_filename = f'data/wkc_depl_files/regex_rules_v11.csv'
        ####CODE for LOCAL

        rule_table = pd.read_csv(rule_table_filename)
        # loading the model
        logger.info('Starting Regex Model')
        model = RegExModelTransformer(rule_table)

        df['prediction'] = model.transform(df[col_name_stdzd])
        logger.info('Successfully Classified with Regex')
        df.rename(columns={'prediction': 'RegexPrediction'}, inplace=True)
        # df_without_regex = df.copy()
    except Exception as e:
        logger.error('Error in Regex: ', e)
        raise

    try:
        df_without_regex = df[df['RegexPrediction'] == 0]
        df_with_regex = df[df['RegexPrediction'] != 0]
        # file_name_to_save = f'{dest_folder}/Regex_Output.csv'
        # logger.info(f"Saving file = {file_name_to_save}")
        # df_with_regex.to_csv(file_name_to_save, index=None)
    except Exception as e:
        logger.error('Error in Regex Post Processing: ', e)
        raise

    try:
        feature_name = col_name_stdzd
        probability_range = 3
        save_pred_res = False
        model_dir = "models"
        bayes_model_nm = 'Bayes.pkl'
        bayes_labels = 'Bayes_Labels.pkl'
        bm25_model_nm = 'BM25.pkl'
        bm25_labels = 'BM25_Labels.pkl'
        nn1_labels = 'NearNghbr_Labels.pkl'
        nn1_model_nm = 'NearNghbr.pkl'
    #-------Future Release-----------------------------
        # dt_sbert_model = 'DT_SBERT.pkl'
        # dt_sbert_labels = 'DT_SBERT_labels.pkl'
        # dt_model = 'DT.pkl'
        # dt_labels = 'DT_labels.pkl'
        # nn_sbert_labels = 'NearNghbr_SBERT_Labels.pkl'
        # nn_sbert_model = 'NearNghbr_SBERT.pkl'
    # -------Future Release-----------------------------
        # required_perc_match = 0.66
        required_perc_match = 3/5

        logger.info(
            f"full_dataset count = {len(df_without_regex)}")
        logger.info(
            f"full_dataset[full_dataset[feature_name].isna()] = {len(df_without_regex[df_without_regex[feature_name].isna()])}")
        df_without_regex = df_without_regex.dropna(subset=[feature_name])
        logger.info(
            f"full_dataset count after removing NA= {len(df_without_regex)}")
    except Exception as e:
        logger.error('Error in PreProcess Classifier: ', e)
        raise
    #-------Future Release-----------------------------
    try:
        dataset_columns_list = df_without_regex.columns.tolist()
        model_name = "deberta-v2-xlarge"
        embedding_model = SentenceTransformer(f'{model_dir}/{model_name}')
        df_without_regex['embeddings'] = df_without_regex[col_name_stdzd].apply(lambda x: embedding_model.encode(x).tolist())
        print(df_without_regex['embeddings'].iloc[0])

        def list_split(df):
            df_train_sbert = df.copy()
            # df_train_sbert['embeddings'] = df_train_sbert.apply(lambda x: eval(x['embeddings']), axis=1)
            df_train_sbert = df_train_sbert.embeddings.apply(pd.Series).merge(df_train_sbert, right_index=True,
                                                                              left_index=True)
            return df_train_sbert

        print(df_without_regex.shape)
        df_without_regex = list_split(df_without_regex)
        df_without_regex.drop(columns=['embeddings'], inplace=True)
        # df_without_regex_sbert.drop(columns=['embeddings'], inplace=True)
        sbert_columns_pred = [elem for elem in df_without_regex.columns.tolist() if elem not in dataset_columns_list]
        print(len(sbert_columns_pred))
        # df_without_regex["embeddings"][i] = embedding_model.encode(df_without_regex[feature_name][i])
    except Exception as e:
        logger.error('Error in Creating Embeddings: ', e)
        raise

        ## Started - NaiveBayes prediction
    try:
        logger.info(f"Started - NaiveBayes prediction**")

        #         model_dir_path_nb = model_dir + '/' + bayes_model_nm
        #         logger.info("loading Bayes model **")
        #         trainingNBModel = bayes_model
        #         labelFile_nb = model_dir + '/' + bayes_labels
        #         label_nb = bayes_label
        #         logger.info(f"Loaded label {labelFile_nb} and model {model_dir_path_nb}")

        df_test_predicted = predictDataOutput(df_without_regex, trainingNBModel, feature_name, label_nb,
                                              probability_range,
                                              save_pred_res)
        df_test_predicted['NB_TOP_CLASS'] = df_test_predicted.apply(
            lambda x: -1 if x['NB_CONFIDENCE'] < 0.95 else x['NB_TOP_CLASS'], axis=1)

        logger.info(f"Ended - NaiveBayes prediction**")
    except Exception as e:
        logger.error('Error in NaiveBayes Classifier: ', e)
        raise

        # -------Future Release-----------------------------

    try:
        logger.info(f"Started - Decision Tree SBERT prediction**")

        # model_dir_path_dt_sbert = model_dir + '/' + dt_sbert_model
        # logger.info("loading DT SBERT model **")
        # trainingDTSBERTModel = pickle.load(open(model_dir_path_dt_sbert, 'rb'))

        # labelFile_dt_sbert = model_dir + '/' + dt_sbert_labels
        # label_dt_sbert = pickle.load(open(labelFile_dt_sbert, 'rb'))
        # logger.info(f"Loaded label {labelFile_dt_sbert} and model {model_dir_path_dt_sbert}")

        df_test_predicted = predictDataOutputDT_SBERT(df_test_predicted, trainingDTSBERTModel, probability_range, save_pred_res,
                                                      sbert_columns_pred)
        df_test_predicted['DT_SBERT_TOP_CLASS'] = df_test_predicted.apply(
            lambda x: -1 if x['DT_SBERT_CONFIDENCE'] < 0.95 else x['DT_SBERT_TOP_CLASS'], axis=1)
        logger.info(f"Ended - Decision Tree SBERT prediction**")
    except Exception as e:
        logger.error('Error in Decision Tree SBERT Classifier: ', e)
        raise

    try:
        logger.info(f"Started - KNN SBERT prediction**")

        # model_dir_path_knn_sbert = model_dir + '/' + nn_sbert_model
        # logger.info("loading KNN SBERT model **")
        # trainingKNNSBERTModel = pickle.load(open(model_dir_path_knn_sbert, 'rb'))
        #
        # labelFile_knn_sbert = model_dir + '/' + nn_sbert_labels
        # label_knn_sbert = pickle.load(open(labelFile_knn_sbert, 'rb'))
        # logger.info(f"Loaded label {labelFile_knn_sbert} and model {model_dir_path_knn_sbert}")
        #

        df_test_predicted = predictDataOutputNN_SBERT(df_test_predicted, trainingKNNSBERTModel, label_knn_sbert,
                                                      probability_range, save_pred_res, sbert_columns_pred)
        df_test_predicted['KNN_SBERT_TOP_CLASS'] = df_test_predicted.apply(
            lambda x: -1 if x['KNN_SBERT_CONFIDENCE'] < 0.95 else x['KNN_SBERT_TOP_CLASS'], axis=1)
        logger.info(f"Ended - KNN SBERT prediction**")
    except Exception as e:
        logger.error('Error in KNN SBERT Classifier: ', e)
        raise
    # -------Future Release-----------------------------

    try:
        logger.info(f"Started - Decision Tree TFIDF prediction**")
        #
        # model_dir_path_dt = model_dir + '/' + dt_model
        # logger.info("loading DT model **")
        # trainingDTModel = pickle.load(open(model_dir_path_dt, 'rb'))

        # labelFile_dt = model_dir + '/' + dt_labels
        # label_dt = pickle.load(open(labelFile_dt, 'rb'))
        # logger.info(f"Loaded label {labelFile_dt} and model {model_dir_path_dt}")


        df_test_predicted = predictDataOutputDT(df_test_predicted, trainingDTModel, feature_name, probability_range,
                                                save_pred_res)
        df_test_predicted['DT_TOP_CLASS'] = df_test_predicted.apply(
            lambda x: -1 if x['DT_CONFIDENCE'] < 0.95 else x['DT_TOP_CLASS'], axis=1)
        logger.info(f"Ended - Decision Tree TFIDF prediction**")
    except Exception as e:
        logger.error('Error in Decision Tree TFIDF Classifier: ', e)
        raise

    ## Starting - NearestNeighbor
    try:
        logger.info(f"Starting - NearestNeighbor**")

        #             model_dir_path_NN = model_dir + '/' + nn1_model_nm
        #             NN1_Model = knn_model
        #             label_file_1nn = model_dir + '/' + nn1_labels
        #             label_1nn = knn_label
        #             logger.info(f"Loaded label {label_file_1nn} and model {model_dir_path_NN}")

        df_test_predicted = predictDataOutputNN(df_test_predicted, NN1_Model, feature_name, label_1nn,
                                                probability_range,
                                                save_pred_res)

        df_test_predicted['NNBR_TOP_CLASS'] = df_test_predicted.apply(
            lambda x: -1 if x['NNBR_CONFIDENCE'] < 0.95 else x['NNBR_TOP_CLASS'], axis=1)

        logger.info(f"NN classified = {df_test_predicted.head(3)}")

        logger.info(f"Ended - NearestNeighbor**")
    except Exception as e:
        logger.error('Error in Nearest Neighbors Classifier: ', e)
        raise

        ## Starting - Integration
    try:
        logger.info(f"Starting - Integration**")
        # cols = ['NB_TOP_CLASS', 'NNBR_TOP_CLASS', 'DT_TOP_CLASS']
        cols = ['NB_TOP_CLASS', 'NNBR_TOP_CLASS', 'DT_TOP_CLASS', 'DT_SBERT_TOP_CLASS', 'KNN_SBERT_TOP_CLASS']
        logger.info(f"cols ={cols}")

        df_test_predicted_system = df_test_predicted.apply(lambda x: matchCols(x[cols], cols, required_perc_match),
                                                           axis=1)
        df_test_predicted = pd.concat([df_test_predicted, df_test_predicted_system], axis=1)
        df_test_predicted = df_test_predicted.loc[:, ~df_test_predicted.columns.duplicated()].copy()
    except Exception as e:
        logger.error('Error in applying matchcols Classifier: ', e)
        raise

    try:
        df_test_predicted.drop(columns=sbert_columns_pred, inplace=True)
        score_cols = ['NNBR_RESULT', 'NB_RESULT', 'KNN_SBERT_RESULT']
        df_test_predicted[score_cols] = df_test_predicted[score_cols].applymap(round_val_dict)
        confidence_cols = ['NB_CONFIDENCE', 'NNBR_CONFIDENCE', 'KNN_SBERT_CONFIDENCE', 'DT_CONFIDENCE', 'DT_SBERT_CONFIDENCE', 'SYSTEM_CONF']
        # df_test_predicted[confidence_cols] = df_test_predicted[confidence_cols].astype('float')
        df_test_predicted[confidence_cols] = np.round(df_test_predicted[confidence_cols], decimals=2)

        if df_with_regex.empty:
            logger.info("*** df_with_regex is empty")
            df_final_predicted = df_test_predicted
        else:
            new_cols = df_test_predicted.columns.difference(df_with_regex.columns).to_list()
            for col in new_cols:
                df_with_regex[new_cols] = ''

            df_final_predicted = df_test_predicted.append(df_with_regex)
        df_final_predicted['SYSTEM_PREDICTION'] = np.where(df_final_predicted['RegexPrediction'] != 0,
                                                           df_final_predicted['RegexPrediction'],
                                                           df_final_predicted['SYSTEM_PREDICTION'])
        # df_final_predicted = df_test_predicted.copy()
        #
        # df_final_predicted['SYSTEM_PREDICTION'] = np.where(
        #     (df_final_predicted['RegexPrediction'] != 0) & (df_final_predicted['SYSTEM_PREDICTION'] == -1),
        #     df_final_predicted['RegexPrediction'],
        #     df_final_predicted['SYSTEM_PREDICTION'])
        df_final_predicted['SYSTEM_CONF'] = np.where(df_final_predicted['RegexPrediction'] != 0, 1.0,
                                                     df_final_predicted['SYSTEM_CONF'])
        df_final_predicted.rename(columns={'NB_RESULT': 'BAYES_RESULT', 'NB_CONFIDENCE': 'BAYES_CONFIDENCE',
                                           'NB_TOP_CLASS': 'BAYES_TOP_CLASS', 'NNBR_RESULT': 'KNN_RESULT',
                                           'NNBR_CONFIDENCE': 'KNN_CONFIDENCE', 'NNBR_TOP_CLASS': 'KNN_TOP_CLASS'},
                                  inplace=True)
        #             fn = f"{dest_folder}/Final_File_With_Integrated_Classifications_{run_date}.csv"
        #             df_final_predicted.to_csv(fn, index=None)
        df_mapping = pd.DataFrame({
            'columnName_new': columnName_list, })
        sort_mapping = df_mapping.reset_index().set_index('columnName_new')
        df_final_predicted['input_columnName_sort'] = df_final_predicted['columnName'].map(sort_mapping['index'])
        df_final_predicted.sort_values('input_columnName_sort', inplace=True)
        df_final_predicted = df_final_predicted.reset_index(drop=True)
        logger.info(f"Final Prediction Dataframe = {df_final_predicted.to_dict()}")
        # df_final_predicted.to_csv('data/df_final_predicted_test.csv')
        final_pred_df = df_final_predicted[['SYSTEM_PREDICTION', 'SYSTEM_CONF']]
        #### CODE for WKC
        # guid_mapping_file = f'{data_dir}/{asset_details["data"]["BT_GUID_MAPPING"]["file_name"]}'
        # guid_mapping_file = pd.read_csv(guid_mapping_file)
        #### CODE for WKC

        ####CODE for LOCAL
        guid_mapping_file = pd.read_csv('data/wkc_depl_files/BT_GUID_MAPPING_DEV.csv')
        ####CODE for LOCAL

        final_pred_df.rename(columns={'SYSTEM_PREDICTION': 'gcaf_business terms'}, inplace=True)
        final_pred_guid = pd.merge(final_pred_df, guid_mapping_file, how='left', on=['gcaf_business terms'])
        final_pred_guid['SYSTEM_PREDICTION'] = np.where(final_pred_guid['GUID'].notnull(),
                                                        final_pred_guid['GUID'],
                                                        '')

        final_pred_guid['SYSTEM_CONF'] = np.where(final_pred_guid['SYSTEM_PREDICTION'] == '',
                                                  '',
                                                  final_pred_guid['SYSTEM_CONF'])
        print("final_pred_guid is saved as csv")
        final_pred_guid.to_csv('data/final_pred_guid.csv')

        system_pred_list = final_pred_guid['SYSTEM_PREDICTION'].values.tolist()
        system_conf_list = final_pred_guid['SYSTEM_CONF'].values.tolist()
        system_pred_list = [[el] for el in system_pred_list]
        system_conf_list = [[el] for el in system_conf_list]
        final_json_string = '''{
          "predictions": [
            {
              "fields": [
                "term_ids",
                "confidences"
              ],
              "values": [
                [ 
                    [],
                 []
                ]              
              ]
            }
          ]
        }'''
        final_json = json.loads(final_json_string)

        for i in range(len(system_pred_list)):
            value1 = []
            if system_pred_list[i][0] == '':
                system_pred_list[i] = []
            if system_conf_list[i][0] == '':
                system_conf_list[i] = []
            value1.append(system_pred_list[i])
            value1.append(system_conf_list[i])
            final_json['predictions'][0]['values'].append(value1)

        logger.info(f"Final Prediction Json = {final_json}")

        logger.info('Output = **********************')
        logger.info(df_test_predicted.head())
        logger.info(f"Ended - Integration**")
        logger.info("Ending SystemOfClassifiers_PredictClassification *******")
    except Exception as e:
        logger.error('Error in Post Classifier: ', e)
        raise

        # Make output data JSON serializeable
    #         result = make_json_serizalizeable(df_final_predicted.values)

    #         v4_scoring_response = {
    #             'predictions': [{'fields': df_final_predicted.columns.tolist(),
    #                              'values': result
    #                             }]
    #         }

    v4_scoring_response = final_json

    return v4_scoring_response



# v4_scoring_response = score('data/input_json_DEV_03072023_updated_json_withQuotes.txt')
v4_scoring_response = score('data/input_json_DEV_03072023.txt')

print("******************* v4_scoring_response")
print(v4_scoring_response)
print("******************* v4_scoring_response")