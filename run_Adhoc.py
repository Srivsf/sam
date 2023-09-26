import sys
import datetime
import os
import json
import logging
import pickle
import pandas as pd
import numpy as np
from Standardization.stdz_data import standardization_main
from RegexClassifier.regexmodeltransformer import RegExModelTransformer
from NaiveBayes.NaiveBayesPredict import predictDataOutput
from SearchEngineBM25.CCSearchEnginePredict_withNgrams import getDocClassBasedOnScore_genrlzd
from NNAlgorithm.NearestNeighborPredict import predictDataOutputNN



logging.basicConfig(filename="logs/run_cognitive.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
# Set the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

def matchCols(x,cols, required_perc_match):
    try:
        num_cols = len(cols)
        # required_match = (num_cols - 1) / num_cols
        is_match = False
        perc_match = 0
        for i in range(0,num_cols):
            matches = 0
            for j in range(0,num_cols):
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

# def read_data_json():
#     while True:
#         line = sys.stdin.readline()
#         if not line:
#             break
#         try:
#             record = json.loads(line)
#         except Exception as e:
#             logger.error('Json Decode Error: ', e)
#             raise
#     return record

def json_to_df(input_json):
    try:
        columns_list = input_json['input_data'][0]['values'][0]
        column_name_list = input_json['input_data'][0]['values'][1:]
        column_name = [list[1] for list in column_name_list]
        columns_list.append(column_name)
        final_data_list = []
        final_data_list.append(columns_list)
        json_df = pd.DataFrame(final_data_list, columns=['ProjectName', 'Domain','DatabaseName', 'TableName', 'DataClass(KDT)', 'AssetDescription', 'columnName'])
        json_df = json_df.explode('columnName')
    except Exception as e:
        json_df = pd.DataFrame()
        logger.error('Error in converting json to dataframe: ', e)

    return json_df


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



if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        # json_string = args[0]
        # json_string = open('input_json_DV_0306.json')
        json_string = "{'input_data': [{'values': [['Project Name', 'Domain', 'CUST_ACCT_V', 'CUST_ACCT_V', 'Data Class KDT', ''], ['CUST_ACCT_V', 'SOR_ID', 'column description'], ['CUST_ACCT_V', 'CUST_ID', 'column description'], ['CUST_ACCT_V', 'ACCT_NUM', 'column description'], ['CUST_ACCT_V', 'MKT_CD', 'column description'], ['CUST_ACCT_V', 'BUS_NM', 'column description'], ['CUST_ACCT_V', 'NM_PRFX', 'column description'], ['CUST_ACCT_V', 'NM_MDLIN', 'column description'], ['CUST_ACCT_V', 'NM_FIRST', 'column description'], ['CUST_ACCT_V', 'NM_LAST', 'column description'], ['CUST_ACCT_V', 'CNTCT_NM', 'column description'], ['CUST_ACCT_V', 'BUS_TEL_NUM', 'column description'], ['CUST_ACCT_V', 'HOME_TEL_NUM', 'column description'], ['CUST_ACCT_V', 'LINE_IN_SVC_CNT', 'column description'], ['CUST_ACCT_V', 'ACCT_STATUS_IND', 'column description'], ['CUST_ACCT_V', 'ACCT_ESTB_DT', 'column description'], ['CUST_ACCT_V', 'ACCT_TERM_DT', 'column description'], ['CUST_ACCT_V', 'CREDIT_CLASS_IND', 'column description'], ['CUST_ACCT_V', 'CREDIT_CARD_TYPE', 'column description'], ['CUST_ACCT_V', 'CREDIT_SCORE', 'column description'], ['CUST_ACCT_V', 'CREDIT_CLASS_CATS', 'column description'], ['CUST_ACCT_V', 'CREDIT_CARD_FLAG', 'column description'], ['CUST_ACCT_V', 'CREDIT_UPD_DT', 'column description'], ['CUST_ACCT_V', 'EXP_FLAG', 'column description'], ['CUST_ACCT_V', 'BILL_CYCLE_IND', 'column description'], ['CUST_ACCT_V', 'FINAL_BILL_DT', 'column description'], ['CUST_ACCT_V', 'SSN_EIN_ID', 'column description'], ['CUST_ACCT_V', 'INTERNET_BILL_PRES_ENROLL_CD', 'column description'], ['CUST_ACCT_V', 'SIC_CD', 'column description'], ['CUST_ACCT_V', 'HIGH_VALUE_DT', 'column description'], ['CUST_ACCT_V', 'HIGH_VALUE_IND', 'column description'], ['CUST_ACCT_V', 'DUNS_NUM', 'column description'], ['CUST_ACCT_V', 'PYMNT_SCORE_IND', 'column description'], ['CUST_ACCT_V', 'SINGL_BILL_IND', 'column description'], ['CUST_ACCT_V', 'SINGL_BILL_ENROLL_DT', 'column description'], ['CUST_ACCT_V', 'SINGL_BILL_TERM_DT', 'column description'], ['CUST_ACCT_V', 'EMAIL_ADDR', 'column description'], ['CUST_ACCT_V', 'COLL_STATUS_IND', 'column description'], ['CUST_ACCT_V', 'CUST_CLASS_CD', 'column description'], ['CUST_ACCT_V', 'LAST_UPD_DT', 'column description'], ['CUST_ACCT_V', 'WO_REAS_CD', 'column description'], ['CUST_ACCT_V', 'WO_AMT', 'column description'], ['CUST_ACCT_V', 'WO_DT', 'column description'], ['CUST_ACCT_V', 'CONV_ACCT_STATUS_IND', 'column description'], ['CUST_ACCT_V', 'CONV_ACCT_TERM_DT', 'column description'], ['CUST_ACCT_V', 'SINGL_BILL_BTN', 'column description'], ['CUST_ACCT_V', 'BILL_FORMAT_CD', 'column description'], ['CUST_ACCT_V', 'GEO_CD', 'column description'], ['CUST_ACCT_V', 'GEO_IND', 'column description'], ['CUST_ACCT_V', 'CR_CREDIT_APP_NUM', 'column description'], ['CUST_ACCT_V', 'BUDGET_CTR_IND', 'column description'], ['CUST_ACCT_V', 'DUNS_LOC_NUM', 'column description'], ['CUST_ACCT_V', 'DUNS_TYPE_CD', 'column description'], ['CUST_ACCT_V', 'DUNS_CONF_IND', 'column description'], ['CUST_ACCT_V', 'DUNS_VZ_OWNER_CD', 'column description'], ['CUST_ACCT_V', 'DUNS_VZ_LIAB_CD', 'column description'], ['CUST_ACCT_V', 'LANG_PREF_IND', 'column description'], ['CUST_ACCT_V', 'HIGH_VALUE_SEG_IND', 'column description'], ['CUST_ACCT_V', 'MAJ_MKT_ID', 'column description'], ['CUST_ACCT_V', 'RLTD_ACCT_ID', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_ACCT_ESTB_DT', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_ACCT_STATUS_IND', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_ACCT_TERM_DT', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_BILL_CYCLE_IND', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_BILL_FORMAT_CD', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_BUS_NM', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_CNTCT_NM', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_CREDIT_CARD_FLAG', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_CREDIT_CARD_TYPE', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_EMAIL_ADDR', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_HOME_TEL_NUM', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_IBP_ENRL_CD', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_LANG_PREF_IND', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_MKT_CD', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_NM_FIRST', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_NM_LAST', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_NM_MDLIN', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_SSN_EIN_ID', 'column description'], ['CUST_ACCT_V', 'XMKT_ACCT_IND', 'column description'], ['CUST_ACCT_V', 'XMKT_ACCT_NUM', 'column description'], ['CUST_ACCT_V', 'XMKT_INIT_ACCT_IND', 'column description'], ['CUST_ACCT_V', 'XMKT_LAST_UPD_DT', 'column description'], ['CUST_ACCT_V', 'INTER_CO_CD', 'column description'], ['CUST_ACCT_V', 'COST_CTR_CD', 'column description'], ['CUST_ACCT_V', 'XMKT_EFF_DT', 'column description'], ['CUST_ACCT_V', 'SINGL_BILL_CHANGE_DT', 'column description'], ['CUST_ACCT_V', 'LATE_FEE_SPRSS_IND', 'column description'], ['CUST_ACCT_V', 'IBP_TYPE_CD', 'column description'], ['CUST_ACCT_V', 'IBP_ENROLL_DT', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_IBP_TYPE_CD', 'column description'], ['CUST_ACCT_V', 'SUB_ACCT_IBP_ENRL_DT', 'column description'], ['CUST_ACCT_V', 'CUST_TYPE_CD', 'column description'], ['CUST_ACCT_V', 'PYMNT_ENROLL_VENUE_CD', 'column description'], ['CUST_ACCT_V', 'PYMNT_ENROLL_DT', 'column description'], ['CUST_ACCT_V', 'BILL_METHOD', 'column description'], ['CUST_ACCT_V', 'AFFIL_CD', 'column description'], ['CUST_ACCT_V', 'BILL_IN_ARREAR_IND', 'column description'], ['CUST_ACCT_V', 'REG_DT', 'column description'], ['CUST_ACCT_V', 'INSTANCE_IND', 'column description'], ['CUST_ACCT_V', 'HOME_AREA_CD', 'column description'], ['CUST_ACCT_V', 'VSN_CUST_TYPE_CD', 'column description'], ['CUST_ACCT_V', 'SINGL_BL_COMPANY_CD', 'column description'], ['CUST_ACCT_V', 'SINGL_BL_TERM_REAS_CD', 'column description'], ['CUST_ACCT_V', 'CUST_ASSOC_ID', 'column description'], ['CUST_ACCT_V', 'ALLTEL_SRC_BILL_ACCT_ID', 'column description'], ['CUST_ACCT_V', 'BILL_PRSNT_METH_CD', 'column description'], ['CUST_ACCT_V', 'MOB_CBR_NUM', 'column description'], ['CUST_ACCT_V', 'ECPD_QUAL_EVNT_CD', 'column description'], ['CUST_ACCT_V', 'BASE_BONUS_IND', 'column description'], ['CUST_ACCT_V', 'BASE_BONUS_ENROLL_DT', 'column description'], ['CUST_ACCT_V', 'EMAIL_ADDR_VALID_DT', 'column description'], ['CUST_ACCT_V', 'M2M_BILL_FORMAT_CD', 'column description'], ['CUST_ACCT_V', 'ORIG_ACCT_ESTB_DT', 'column description'], ['CUST_ACCT_V', 'INSERT_DT', 'column description'], ['CUST_ACCT_V', 'CUST_SSN_ESTB_DT', 'column description'], ['CUST_ACCT_V', 'MEDIA_TYPE_CD', 'column description'], ['CUST_ACCT_V', 'NON_MDN_LOS_CNT', 'column description'], ['CUST_ACCT_V', 'GEO_OVRIDE_CD', 'column description']], 'fields': None}]}"
        json_string_formt = json_string.replace('None', 'null')
        json_string_new = correctQuoteJSON(json_string_formt)
        logger.info(f'input = {json_string}')
#         json_string = '''{
# 	"input_data": [
# 		["vz-it-pr-gk1v-cwlspr-0", "", "vzw_uda_prd_tbls", "cjcm_vlead_case", "case_tm", "", "", "", "", "", ""],
# 		["vz-it-pr-gk1v-cwlspr-0", "", "vzw_uda_prd_tbls", "cjcm_vlead_case", "case_tm_est", "", "", "", "", "", ""],
# 		["vz-it-pr-gk1v-cwlspr-0", "", "vzw_uda_prd_tbls", "cjcm_vlead_case", "case_upd_ts", "", "", "", "", "", ""],
# 		["vz-it-pr-gk1v-cwlspr-0", "", "vzw_uda_prd_tbls", "cjcm_vlead_case", "case_upd_ts_est", "", "", "", "", "", ""],
# 		["vz-it-pr-gk1v-cwlspr-0", "", "vzw_uda_prd_tbls", "cjcm_vlead_case", "chnl_desc", "", "", "", "", "", ""]
# 	]
# }'''

        col_name = 'columnName' #args[1] #'columnName'
        col_name_stdzd = 'col_name_stdzd' #args[2] #'col_name_stdzd'
        dest_folder = 'RESULTS'# args[3] #'RESULTS'
        label = 'business_term'
        # input_json = json.loads(json_string)
        input_json = json.loads(json_string_new)


        if len(args) == 1:
            logger.info(f'input_json = {input_json}')
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
    except Exception as e:
        logger.error('Error with Converting Json to Dataframe: ', e)
        raise

    try:
        date_today = datetime.datetime.now()
        run_date = f'{date_today.month}_{date_today.day}_{date_today.year}'
    except Exception as e:
        logger.error('Error in extracting run date ', e)


    try:
        logger.info('Starting Standardization')
        # ip_data_df = pd.DataFrame(input_json['input_data'], columns=['ProjectName', 'Domain', 'DatabaseName', 'TableName', 'columnName',
        #                                                              'other_columns_in_table(pipe separated)', 'other_columns_In_database',
        #                                                              'Connection Name', 'data class (using KDT)', 'asset description','column description'])
        df = standardization_main(ip_data_df, col_name, col_name_stdzd, label)
        logger.info('Successfully Standardized')
    except Exception as e:
        logger.error('Error in Standardization: ', e)
        raise

    try:
        rule_table_filename = 'data/regex_rules_v6.csv'
        rule_table = pd.read_csv(rule_table_filename)
        # loading the model
        logger.info('Starting Regex Model')
        model = RegExModelTransformer(rule_table)

        df['prediction'] = model.transform(df[col_name_stdzd])
        logger.info('Successfully Classified with Regex')
        df.rename(columns={'prediction': 'RegexPrediction'}, inplace=True)
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
        # required_perc_match = 0.66
        required_perc_match = 1

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

    try:
        logger.info(f"Started - NaiveBayes prediction**")

        model_dir_path_nb = model_dir + '/' + bayes_model_nm
        logger.info("loading Bayes model **")
        trainingNBModel = pickle.load(open(model_dir_path_nb, 'rb'))

        labelFile_nb = model_dir + '/' + bayes_labels
        label_nb = pickle.load(open(labelFile_nb, 'rb'))
        logger.info(f"Loaded label {labelFile_nb} and model {model_dir_path_nb}")



        df_test_predicted = predictDataOutput(df_without_regex, trainingNBModel, feature_name, label_nb, probability_range,
                                              save_pred_res)
        df_test_predicted['NB_TOP_CLASS'] = df_test_predicted.apply(
            lambda x: -1 if x['NB_CONFIDENCE'] < 0.95 else x['NB_TOP_CLASS'], axis=1)

        logger.info(f"Ended - NaiveBayes prediction**")
    except Exception as e:
        logger.error('Error in NaiveBayes Classifier: ', e)
        raise

    try:
        logger.info(f"Starting - SearchBased BM25**")

        model_dir_path_bm25 = model_dir + '/' + bm25_model_nm
        bm25 = pickle.load(open(model_dir_path_bm25, 'rb'))

        labelFile_bm25 = model_dir + '/' + bm25_labels
        label_bm25 = pickle.load(open(labelFile_bm25, 'rb'))

        logger.info(f"Loaded label {labelFile_bm25} and model {model_dir_path_bm25}")

        df_test_predicted = df_test_predicted.apply(
            lambda x: getDocClassBasedOnScore_genrlzd(bm25, label_bm25, x, feature_name), axis=1)

        df_test_predicted['BM25_TOP_CLASS'] = df_test_predicted.apply(
            lambda x: -1 if x['BM25_CONFIDENCE'] < 5 else x['BM25_TOP_CLASS'], axis=1)

        logger.info(f"df_test_bm25_classified = {df_test_predicted.head(3)}")
        logger.info(f"Ended - SearchBased BM25**")
    except Exception as e:
        logger.error('Error in BM25 Classifier: ', e)
        raise

    try:
        logger.info(f"Starting - NearestNeighbor**")

        model_dir_path_NN = model_dir + '/' + nn1_model_nm
        NN1_Model = pickle.load(open(model_dir_path_NN, 'rb'))

        label_file_1nn = model_dir + '/' + nn1_labels
        label_1nn = pickle.load(open(label_file_1nn, 'rb'))
        logger.info(f"Loaded label {label_file_1nn} and model {model_dir_path_NN}")

        df_test_predicted = predictDataOutputNN(df_test_predicted, NN1_Model, feature_name, label_1nn, probability_range,
                                                save_pred_res)

        df_test_predicted['NNBR_TOP_CLASS'] = df_test_predicted.apply(
            lambda x: -1 if x['NNBR_CONFIDENCE'] < 0.95 else x['NNBR_TOP_CLASS'], axis=1)

        logger.info(f"NN classified = {df_test_predicted.head(3)}")

        logger.info(f"Ended - NearestNeighbor**")
    except Exception as e:
        logger.error('Error in Nearest Neighbors Classifier: ', e)
        raise

    try:
        logger.info(f"Starting - Integration**")
        cols = ['NB_TOP_CLASS', 'BM25_TOP_CLASS', 'NNBR_TOP_CLASS']
        logger.info(f"cols ={cols}")

        df_test_predicted_system = df_test_predicted.apply(lambda x: matchCols(x[cols], cols, required_perc_match), axis=1)
        df_test_predicted = pd.concat([df_test_predicted, df_test_predicted_system], axis=1)
        df_test_predicted = df_test_predicted.loc[:, ~df_test_predicted.columns.duplicated()].copy()
    except Exception as e:
        logger.error('Error in applying matchcols Classifier: ', e)
        raise

    try:
        score_cols = ['BM25_SCORES', 'BM25_PER_TOKEN_SCORES', 'NNBR_RESULT', 'NB_RESULT']
        df_test_predicted[score_cols] = df_test_predicted[score_cols].applymap(round_val_dict)
        confidence_cols = ['NB_CONFIDENCE', 'BM25_CONFIDENCE', 'NNBR_CONFIDENCE', 'SYSTEM_CONF']
        # df_test_predicted[confidence_cols] = df_test_predicted[confidence_cols].astype('float')
        df_test_predicted[confidence_cols] = np.round(df_test_predicted[confidence_cols], decimals=2)

        new_cols = df_test_predicted.columns.difference(df_with_regex.columns).to_list()
        for col in new_cols:
            df_with_regex[new_cols] = ''

        df_final_predicted = df_test_predicted.append(df_with_regex)
        df_final_predicted['SYSTEM_PREDICTION'] = np.where(df_final_predicted['RegexPrediction'] != 0, df_final_predicted['RegexPrediction'], df_final_predicted['SYSTEM_PREDICTION'])
        df_final_predicted['SYSTEM_CONF'] = np.where(df_final_predicted['RegexPrediction'] != 0, 1.0, df_final_predicted['SYSTEM_CONF'])
        df_final_predicted = df_final_predicted[['DatabaseName', 'TableName', 'columnName', 'col_name_stdzd',     #'ProjectName', 'Domain',
                                                 'RegexPrediction', 'NB_RESULT','NB_CONFIDENCE', 'NB_TOP_CLASS','BM25_RESULT', 'BM25_SCORES',
                                                 'BM25_PER_TOKEN_SCORES', 'BM25_TOP_CLASS', 'BM25_CONFIDENCE','NNBR_RESULT', 'NNBR_CONFIDENCE',
                                                 'NNBR_TOP_CLASS', 'IS_MATCH', 'SYSTEM_PREDICTION', 'SYSTEM_CONF']]

        df_mapping = pd.DataFrame({
            'columnName_new': columnName_list,})
        sort_mapping = df_mapping.reset_index().set_index('columnName_new')
        df_final_predicted['input_columnName_sort'] = df_final_predicted['columnName'].map(sort_mapping['index'])
        df_final_predicted.sort_values('input_columnName_sort', inplace=True)
        logger.info(f"Final Prediction Dataframe = {df_final_predicted.to_dict()}")
        final_pred_df = df_final_predicted[['SYSTEM_PREDICTION', 'SYSTEM_CONF']]
        guid_mapping_file = pd.read_csv('data/BT_GUID_MAPPING_DEV.csv')
        final_pred_df.rename(columns={'SYSTEM_PREDICTION': 'gcaf_business terms'}, inplace=True)
        final_pred_guid = pd.merge(final_pred_df, guid_mapping_file, how='left', on=['gcaf_business terms'])
        final_pred_guid['SYSTEM_PREDICTION'] = np.where(final_pred_guid['GUID'].notnull(),
                                                           final_pred_guid['GUID'],
                                                           '')
        # empty_list = []
        # final_pred_guid['SYSTEM_PREDICTION'] = np.where(final_pred_guid['SYSTEM_PREDICTION'] == -1,
        #                                                    '',
        #                                                    final_pred_guid['SYSTEM_PREDICTION'])
        final_pred_guid['SYSTEM_CONF'] = np.where(final_pred_guid['SYSTEM_PREDICTION'] == '',
                                                           '',
                                                           final_pred_guid['SYSTEM_CONF'])

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
        fn = f"{dest_folder}/Final_File_With_Integrated_Classifications_{run_date}.json"
        final_json.to_json(fn)

        logger.info('Output = **********************')
        logger.info(df_test_predicted.head())
        logger.info(f"Ended - Integration**")
        logger.info("Ending SystemOfClassifiers_PredictClassification *******")
    except Exception as e:
        logger.error('Error in Post Classifier: ', e)
        raise


