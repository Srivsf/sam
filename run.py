#v1.17
import sys
import datetime
import os
import yaml
import logging
import pickle
import pandas as pd
import numpy as np
import joblib
import timeit
from Standardization.stdz_data import standardization_main
from Standardization.stdz_data_v2 import standardization
from RegexClassifier.regexmodeltransformer import RegExModelTransformer
from NaiveBayes.NaiveBayesPredict import predictDataOutput
from NNAlgorithm.NearestNeighborPredict import predictDataOutputNN
from NNAlgorithm.NearestNeighborSBERT import predictDataOutputNN_SBERT
from DecisionTrees.DecisionTreesClassifier import predictDataOutputDT
from DecisionTrees.DecisionTreeSBERT import predictDataOutputDT_SBERT
from sentence_transformers import SentenceTransformer
from memory_profiler import profile


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

if __name__ == "__main__":
    try:
        dt_start = datetime.datetime.now()
        args = sys.argv[1:]
        file_name = args[0]
        col_name = args[1]
        col_name_stdzd = args[2]
        dest_folder = args[3]
        latest_lookup = args[4]
        pickle_v_BT = args[5]
        pickle_v_POLT = args[6]
        rule_table_filename_bt = args[7]
        rule_table_filename_polt = args[8]


        first_label_run = 0
        with open("config.yaml", "r") as file:
            try:
                configs = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        Class_num_threshold = configs['Class_number_threshold']
        original_label_column_list = []
        if configs['business_term'] == 1:
            original_label_column_list += ['business_term']
        else:
            pass

        if configs['policy_tag'] == 1:
            original_label_column_list += ['policytagid']
        else:
            pass
    except Exception as e:
        logger.error('Error with Arguments: ', e)
        raise

    for label in original_label_column_list:
        if label == 'business_term':
            model_version = f'BT_{pickle_v_BT}'
            rule_table_filename = rule_table_filename_bt
        else:
            model_version = f'POLT_{pickle_v_POLT}'
            rule_table_filename = rule_table_filename_polt

        if len(args) == 8:
            logger.info(f'file_name = {file_name}')
            logger.info(f'col_name = {col_name} ')
            logger.info(f'standardized col_name = {col_name_stdzd} ')
            logger.info(f'dest_folder = {dest_folder}')
        else:
            logger.info('Required args are not provided')

        date_today = datetime.datetime.now()
        try:
            run_date_string = ''.join(x for x in file_name if x.isdigit())
            if len(run_date_string) == 6:
                run_date = f'{run_date_string[:2]}_{run_date_string[2:4]}_{run_date_string[4:6]}'
            else:
                run_date = f'{date_today.month}_{date_today.day}_{date_today.year}'
        except Exception as e:
            logger.error('Error in extracting run date ', e)
            run_date = f'{date_today.month}_{date_today.day}_{date_today.year}'

        if first_label_run == 0:
            try:
                logger.info('Starting Standardization')
                if file_name.endswith('csv'):
                    ip_data_df = pd.read_csv(file_name)
                elif file_name.endswith('parquet'):
                    ip_data_df = pd.read_parquet(file_name)
                else:
                    ip_data_df = pd.DataFrame()
                ip_data_df = ip_data_df[0:1000]
                # ip_data_df.rename(columns={'Column Name': 'columnName'}, inplace=True)
                dt_stand_start = datetime.datetime.now()
                # df = standardization_main(ip_data_df, col_name, col_name_stdzd, label, latest_lookup)
                # timeit
                df = standardization(latest_lookup, ip_data_df, col_name, col_name_stdzd, label)
                dt_stand_end = datetime.datetime.now()
                standardization_time = dt_stand_end-dt_stand_start
                print('Standardization Time :', standardization_time)
                logger.info('Successfully Standardized')
            except Exception as e:
                logger.error('Error in Standardization: ', e)
                raise

        else:
            df = df_final_predicted.copy()
            del df_final_predicted


        cols = []
        confidence_cols = []
        score_cols = []

        if configs['regex'] == 'on':
            try:
                dt_regex_start = datetime.datetime.now()
                # rule_table_filename = 'data/regex_rules_v11.csv'
                rule_table = pd.read_csv(rule_table_filename)
                # rule_table = rule_table[rule_table['label_name'] == label]
                # rule_table.rename(columns={'term': label}, inplace=True)
                # validating Regex Rules file
                if label in rule_table.columns:
                    pass
                else:
                    logger.error('Not a Valid Regex Rules File')
                # loading the model
                logger.info('Starting Regex Model')
                model = RegExModelTransformer(rule_table)
                timeit
                df['prediction'] = model.transform(df[col_name_stdzd])
                logger.info('Successfully Classified with Regex')
                df.rename(columns={'prediction': 'RegexPrediction'}, inplace=True)
                dt_regex_end = datetime.datetime.now()
                print('Regex Execution Time :', dt_regex_end - dt_regex_start)
                # df_without_regex = df.copy()
            except Exception as e:
                logger.error('Error in Regex: ', e)
                raise
        # -------------------------------------change1-----------------------------------------------
            if configs['regex_override'] == 1:
                df_without_regex = df.copy()
                df_with_regex = pd.DataFrame()
            else:
                try:
                    df_without_regex = df[df['RegexPrediction'] == 0]
                    df_with_regex = df[df['RegexPrediction'] != 0]
                except Exception as e:
                    logger.error('Error in Regex Post Processing: ', e)
                    raise
        # -------------------------------------change1-----------------------------------------------
        else:
            df_without_regex = df.copy()
            df_with_regex = pd.DataFrame()

        try:
            if configs['inferred_domain'] == 'on':
                dmodel = joblib.load("domain_id_models/domain_model.pkl")
                dvectorizer = joblib.load("domain_id_models/domain_vectorizer.pkl")
                df_without_regex["domain"] = dmodel.predict(dvectorizer.transform(df_without_regex["COLUMN_NAME_STDZD"]))
            if configs['inferred_identifier'] == 'on':
                imodel = joblib.load("domain_id_models/identifier_model.pkl")
                ivectorizer = joblib.load("domain_id_models/identifier_vectorizer.pkl")
                df_without_regex["identifier"] = imodel.predict(ivectorizer.transform(df_without_regex["COLUMN_NAME_STDZD"]))
            if (configs['inferred_domain'] == 'on') | (configs['inferred_identifier'] == 'on'):
                if (configs['inferred_domain'] == 'on') & (configs['inferred_identifier'] == 'on'):
                    df_without_regex["DI_COLUMN_NAME_STDZD"] = "inferred " + df_without_regex["domain"] + " inferred " + df_without_regex[
                        "identifier"] + " " + df_without_regex["COLUMN_NAME_STDZD"]
                elif configs['inferred_identifier'] == 'on':
                    df_without_regex["DI_COLUMN_NAME_STDZD"] = "inferred " + df_without_regex["identifier"] + " " + df_without_regex["COLUMN_NAME_STDZD"]
                elif configs['inferred_domain'] == 'on':
                    df_without_regex["DI_COLUMN_NAME_STDZD"] = "inferred " + df_without_regex["domain"] + " " + df_without_regex["COLUMN_NAME_STDZD"]
                df_without_regex.rename(columns={'COLUMN_NAME_STDZD': 'COLUMN_NAME_STDZD_old',
                                   'DI_COLUMN_NAME_STDZD': 'COLUMN_NAME_STDZD'}, inplace=True)
            else:
                pass
        except Exception as e:
            logger.error(f"Error in Creating Domain Identifier: {e}")
            raise

        try:
            feature_name = col_name_stdzd
            probability_range = 3
            save_pred_res = False
            model_dir = "models"
            bayes_model_nm = f'Bayes_{model_version}.pkl'
            bayes_labels = f'Bayes_Labels_{model_version}.pkl'
            # bm25_model_nm = 'BM25.pkl'
            # bm25_labels = 'BM25_Labels.pkl'
            nn1_labels = f'NearNghbr_Labels_{model_version}.pkl'
            nn1_model_nm = f'NearNghbr_{model_version}.pkl'

            dt_sbert_model = f'DT_SBERT_{model_version}.pkl'
            dt_model = f'DT_{model_version}.pkl'
            nn_sbert_labels = f'NearNghbr_SBERT_Labels_{model_version}.pkl'
            nn_sbert_model = f'NearNghbr_SBERT_{model_version}.pkl'

            # required_perc_match = 0.66
            required_perc_match = configs['voting_config'][0]['classifier_vote']/configs['voting_config'][1]['total_classifiers']

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

        # if (configs['bayes_tfidf'] == 'on') | (configs['dt_tfidf'] == 'on') | (configs['knn_tfidf'] == 'on'):
        #     try:
        #         vectorizer = joblib.load('models/')
        #         doc_term_matrix = vectorizer.fit_transform()
        #         joblib.dump(vectorizer, 'models/identifier_vectorizer.pkl')
        #         df_train_vectorized = pd.DataFrame(doc_term_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        #     except Exception as e:
        #         logger.error('Error in Creating TFIDF: ', e)
        #         raise

        if (configs['dt_llm'] == 'on') | (configs['knn_llm'] == 'on'):
            try:
                dataset_columns_list = df_without_regex.columns.tolist()
                model_name = "deberta-v2-xlarge"
                embedding_model = SentenceTransformer(f'{model_dir}/{model_name}')
                # df_without_regex['embeddings'] = df_without_regex[col_name_stdzd].apply(lambda x: embedding_model.encode(x).tolist())
                encoded_col_name_stdzd = embedding_model.encode(df_without_regex[col_name_stdzd].tolist(), batch_size=1000)
                df_without_regex['embeddings'] = encoded_col_name_stdzd.tolist()
                print(df_without_regex['embeddings'].iloc[0])

                def list_split(df):
                    df_train_sbert = df.copy()
                    df_train_sbert = df_train_sbert.reset_index(drop=True)
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

        if configs['bayes_tfidf'] == 'on':
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
                    lambda x: -1 if x['NB_CONFIDENCE'] < Class_num_threshold else x['NB_TOP_CLASS'], axis=1)

                cols += ['NB_TOP_CLASS']
                confidence_cols += ['NB_CONFIDENCE']
                score_cols += ['NB_RESULT']
                logger.info(f"Ended - NaiveBayes prediction**")
            except Exception as e:
                logger.error('Error in NaiveBayes Classifier: ', e)
                raise
        else:
            df_test_predicted = df_without_regex.copy()
            logger.info('Bayes TFIDF Classifier if Turned Off from Configs')

        if configs['dt_llm'] == 'on':
            try:
                logger.info(f"Started - Decision Tree SBERT prediction**")

                model_dir_path_dt_sbert = model_dir + '/' + dt_sbert_model
                logger.info("loading DT SBERT model **")
                trainingDTSBERTModel = pickle.load(open(model_dir_path_dt_sbert, 'rb'))

                df_test_predicted = predictDataOutputDT_SBERT(df_test_predicted, trainingDTSBERTModel, probability_range, save_pred_res,
                                                              sbert_columns_pred)
                df_test_predicted['DT_SBERT_TOP_CLASS'] = df_test_predicted.apply(
                    lambda x: -1 if x['DT_SBERT_CONFIDENCE'] < Class_num_threshold else x['DT_SBERT_TOP_CLASS'], axis=1)
                cols += ['DT_SBERT_TOP_CLASS']
                confidence_cols += ['DT_SBERT_CONFIDENCE']
                logger.info(f"Ended - Decision Tree SBERT prediction**")
            except Exception as e:
                logger.error('Error in Decision Tree SBERT Classifier: ', e)
                raise
        else:
            logger.info('Decision Tree SBERT Classifier if Turned Off from Configs')

        if configs['knn_llm'] == 'on':
            try:
                logger.info(f"Started - KNN SBERT prediction**")

                model_dir_path_knn_sbert = model_dir + '/' + nn_sbert_model
                logger.info("loading KNN SBERT model **")
                trainingKNNSBERTModel = pickle.load(open(model_dir_path_knn_sbert, 'rb'))

                labelFile_knn_sbert = model_dir + '/' + nn_sbert_labels
                label_knn_sbert = pickle.load(open(labelFile_knn_sbert, 'rb'))
                logger.info(f"Loaded label {labelFile_knn_sbert} and model {model_dir_path_knn_sbert}")


                df_test_predicted = predictDataOutputNN_SBERT(df_test_predicted, trainingKNNSBERTModel, label_knn_sbert,
                                                              probability_range, save_pred_res, sbert_columns_pred)
                df_test_predicted['KNN_SBERT_TOP_CLASS'] = df_test_predicted.apply(
                    lambda x: -1 if x['KNN_SBERT_CONFIDENCE'] < Class_num_threshold else x['KNN_SBERT_TOP_CLASS'], axis=1)
                cols += ['KNN_SBERT_TOP_CLASS']
                confidence_cols += ['KNN_SBERT_CONFIDENCE']
                score_cols = ['KNN_SBERT_RESULT']
                logger.info(f"Ended - KNN SBERT prediction**")
            except Exception as e:
                logger.error('Error in KNN SBERT Classifier: ', e)
                raise
        else:
            logger.info('KNN SBERT Classifier if Turned Off from Configs')

        if configs['dt_tfidf'] == 'on':
            try:
                logger.info(f"Started - Decision Tree TFIDF prediction**")

                model_dir_path_dt = model_dir + '/' + dt_model
                logger.info("loading DT model **")
                trainingDTModel = pickle.load(open(model_dir_path_dt, 'rb'))

                # labelFile_dt = model_dir + '/' + dt_labels
                # label_dt = pickle.load(open(labelFile_dt, 'rb'))
                # logger.info(f"Loaded label {labelFile_dt} and model {model_dir_path_dt}")


                df_test_predicted = predictDataOutputDT(df_test_predicted, trainingDTModel, feature_name, probability_range,
                                                        save_pred_res)
                df_test_predicted['DT_TOP_CLASS'] = df_test_predicted.apply(
                    lambda x: -1 if x['DT_CONFIDENCE'] < Class_num_threshold else x['DT_TOP_CLASS'], axis=1)
                cols += ['DT_TOP_CLASS']
                confidence_cols += ['DT_CONFIDENCE']
                logger.info(f"Ended - Decision Tree TFIDF prediction**")
            except Exception as e:
                logger.error('Error in Decision Tree TFIDF Classifier: ', e)
                raise
        else:
            logger.info('Decision Tree TFIDF Classifier if Turned Off from Configs')
        if configs['knn_tfidf'] == 'on':
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
                    lambda x: -1 if x['NNBR_CONFIDENCE'] < Class_num_threshold else x['NNBR_TOP_CLASS'], axis=1)

                logger.info(f"NN classified = {df_test_predicted.head(3)}")
                cols += ['NNBR_TOP_CLASS']
                confidence_cols += ['NNBR_CONFIDENCE']
                score_cols = ['NNBR_RESULT']

                logger.info(f"Ended - NearestNeighbor**")
            except Exception as e:
                logger.error('Error in Nearest Neighbors Classifier: ', e)
                raise
        else:
            logger.info('KNN TFIDF Classifier if Turned Off from Configs')

        try:
            logger.info(f"Starting - Integration**")
            # cols = ['NB_TOP_CLASS','NNBR_TOP_CLASS', 'DT_TOP_CLASS', 'DT_SBERT_TOP_CLASS', 'KNN_SBERT_TOP_CLASS']
            logger.info(f"cols ={cols}")

            df_test_predicted_system = df_test_predicted.apply(lambda x: matchCols(x[cols], cols, required_perc_match), axis=1)
            df_test_predicted = pd.concat([df_test_predicted, df_test_predicted_system], axis=1)
            df_test_predicted = df_test_predicted.loc[:, ~df_test_predicted.columns.duplicated()].copy()
        except Exception as e:
            logger.error('Error in applying matchcols Classifier: ', e)
            raise

        try:
            df_test_predicted.drop(columns=sbert_columns_pred, inplace=True)
            # score_cols = ['NNBR_RESULT', 'NB_RESULT', 'KNN_SBERT_RESULT']
            df_test_predicted[score_cols] = df_test_predicted[score_cols].applymap(round_val_dict)
            # confidence_cols = ['NB_CONFIDENCE', 'NNBR_CONFIDENCE', 'KNN_SBERT_CONFIDENCE', 'DT_CONFIDENCE', 'DT_SBERT_CONFIDENCE', 'SYSTEM_CONF']
            # df_test_predicted[confidence_cols] = df_test_predicted[confidence_cols].astype('float')
            df_test_predicted[confidence_cols] = np.round(df_test_predicted[confidence_cols], decimals=2)

            if configs['regex'] == 'on':
                #-------------------------------------change2-----------------------------------------------
                new_cols = df_test_predicted.columns.difference(df_with_regex.columns).to_list()
                for col in new_cols:
                    df_with_regex[new_cols] = ''

                df_final_predicted = df_test_predicted.append(df_with_regex)
                # -------------------------------------change2-----------------------------------------------

                if configs['regex_override'] == 1:
                    df_final_predicted['SYSTEM_PREDICTION'] = np.where(
                        (df_final_predicted['RegexPrediction'] != 0) & (df_final_predicted['SYSTEM_PREDICTION'] == -1),
                        df_final_predicted['RegexPrediction'],
                        df_final_predicted['SYSTEM_PREDICTION'])
                else:
                    df_final_predicted['SYSTEM_PREDICTION'] = np.where(df_final_predicted['RegexPrediction'] != 0,
                                                                       df_final_predicted['RegexPrediction'],
                                                                       df_final_predicted['SYSTEM_PREDICTION'])

                    df_final_predicted['SYSTEM_CONF'] = np.where(df_final_predicted['RegexPrediction'] != 0, 1.0,
                                                                 df_final_predicted['SYSTEM_CONF'])
            else:
                df_final_predicted = df_test_predicted.copy()


            if first_label_run == 0:
                df_final_predicted.rename(columns={'NB_RESULT': 'BAYES_RESULT_BT', 'NB_CONFIDENCE': 'BAYES_CONFIDENCE_BT',
                                                   'NB_TOP_CLASS': 'BAYES_TOP_CLASS_BT', 'NNBR_RESULT': 'KNN_RESULT_BT',
                                                   'NNBR_CONFIDENCE': 'KNN_CONFIDENCE_BT', 'NNBR_TOP_CLASS': 'KNN_TOP_CLASS_BT',
                                                   'DT_TOP_CLASS': 'DT_TOP_CLASS_BT', 'DT_CONFIDENCE': 'DT_CONFIDENCE_BT',
                                                   'DT_SBERT_TOP_CLASS': 'DT_SBERT_TOP_CLASS_BT', 'DT_SBERT_CONFIDENCE':'DT_SBERT_CONFIDENCE_BT',
                                                   'KNN_SBERT_RESULT': 'KNN_SBERT_RESULT_BT', 'KNN_SBERT_CONFIDENCE': 'KNN_SBERT_CONFIDENCE_BT',
                                                   'KNN_SBERT_TOP_CLASS': 'KNN_SBERT_TOP_CLASS_BT', 'SYSTEM_PREDICTION': 'SYSTEM_PREDICTION_BT',
                                                   'SYSTEM_CONF': 'SYSTEM_CONF_BT', 'RegexPrediction': 'RegexPrediction_BT'},inplace=True)
            elif first_label_run == 1:
                df_final_predicted.rename(columns={'NB_RESULT': 'BAYES_RESULT_POLT', 'NB_CONFIDENCE': 'BAYES_CONFIDENCE_POLT',
                                                   'NB_TOP_CLASS': 'BAYES_TOP_CLASS_POLT', 'NNBR_RESULT': 'KNN_RESULT_POLT',
                                                   'NNBR_CONFIDENCE': 'KNN_CONFIDENCE_POLT', 'NNBR_TOP_CLASS': 'KNN_TOP_CLASS_POLT',
                                                   'DT_TOP_CLASS': 'DT_TOP_CLASS_POLT', 'DT_CONFIDENCE': 'DT_CONFIDENCE_POLT',
                                                   'DT_SBERT_TOP_CLASS': 'DT_SBERT_TOP_CLASS_POLT', 'DT_SBERT_CONFIDENCE':'DT_SBERT_CONFIDENCE_POLT',
                                                   'KNN_SBERT_RESULT': 'KNN_SBERT_RESULT_POLT', 'KNN_SBERT_CONFIDENCE': 'KNN_SBERT_CONFIDENCE_POLT',
                                                   'KNN_SBERT_TOP_CLASS': 'KNN_SBERT_TOP_CLASS_POLT', 'SYSTEM_PREDICTION': 'SYSTEM_PREDICTION_POLT',
                                                   'SYSTEM_CONF': 'SYSTEM_CONF_POLT',  'RegexPrediction': 'RegexPrediction_POLT'},inplace=True)
            first_label_run += 1
        except Exception as e:
            logger.error('Error in Post Classifier: ', e)
            raise
    try:
        fn = f"{dest_folder}/Final_File_With_Integrated_Classifications_{run_date}_classified.csv"
        df_final_predicted.to_csv(fn, index=None)

        logger.info('Output = **********************')
        logger.info(df_final_predicted.head())
        logger.info(f"Ended - Integration**")
        logger.info("Ending SystemOfClassifiers_PredictClassification *******")
        dt2 = datetime.datetime.now()
        print('Time finished in : ', dt2 - dt_start)
    except Exception as e:
        logger.error('Error in Writing df_final_predicted to CSV File: ', e)
        raise

    try:
        final_pred_df = df_final_predicted.copy()
        final_pred_df.rename(
            columns={'SYSTEM_PREDICTION_BT': 'Business_Term_guid', 'SYSTEM_PREDICTION_POLT': 'VAST Data Element ID'},
            inplace=True)
        guid_mapping_file = pd.read_csv('data/GCAF_BT_POLT_GUID.csv', encoding='unicode_escape')
        if configs['business_term'] == 1:
            # final_pred_df = df_final_predicted[['columnName', 'SYSTEM_PREDICTION_BT', 'SYSTEM_CONF_BT']]
            guid_mapping_file['GUID'] = guid_mapping_file['Source Repository Id'] + '_' + guid_mapping_file['Term Id']

            guid_mapping_file.rename(columns={'Business Term': 'Business_Term_guid'}, inplace=True)
            guid_mapping_file_bt = guid_mapping_file[['GUID', 'Business_Term_guid']]


            final_pred_guid = pd.merge(final_pred_df, guid_mapping_file_bt, how='left', on=['Business_Term_guid'])
            final_pred_guid.rename(columns={'GUID': 'GUID_BT'}, inplace=True)
            final_pred_guid['SYSTEM_PREDICTION_BT_GUID'] = np.where(final_pred_guid['GUID_BT'].notnull(),
                                                                    final_pred_guid['GUID_BT'], '')

        if configs['policy_tag'] == 1:
            guid_mapping_file_polt = guid_mapping_file[['GUID', 'VAST Data Element ID']]
            final_pred_guid = pd.merge(final_pred_guid, guid_mapping_file_polt, how='left', on=['VAST Data Element ID'])
            final_pred_guid['SYSTEM_PREDICTION_POLT_GUID'] = np.where(final_pred_guid['GUID'].notnull(),
                                                                    final_pred_guid['GUID'], '')
        dt_end = datetime.datetime.now()
        print('Processing_Time : ', dt_end-dt_start)
    except Exception as e:
        logger.error('Error in Writing Applying GCAF : ', e)
        raise
    #
    # model_name = "deberta-v2-xlarge"
    # embedding_model = SentenceTransformer(modelpath)
    # data_x_embeddings = embedding_model.encode(df[feature_name].tolist(), show_progress_bar=True)
    # print(data_x_embeddings.shape)
    # df["embeddings"] = data_x_embeddings.tolist()
    #
    # df_new = df.embeddings.apply(pd.Series)
    # df = df.loc[:, df.columns != 'embeddings']



