# v1.14
import sys
import datetime
import logging
import pickle
import pandas as pd
import numpy as np
from Standardization.stdz_data import standardization_main
from RegexClassifier.regexmodeltransformer import RegExModelTransformer
from NaiveBayes.NaiveBayesPredict import predictDataOutput
from NNAlgorithm.NearestNeighborPredict import predictDataOutputNN
from DecisionTrees.DecisionTreesClassifier import predictDataOutputDT
# from DecisionTrees.DecisionTreeSBERT import predictDataOutputDT_SBERT
# from sentence_transformers import SentenceTransformer

logging.basicConfig(filename="logs/run_cognitive.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
# Set the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


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


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        file_name = args[0]
        col_name = args[1]
        col_name_stdzd = args[2]
        dest_folder = args[3]
        label = args[4]
        latest_lookup = args[5]
        pickle_v = args[6]

        if label == 'business_term':
            model_version = f'BT_{pickle_v}'
        else:
            model_version = f'POLT_{pickle_v}'


        if len(args) == 7:
            logger.info(f'file_name = {file_name}')
            logger.info(f'col_name = {col_name} ')
            logger.info(f'standardized col_name = {col_name_stdzd} ')
            logger.info(f'dest_folder = {dest_folder}')
        else:
            logger.info('Required args are not provided')
    except Exception as e:
        logger.error('Error with Arguments: ', e)
        raise
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

    try:
        logger.info('Starting Standardization')
        if file_name.endswith('csv'):
            ip_data_df = pd.read_csv(file_name)
        elif file_name.endswith('parquet'):
            ip_data_df = pd.read_parquet(file_name)
        else:
            ip_data_df = pd.DataFrame()
        # ip_data_df = ip_data_df[0:10]
        df = standardization_main(ip_data_df, col_name, col_name_stdzd, label, latest_lookup)
        logger.info('Successfully Standardized')
    except Exception as e:
        logger.error('Error in Standardization: ', e)
        raise

    try:
        rule_table_filename = 'data/regex_rules_v11.csv'
        rule_table = pd.read_csv(rule_table_filename)
        # loading the model
        logger.info('Starting Regex Model')
        model = RegExModelTransformer(rule_table)

        df['prediction'] = model.transform(df[col_name_stdzd])
        logger.info('Successfully Classified with Regex')
        df.rename(columns={'prediction': 'RegexPrediction'}, inplace=True)
        # -------------------------------------Uncomment if Classifier Overrides Regex-----------------------------------------------
        # df_without_regex = df.copy()
        # -------------------------------------Uncomment if Classifier Overrides Regex-----------------------------------------------
    except Exception as e:
        logger.error('Error in Regex: ', e)
        raise
    # -------------------------------------comment if Classifier Overrides Regex-----------------------------------------------

    try:
        df_without_regex = df[df['RegexPrediction'] == 0]
        df_with_regex = df[df['RegexPrediction'] != 0]
    except Exception as e:
        logger.error('Error in Regex Post Processing: ', e)
        raise
    # -------------------------------------comment if Classifier Overrides Regex-----------------------------------------------

    try:
        feature_name = col_name_stdzd
        probability_range = 3
        save_pred_res = False
        model_dir = "models"
        bayes_model_nm = f'Bayes_{model_version}.pkl'
        bayes_labels = f'Bayes_Labels_{model_version}.pkl'
        # bm25_model_nm = f'BM25_{model_version}.pkl'
        # bm25_labels = f'BM25_Labels.pkl'
        nn1_labels = f'NearNghbr_Labels_{model_version}.pkl'
        nn1_model_nm = f'NearNghbr_{model_version}.pkl'

        dt_sbert_model = f'DT_SBERT_{model_version}.pkl'
        # dt_sbert_labels = f'DT_SBERT_labels_{model_version}.pkl'
        dt_model = f'DT_{model_version}.pkl'
        # dt_labels = f'DT_labels_{model_version}.pkl'
        nn_sbert_labels = f'NearNghbr_SBERT_Labels_{model_version}.pkl'
        nn_sbert_model = f'NearNghbr_SBERT_{model_version}.pkl'

        # required_perc_match = 0.66
        required_perc_match = 3 / 5

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

        df_test_predicted = predictDataOutput(df_without_regex, trainingNBModel, feature_name, label_nb,
                                              probability_range,
                                              save_pred_res)
        df_test_predicted['NB_TOP_CLASS'] = df_test_predicted.apply(
            lambda x: -1 if x['NB_CONFIDENCE'] < 0.95 else x['NB_TOP_CLASS'], axis=1)

        logger.info(f"Ended - NaiveBayes prediction**")
    except Exception as e:
        logger.error('Error in NaiveBayes Classifier: ', e)
        raise

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
            lambda x: -1 if x['DT_CONFIDENCE'] < 0.95 else x['DT_TOP_CLASS'], axis=1)
        logger.info(f"Ended - Decision Tree TFIDF prediction**")
    except Exception as e:
        logger.error('Error in Decision Tree TFIDF Classifier: ', e)
        raise

    try:
        logger.info(f"Starting - NearestNeighbor**")

        model_dir_path_NN = model_dir + '/' + nn1_model_nm
        NN1_Model = pickle.load(open(model_dir_path_NN, 'rb'))

        label_file_1nn = model_dir + '/' + nn1_labels
        label_1nn = pickle.load(open(label_file_1nn, 'rb'))
        logger.info(f"Loaded label {label_file_1nn} and model {model_dir_path_NN}")

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

    try:
        logger.info(f"Starting - Integration**")
        cols = ['NB_TOP_CLASS', 'NNBR_TOP_CLASS', 'DT_TOP_CLASS']
        logger.info(f"cols ={cols}")

        df_test_predicted_system = df_test_predicted.apply(lambda x: matchCols(x[cols], cols, required_perc_match),
                                                           axis=1)
        df_test_predicted = pd.concat([df_test_predicted, df_test_predicted_system], axis=1)
        df_test_predicted = df_test_predicted.loc[:, ~df_test_predicted.columns.duplicated()].copy()
    except Exception as e:
        logger.error('Error in applying matchcols Classifier: ', e)
        raise

    try:
        # df_test_predicted.drop(columns=sbert_columns_pred, inplace=True)
        score_cols = ['NNBR_RESULT', 'NB_RESULT']
        df_test_predicted[score_cols] = df_test_predicted[score_cols].applymap(round_val_dict)
        confidence_cols = ['NB_CONFIDENCE', 'NNBR_CONFIDENCE', 'DT_CONFIDENCE', 'SYSTEM_CONF']
        # df_test_predicted[confidence_cols] = df_test_predicted[confidence_cols].astype('float')
        df_test_predicted[confidence_cols] = np.round(df_test_predicted[confidence_cols], decimals=2)

        # -------------------------------------comment if Classifier Overrides Regex-----------------------------------------------
        new_cols = df_test_predicted.columns.difference(df_with_regex.columns).to_list()
        for col in new_cols:
            df_with_regex[new_cols] = ''

        df_final_predicted = df_test_predicted.append(df_with_regex)

        df_final_predicted['SYSTEM_PREDICTION'] = np.where(df_final_predicted['RegexPrediction'] != 0, df_final_predicted['RegexPrediction'], df_final_predicted['SYSTEM_PREDICTION'])

        # -------------------------------------comment if Classifier Overrides Regex-----------------------------------------------
        # -------------------------------------Uncomment if Classifier Overrides Regex-----------------------------------------------

        # df_final_predicted = df_test_predicted.copy()
        #
        # df_final_predicted['SYSTEM_PREDICTION'] = np.where(
        #     (df_final_predicted['RegexPrediction'] != 0) & (df_final_predicted['SYSTEM_PREDICTION'] == -1),
        #     df_final_predicted['RegexPrediction'],
        #     df_final_predicted['SYSTEM_PREDICTION'])
        # -------------------------------------Uncomment if Classifier Overrides Regex-----------------------------------------------


        df_final_predicted['SYSTEM_CONF'] = np.where(df_final_predicted['RegexPrediction'] != 0, 1.0,
                                                     df_final_predicted['SYSTEM_CONF'])

        df_final_predicted.rename(columns={'NB_RESULT': 'BAYES_RESULT', 'NB_CONFIDENCE': 'BAYES_CONFIDENCE',
                                           'NB_TOP_CLASS': 'BAYES_TOP_CLASS', 'NNBR_RESULT': 'KNN_RESULT',
                                           'NNBR_CONFIDENCE': 'KNN_CONFIDENCE', 'NNBR_TOP_CLASS': 'KNN_TOP_CLASS'},
                                  inplace=True)

        fn = f"{dest_folder}/Final_File_With_Integrated_Classifications_{run_date}_classified.csv"
        df_final_predicted.to_csv(fn, index=None)

        logger.info('Output = **********************')
        logger.info(df_test_predicted.head())
        logger.info(f"Ended - Integration**")
        logger.info("Ending SystemOfClassifiers_PredictClassification *******")
    except Exception as e:
        logger.error('Error in Post Classifier: ', e)
        raise

