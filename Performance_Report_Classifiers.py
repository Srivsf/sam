import pandas as pd
import numpy as np
import os
import warnings
from sklearn import metrics


def compareActualPredicted_up(dfinal, orig_label, sec_class_lst, fold, classifier):
    combined_df = pd.DataFrame()
    th_lst = [0.95]
    if classifier == 'BM25':
        th_lst = [7]
    if classifier == 'SYSTEM':
        test_dev = dfinal
        test_dev['label'] = test_dev['predicted']
        test_dev['actual'] = test_dev[orig_label]
        print(test_dev[['label', 'actual']].head())

        y_act = test_dev['actual'].values
        y_pred = test_dev['label'].values

        print(y_act)
        print(y_pred)

        report = metrics.classification_report(y_act, y_pred, output_dict=True, labels=sec_class_lst)

        df = pd.DataFrame(report).transpose()
        df['fold'] = fold
        df = df.drop(['micro avg', 'macro avg', 'weighted avg'])
        # df = df.drop(['macro avg', 'weighted avg'])
        combined_df = combined_df.append(df)

    else:
    # for th in [0.6,0.65,0.7,0.75,0.8, 0.85,0.9,0.91,0.92,0.93,0.94, 0.95]:
        for th in th_lst:
            test_dev = dfinal
            test_dev['label'] = test_dev.apply(lambda x: -1 if x['CONFIDENCE'] < th else x['predicted'], axis=1)
            test_dev['actual'] = test_dev[orig_label]
            print(test_dev[['label', 'actual']].head())

            y_act = test_dev['actual'].values
            y_pred = test_dev['label'].values

            print(y_act)
            print(y_pred)

            report = metrics.classification_report(y_act, y_pred, output_dict=True, labels=sec_class_lst)

            df = pd.DataFrame(report).transpose()
            df['threshold'] = th
            df['fold'] = fold
            if classifier == 'NB' or classifier == 'BM25' or classifier == 'NNBR':
                df = df.drop(['micro avg','macro avg','weighted avg'])
            else:
                df = df.drop(['macro avg', 'weighted avg', 'accuracy'])

            # df = df.drop(['macro avg', 'weighted avg'])
            combined_df = combined_df.append(df)

    print('complete')
    return combined_df


def main():
    print('Start of main()**')
    warnings.filterwarnings("ignore")

    label = 'business_term'
    classifier = 'SYSTEM'

    if classifier == 'NB':
        res_folder = './/NaiveBayes//RESULTS/'
        pred_class_column = 'NB_TOP_CLASS'
        confidence_col_name = 'NB_CONFIDENCE'
    if classifier == 'BM25':
        res_folder = '..//SearchEngineBM25//RESULTS/'
        pred_class_column = 'BM25_TOP_CLASS'
        confidence_col_name = 'BM25_CONFIDENCE'

    if classifier == 'NNBR':
        res_folder = '..//NNAlgorithm//RESULTS/'
        pred_class_column = 'NNBR_TOP_CLASS'
        confidence_col_name = 'NNBR_CONFIDENCE'

    if classifier == 'SYSTEM':
        res_folder = './RESULTS/'
        pred_class_column = 'SYSTEM_PREDICTION'

    print(f"res_folder  = {res_folder}")
    entries = os.listdir(res_folder)
    reports_lst = []
    fld = 1
    for classified_file in entries:
        if classified_file.endswith("classified.csv"):
            print("classified_file ********* =", classified_file)
            df_classified = pd.read_csv(res_folder + classified_file)
            df_classified['predicted'] = df_classified[pred_class_column]
            if not classifier == 'SYSTEM':
                df_classified['CONFIDENCE'] = df_classified[confidence_col_name]
            sec_class_lst = df_classified[label].unique().tolist()
            rpt = compareActualPredicted_up(df_classified, label, sec_class_lst, fld, classifier)
            fld += 1
            reports_lst.append(rpt)
    final_rpt = pd.DataFrame()
    for i in range(0, len(reports_lst)):
        rpt = reports_lst[i]
        rpt.reset_index(inplace=True)
        rpt = rpt.rename(columns={'index': 'sec_class'})
        final_rpt = final_rpt.append(rpt)

    ##average across all folds
    print('Avearge Precision and Recall per class across folds--> ')
    avg_res = final_rpt.groupby('sec_class').agg({'precision': np.mean, 'recall': np.mean, 'support': np.mean})
    # avg_res.to_csv(".\\Results\\avg_res.csv", index=None)
    avg_res = avg_res.reset_index()
    avg_res.rename(columns={'index': 'sec_class'}, inplace=True)
    avg_res.to_csv(f".\\RESULTS\\{classifier}_avg_res.csv", index=None)
    print(avg_res)

    print('Overall Precision and Recall--> ')
    # print(final_rpt.groupby('sec_class').agg({'precision': np.mean, 'recall': np.mean}))
    print('Overall precision = ', avg_res['precision'].mean())
    print('Overall recall = ', avg_res['recall'].mean())


##
# python3.9

if __name__ == "__main__":
    main()
