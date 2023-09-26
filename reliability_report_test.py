import pandas as pd
import numpy as np
import os
import warnings
from sklearn import metrics

def get_results(y_act, y_pred, labels, classifier_name, toy=False):
        if toy:
            y_act = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
            y_pred = [1, 1, 3, 2, 2, 1, 3, 3, 3, 3, 1, 2, 3]
            report = metrics.classification_report(y_act, y_pred, output_dict=True, target_names=[1, 2, 3], labels=[1, 2, 3])
            #confusion_matrix = metrics.confusion_matrix(y_act, y_pred, labels=[1, 2, 3])
            ml_confusion_matrix = metrics.multilabel_confusion_matrix(y_act, y_pred, labels=[1, 2, 3])
        else:
            report = metrics.classification_report(y_act, y_pred, output_dict=True, labels=labels)
            #confusion_matrix = metrics.confusion_matrix(y_act, y_pred, labels=labels)
            ml_confusion_matrix = metrics.multilabel_confusion_matrix(y_act, y_pred, labels=labels)
        df = pd.DataFrame(report).transpose()
        drop_cols = ['micro avg', 'macro avg', 'weighted avg', 'accuracy']
        for col in drop_cols:
            if col in df.index:
                df = df.drop([col])
        report = ci_report(ml_confusion_matrix, df)
        if toy:
            report['recall_score'] = metrics.recall_score(y_act, y_pred, labels=[1,2,3], average=None)
        else:
            report['recall_score'] = metrics.recall_score(y_act, y_pred, labels=labels, average=None)
        return report

def ci_report(confusion_matrix, df):
    # # ---- Get confidence intervals using confusion matrix ---- #
    FN = confusion_matrix[:, 1, 0]
    FP = confusion_matrix[:, 0, 1]
    TP = confusion_matrix[:, 1, 1]
    #FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    #FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    #TP = np.diag(confusion_matrix)
    # TN = confusion_matrix.values.sum() - (FP + FN + TP)
    class_precisions = []
    class_recalls = []
    for i in range(TP.shape[0]):
        class_precisions.append(ci(TP[i], TP[i]+FP[i]))
        class_recalls.append(ci(TP[i], TP[i]+FN[i]))
    df['true_positives'] = TP
    df['false_positives'] = FP
    df['false_negative'] = FN
    df['precision_cnfs_mat'] = TP/(TP+FP)
    df['recal_cnfs_mat'] = TP/(TP+FN)
    df['ci_precision'] = class_precisions
    df['ci_recall'] = class_recalls
    df['mean_precision'] = df['precision'].mean()
    df['mean_recall'] = df['recall'].mean()
    # calculate means without low samples classes 
    df_ab50 = df[df['support'] >= 50]
    df_bl50 = df[df['support'] < 50]
    df['mean_precision_50+'] = df_ab50['precision'].mean()
    df['mean_recall_50+'] = df_ab50['recall'].mean()
    df['mean_precision_49-'] = df_bl50['precision'].mean()
    df['mean_recall_49-'] = df_bl50['recall'].mean()
    return df

from scipy.stats import norm
def ci(tp, n, alpha=0.05):
    """ Estimates confidence interval for Bernoulli p
        Args:
            tp: number of positive outcomes, TP in this case
            n: number of attempts, TP+FP for Precision, TP+FN for Recall
            alpha: confidence level
        Returns:
            Tuple[float, float]: lower and upper bounds of the confidence interval
    """
    p_hat = float(tp) / n
    z_score = norm.isf(alpha * 0.5)  # two sides, so alpha/2 on each side
    variance_of_sum = p_hat * (1-p_hat) / n
    std = variance_of_sum ** 0.5
    return p_hat - z_score * std, p_hat + z_score * std


def main():
    TOY = False
    print('Start...')
    
    data = pd.read_csv(".//RESULTS//Final_File_With_Integrated_Classifications_3_20_2023_classified.csv")
    
    out = data[data['RegexPrediction'] == 'VZID and EID']
    print(out)
    

    y_act_col = 'business_term'
    class_list = data[y_act_col].unique().tolist()
    print(len(class_list))
    #print(f'class list: {class_list}')
    #'RegexPrediction': i''
    models_dict = {'RegexPrediction': '', 'NB_TOP_CLASS': 'NB_CONFIDENCE', 'NNBR_TOP_CLASS': 'NNBR_CONFIDENCE', 'SYSTEM_PREDICTION': 'SYSTEM_CONF'}
    
    #models_dict = {'SYSTEM_PREDICTION': 'SYSTEM_CONF'}
    for model_pred_col, model_conf_col in models_dict.items():
        model_name = model_pred_col.split("_")[0]
        if TOY:
            report = get_results(model_y_act, model_y_pred, class_list, model_name, toy=True)
        else:
            model_y_act = data[y_act_col].values
            th_lst = 0.95
            if model_name == 'BM25':
                th_lst = 7
            if (model_name != 'SYSTEM') and (model_name != 'RegexPrediction'):
                data = data.replace({model_pred_col: {np.nan: "-1"}})
                #data['y_pred'] = data.apply(lambda x: -1 if x[model_pred_col] == np.nan else x[model_pred_col], axis=1)
                model_y_pred = data.apply(lambda x: '-1' if x[model_conf_col] < th_lst else x[model_pred_col], axis=1)
                model_y_pred = model_y_pred.values
            else:
                model_y_pred = data[model_pred_col].values
            print(f'Y_actual: {pd.unique(model_y_act)}')
            print(f'y_predicted: {pd.unique(model_y_pred)}')
            report = get_results(model_y_act, model_y_pred, class_list, model_pred_col.split("_")[0])         
        report = report.reset_index()
        report.rename(columns={'index': 'sec_class'}, inplace=True)
        if TOY == True:
            path = rf"reliability_results/{model_name}_toy_res.csv"
        else:
            path = rf"reliability_results/{model_name}_res.csv"
        report.to_csv(path, index=False)
        if model_name == 'SYSTEM':
            print(report)
            print('Overall Precision and Recall--> ')
            # print(final_rpt.groupby('sec_class').agg({'precision': np.mean, 'recall': np.mean}))
            print('Overall precision = ', report['precision'].mean())
            print('Overall recall = ', report['recall'].mean())

if __name__ == "__main__":
        main()

