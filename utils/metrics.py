import numpy as np
import pandas as pd
from MasterThesis import EDA
from sklearn.metrics import confusion_matrix

## Confusion matrix
def pixel_confusion_matrix(y_true, y_pred, class_num):

    """
    Computes confunsion matrix at a pixel level 
    
    Argumetns:
        y_true: 2D numpy array:
            True labels
        y_pred: 2D numpy array
            Predicted labels
        class_num: int
            number of classes
    """

    label_class = [x for x in range(class_num)]

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    conf_mt = confusion_matrix(y_true, y_pred, labels=label_class)

    return conf_mt
  
  
def class_IoU(y_true, y_pred, num_classes):

    """
    Computes Intersection Over Union

    Argumetns:
        y_true: 2D numpy array:
            True labels
        y_pred: 2D numpy array
            Predicted labels
        class_num: int
            number of classes
    """

    conf_mt = confusion_matrix(y_true, y_pred, class_num=num_classes)
    class_iou = np.diag(conf_mt)/(conf_mt.sum(axis=1) + conf_mt.sum(axis=0) - np.diag(conf_mt) + 10e-8)

    return class_iou


def model_evaluation(conf_mt, class_name, dataset_label='Test'):

    """
    Computes different classification metricos at pixel level

    Argumetns:
        conf_mt: 2D numpy array
            confusion matrix
        class_name: list
            name for each class
        dataset_label: str. Options 'Train' or 'Test'
            name to identify the train or test dataset
    """

    #Overall Accuracy
    overall_acc = np.diag(conf_mt).sum()/conf_mt.sum()
    overall_acc = round(overall_acc, 5)

    #Accuracy by Class
    acc_by_class = (conf_mt.diagonal()/conf_mt.sum()).round(5)

    #Precision per class
    precision = np.round(np.diag(conf_mt)/(conf_mt.sum(axis=0)+1e-8),5)
    unweighted_precision_avg = round(precision.mean(),5)
    weighted_precision_avg = round(np.average(precision, weights=conf_mt.sum(axis=1)) ,5)
    precision_dict = {f'{name}':round(score,5) for name, score in zip(class_name, precision)}

    #Recall per class
    recall = np.round(np.diag(conf_mt)/(conf_mt.sum(axis=1)+1e-8),5)
    unweighted_recall_avg = round(recall.mean(),5)
    weighted_recall_avg = round(np.average(recall, weights=conf_mt.sum(axis=1)) ,5)
    recall_dict = {f'{name}':round(score,5) for name, score in zip(class_name, recall)}

    #F1 Score
    f1_score = np.round((2*precision*recall)/(recall+precision+1e-8),5)
    unweighted_f1_score_avg = round(f1_score.mean(),5)
    weighted_f1_score_avg = round(np.average(f1_score, weights=conf_mt.sum(axis=1)) ,5)
    f1_score_dict = {f'{name}':round(score,5) for name, score in zip(class_name, f1_score)}


    #IoU
    class_iou = np.diag(conf_mt)/(conf_mt.sum(axis=1) + conf_mt.sum(axis=0) - np.diag(conf_mt) + 10e-8)
    unweighted_iou = np.round(np.mean(class_iou), 5)
    weighted_iou = np.round(np.average(class_iou, weights=conf_mt.sum(axis=1)), 5)

    scores = pd.DataFrame({f'{dataset_label} Precision':precision, f"{dataset_label} Recall":recall, 
                                                f"{dataset_label} F1_score":f1_score, f"{dataset_label} IoU":class_iou,
                                                f"{dataset_label} Acc_by_Class":acc_by_class}, index=class_name)


    logs = {f'{i}_Precision':j for i, j in zip(scores.index, scores[f'{dataset_label} Precision'])}
    logs.update({f'{i}_Recall':j for i, j in zip(scores.index, scores[f'{dataset_label} Recall'])})
    logs.update({f'{i}_F1_score':j for i, j in zip(scores.index, scores[f'{dataset_label} F1_score'])})
    logs.update({f'{i}_IoU':j for i, j in zip(scores.index, scores[f'{dataset_label} IoU'])})



    scores.loc["", [f'{dataset_label} Precision', f'{dataset_label} Recall', f'{dataset_label} F1_score', f'{dataset_label} IoU']] = ["", "","",""]
    scores.loc['unweighted_Avg', [f'{dataset_label} Precision', f'{dataset_label} Recall', f'{dataset_label} F1_score', f'{dataset_label} IoU']] = \
                                                                                        [unweighted_precision_avg, unweighted_recall_avg,  unweighted_f1_score_avg, unweighted_iou]

    scores.loc['weighted Avg', [f'{dataset_label} Precision', f'{dataset_label} Recall', f'{dataset_label} F1_score', f'{dataset_label} IoU']] = \
                                                                                                  [weighted_precision_avg, weighted_recall_avg, weighted_f1_score_avg, weighted_iou]                                                                                                              
    scores.fillna("", inplace=True)

    logs.update({f'{dataset_label} Global Accuracy:': overall_acc, 
                 f'Unweighted {dataset_label} F1_score':unweighted_f1_score_avg,  f'Weighted {dataset_label} F1_score':weighted_f1_score_avg,
                 f'Unweighted {dataset_label} Precision':unweighted_precision_avg,  f'Weighted {dataset_label} Precision':weighted_precision_avg,
                 f'Unweighted {dataset_label} Recall':unweighted_recall_avg,  f'Weighted {dataset_label} Recall':weighted_recall_avg,
                 f'Unweighted {dataset_label} MIou':unweighted_iou, f'Weighted {dataset_label} MIou':weighted_iou})

    return scores, logs
