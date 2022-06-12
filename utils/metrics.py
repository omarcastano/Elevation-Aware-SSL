import numpy as np
import pandas as pd
from MasterThesis import EDA
from sklearn.metrics import confusion_matrix
import plotly.express as px

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


def per_class_accuracy(conf_mt):

    """
    Computes per class accuracy

    Argumetns:
        y_true: 1D numpy array
            true labels
        y_pred: 1D numpy array
            predicted class labels
    """

    acc = lambda tn, fp, fn, tp:(tp+tn)/(tp+fp+tn+fn+10e-8)
    
    acc_by_class = []

    for y in range(conf_mt.shape[0]):
    
        tp = conf_mt[:,y][y]
        tn = np.delete((conf_mt.sum(axis=1) - conf_mt[:,y]), y).sum()
        acc_by_class.append((tp+tn)/conf_mt.sum())

    return np.array(acc_by_class)

#threshold dependence
def threshold_metric_evaluation(y_true, y_score, metric='Recall'):

    """
    Plot the value of a given metric for several probability threshold. 

    Argumetns:
        y_true: 3D numpy array (N,W,H)
            Batch of N sample with the true labels
        y_score: 4D numpy array (N,C,W,H)
            Batch of N samples with predicted scores 
        metric: string
            one of the metric from the following list
            Accuracy, Precision, Recall, f1_score, FPR
            FNR, NPV, TNR
        threshold: Threshold for the probability
    """


    metrics_dict = {'Accuracy':lambda tn, fp, fn, tp:(tp+tn)/(tp+fp+tn+fn+10e-8), 
                    'Precision':lambda tn, fp, fn, tp:(tp)/(tp+fp+10e-8), 
                    'Recall':lambda tn, fp, fn, tp:(tp)/(tp+fn+10e-8),
                    'f1_score': lambda tn, fp, fn, tp:(2*((tp)/(tp+fp+10e-8))*((tp)/(tp+fn+10e-8)))/(((tp)/(tp+fp+10e-8)) + ((tp)/(tp+fn+10e-8))),
                    'FPR': lambda tn, fp, fn, tp: fp/(fp+tn+10e-8),
                    'FNR': lambda tn, fp, fn, tp: fn/(fn+tp+10e-8),
                    'NPV': lambda tn, fp, fn, tp: tn/(tn+fn+10e-8),
                    'TNR': lambda tn, fp, fn, tp: tn/(tn+fp+10e-8)
                    }

    metrics = []
    thresholds = []
    result = []

    y_true = y_true.ravel()

    for y in np.unique(y_true):
        y_pred_proba = y_score[:,y,:,:].ravel()

        for t in np.arange(0.01,0.99,0.01):

            y_pred = (y_pred_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix((y_true==y)*1, y_pred).ravel()
            metrics = metrics_dict[metric](tn, fp, fn, tp)
            result.append({'metrics':metrics, 'thresholds':t, 'class':y})

    result = pd.DataFrame(result)
    fig = px.line(data_frame=result, x='thresholds', y='metrics', color='class', markers=True)
    fig.update_layout(
        yaxis_title=f'{metric}',
        xaxis_title='Threshold',
        font={'size':14}
        )

    return fig

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
    acc_by_class = per_class_accuracy(conf_mt)

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
    logs.update({f'{i}_Acc_by_Class':j for i, j in zip(scores.index, scores[f'{dataset_label} Acc_by_Class'])})



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


def plot_metrics_from_logs(logs, metric='F1_score'):

    """
    Helper function to plot metrics from wandb logs

    Argumetns:
        logs: dict
            WandB dictionary with logs
        metric: str:
            on of the follow metrics 'F1_score', 'Recall', 
            'Precision' or 'Acc_by_Class'
    """
  
    assert metric in ['F1_score', 'Recall', 'Precision', 'Acc_by_Class'], 'The parameter metric must be one of the metrics F1_score, Recall or Precision'

    metric_name = []
    metric_score = []

    for i, j in logs.items():
        if (f'{metric}' in i) and ('weighted' not in i.lower()):
            metric_name.append(i.replace('_'+ f'{metric}', ''))
            metric_score.append(j)

        
    fig = px.bar(x=metric_name, y=metric_score)
    fig.update_layout(
        xaxis_title = 'Labels',
        yaxis_title = f'{metric}',
        font={"size":15}
    )
    
    return fig
