import numpy as np
import pandas as pd
from MasterThesis import EDA
from sklearn.metrics import confusion_matrix
import plotly.express as px
from sklearn.metrics import recall_score, precision_score, f1_score

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

#precision recall curve
class threshold_metric_evaluation:

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
            select_classes: list
                list with the name of the classes
    """

    def __init__(self, select_classes):
        self.select_classes = select_classes
        self.epoch = 0

    def metric_evaluation(self, y_true, y_score):

        result = []

        y_true = y_true.ravel()

        for y in np.unique(y_true):
            y_pred_proba = y_score[:,y,:,:].ravel()

            for t in np.arange(0.00,0.99,0.1):

                y_pred = (y_pred_proba >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix((y_true==y)*1, y_pred).ravel()
                recall = recall_score((y_true==y)*1, y_pred, pos_label=1, zero_division=0)
                precision = precision_score((y_true==y)*1, y_pred, pos_label=1, zero_division=0)
                f1 = f1_score((y_true==y)*1, y_pred, pos_label=1, zero_division=0)
                result.append({'recall':recall, 'precision':precision, 'f1_score':f1,
                               'thresholds':round(t, 3), 'class':self.select_classes[y]})
                
        if self.epoch == 0:
            self.result = pd.DataFrame(result)
            self.epoch += 1
        else:
            self.result.recall += pd.DataFrame(result).recall
            self.result.precision += pd.DataFrame(result).precision
            self.result.f1_score += pd.DataFrame(result).f1_score
            self.epoch += 1


    def plot_PR_curve(self, color='class'):

        result = self.result.copy()

        if self.epoch != 0:
            result.recall /= self.epoch
            result.precision /= self.epoch
            result.f1_score /= self.epoch

        fig = px.line(data_frame=result, x='recall', y='precision', color=color, markers=True, hover_data=['thresholds'])
        fig.update_layout(
            yaxis_title=f'Precision',
            xaxis_title='Recall',
            font={'size':14})
        
        return fig
    

    def get_bar_plot(self, metric='Recall', color='class'):

        result = self.result.copy()

        if self.epoch != 0:
            result.recall /= self.epoch
            result.precision /= self.epoch
            result.f1_score /= self.epoch

        fig = px.line(data_frame=result, x='thresholds', y=f"{metric}", color=color, markers=True)
        fig.update_layout(
            yaxis_title=f'{metric}',
            xaxis_title='Threshold',
            font={'size':14})

        return fig

    def get_table(self):

        result = self.result.copy()

        if self.epoch != 0:
            result.recall /= self.epoch
            result.precision /= self.epoch
            result.f1_score /= self.epoch

        return result


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
        if (f'{metric}' in i) and ('weighted' not in i.lower() and (type(j)==float)):
            metric_name.append(i.replace('_'+ f'{metric}', ''))
            metric_score.append(j)

        
    fig = px.bar(x=metric_name, y=metric_score)
    fig.update_layout(
        xaxis_title = 'Labels',
        yaxis_title = f'{metric}',
        font={"size":15}
    )
    
    return fig

def plot_metrics_from_wandb(wandb, project='MasterThesis', entity='omar_castno', version=['baseline'], ft_data=[0.02], metric='F1_score'):

    """
    Helper function to plot metrics from several runs stored in wandb

    Argumetns:
        wandb: wandb module
        project: str
            wandb project name
        entity: str
            name of the wandb entity
        version: list[str]
            identifier to filter wandb runs
        metric: str
            on of the follow metrics 'F1_score', 'Recall', 
            'Precision' or 'Acc_by_Class'
    """
  
    assert metric in ['F1_score', 'Recall', 'Precision', 'Acc_by_Class'], 'The parameter metric must be one of the metrics F1_score, Recall or Precision'

    api = wandb.Api()
    runs = api.runs(entity + "/" + project) 

    summary_runs = []
    for run in runs:
        summary = run.summary._json_dict
        summary.update(run.config)
        summary.update({'id':run.id})
        summary_runs.append(summary)


    metric_name = []
    metric_score = []
    metric_version = []
    for logs in summary_runs:
        if (logs['version'] in version) and (logs['amount_of_ft_data'] in ft_data):
            for i, j in logs.items():
                if (f'{metric}' in i) and ('weighted' not in i.lower() and (type(j)==float)):
                    metric_name.append(i.replace('_'+ f'{metric}', ''))
                    metric_score.append(j)
                    metric_version.append(logs['version'])

    df = pd.DataFrame({'metrics':metric_score, 'class':metric_name, 'version':metric_version})
    df_1 = df.copy()
    df = df.groupby(['class', 'version'], as_index=False).agg(mean=('metrics', 'mean'), std=('metrics', 'std'))
        
    fig = px.bar(data_frame=df, y='mean', x='class', error_y='std', color='version', barmode='group')
    fig.update_layout(
        xaxis_title = 'Labels',
        yaxis_title = f'{metric}',
        font={"size":15}
    )
    
    return fig

def plot_metrics_from_artifacts(wandb, table_name='Table_Metrics', project='MasterThesis', entity='omar_castno', version='baseline', ft_data=0.02):
    
    """
    Helper function to plot metrics from WandB artifacts(Tables)


    Argumetns:
        wandb: wandb module
        table_name: str
            Table artifact name
        project: str
            wandb project name
        entity: str
            name of the wandb entity
        version: str
            identifier to filter wandb runs
    """
    api = wandb.Api()
    runs = api.runs(entity + "/" + project) 

    summary_runs = []
    for run in runs:
        summary = run.summary._json_dict
        summary.update(run.config)
        summary.update({'id':run.id})
        summary_runs.append(summary)    

    run =  wandb.init()
    precision = 0
    recall = 0
    f1_score = 0
    ids = [j['id'] for j in summary_runs if (j['version'] == version) and (j['amount_of_ft_data'] == ft_data)]

    for n, id in enumerate(ids, 1):
        my_table = run.use_artifact(f"omar_castno/MasterThesis/run-{id}-{table_name}:v0").get(table_name)
        my_table = pd.DataFrame(data=my_table.data, columns=my_table.columns)
        precision += my_table.precision
        recall += my_table.recall
        f1_score += my_table.f1_score

    my_table['recall'] = recall/n
    my_table['precision'] = precision/n
    my_table['f1_score'] = f1_score/n
    my_table['version'] = version

    return my_table

def plot_loss_from_artifact(wandb, table_name='Loss', project='MasterThesis', entity='omar_castno', version='baseline'):

    """
    Helper function to plot average train and test loss from different runs


    Argumetns:
        wandb: wandb module
        table_name: str
            Table artifact name
        project: str
            wandb project name
        entity: str
            name of the wandb entity
        version: str
            identifier to filter wandb runs
    """

    api = wandb.Api()
    runs = api.runs(entity + "/" + project) 

    summary_runs = []
    for run in runs:
        summary = run.summary._json_dict
        summary.update(run.config)
        summary.update({'id':run.id})
        summary_runs.append(summary)    

    run =  wandb.init()
    df = pd.DataFrame(columns=['train_loss', 'test_loss'])
    train_loss = 0
    test_loss = 0
    ids = [j['id'] for j in summary_runs if j['version'] == version]

    for n, id in enumerate(ids, 1):
        my_table = run.use_artifact(f"omar_castno/MasterThesis/run-{id}-{table_name}:v0").get(table_name)
        my_table = pd.DataFrame(data=my_table.data, columns=my_table.columns)
        df = df.append(my_table, ignore_index=False)
        train_loss += my_table.train_loss
        test_loss += my_table.test_loss

    df.reset_index(inplace=True)
    df['index'] += 1
    df.rename(columns={'index':'epoch'}, inplace = True)
    df = df.groupby('epoch', as_index=False).agg(train_loss=('train_loss','mean'),
                                                train_loss_std=('train_loss','std'),
                                                test_loss=('test_loss','mean'),
                                                test_loss_std=('test_loss','std'))


    fig = px.line(data_frame=df, x='epoch' ,y=['train_loss', 'test_loss'], markers=True)
    fig.update_layout(
            yaxis_title='Score',
            xaxis_title='Epoch',
            font={'size':14})

    return fig