import numpy as np
import pandas as pd
from MasterThesis import EDA
from sklearn.metrics import confusion_matrix
import plotly.express as px
from sklearn.metrics import recall_score, precision_score, f1_score
from typing import List


def per_class_accuracy(conf_mt):

    """
    Computes per class accuracy

    Argumetns:
        y_true: 1D numpy array
            true labels
        y_pred: 1D numpy array
            predicted class labels
    """

    acc_by_class = []

    for y in range(conf_mt.shape[0]):

        tp = conf_mt[:, y][y]
        tn = np.delete((conf_mt.sum(axis=1) - conf_mt[:, y]), y).sum()
        acc_by_class.append((tp + tn) / conf_mt.sum())

    return np.array(acc_by_class)


def model_evaluation(conf_mt, class_name, dataset_label="Test"):

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

    # Overall Accuracy
    overall_acc = np.diag(conf_mt).sum() / conf_mt.sum()
    overall_acc = round(overall_acc, 5)

    # Accuracy by Class
    acc_by_class = per_class_accuracy(conf_mt)

    # Precision per class
    precision = np.round(np.diag(conf_mt) / (conf_mt.sum(axis=0) + 1e-8), 5)
    unweighted_precision_avg = round(precision.mean(), 5)
    weighted_precision_avg = round(np.average(precision, weights=conf_mt.sum(axis=1)), 5)
    precision_dict = {f"{name}": round(score, 5) for name, score in zip(class_name, precision)}

    # Recall per class
    recall = np.round(np.diag(conf_mt) / (conf_mt.sum(axis=1) + 1e-8), 5)
    unweighted_recall_avg = round(recall.mean(), 5)
    weighted_recall_avg = round(np.average(recall, weights=conf_mt.sum(axis=1)), 5)
    recall_dict = {f"{name}": round(score, 5) for name, score in zip(class_name, recall)}

    # F1 Score
    f1_score = np.round((2 * precision * recall) / (recall + precision + 1e-8), 5)
    unweighted_f1_score_avg = round(f1_score.mean(), 5)
    weighted_f1_score_avg = round(np.average(f1_score, weights=conf_mt.sum(axis=1)), 5)
    f1_score_dict = {f"{name}": round(score, 5) for name, score in zip(class_name, f1_score)}

    # IoU
    class_iou = np.diag(conf_mt) / (conf_mt.sum(axis=1) + conf_mt.sum(axis=0) - np.diag(conf_mt) + 10e-8)
    unweighted_iou = np.round(np.mean(class_iou), 5)
    weighted_iou = np.round(np.average(class_iou, weights=conf_mt.sum(axis=1)), 5)

    scores = pd.DataFrame(
        {
            f"{dataset_label} Precision": precision,
            f"{dataset_label} Recall": recall,
            f"{dataset_label} F1_score": f1_score,
            f"{dataset_label} IoU": class_iou,
            f"{dataset_label} Acc_by_Class": acc_by_class,
        },
        index=class_name,
    )

    logs = {f"{i}_Precision": j for i, j in zip(scores.index, scores[f"{dataset_label} Precision"])}
    logs.update({f"{i}_Recall": j for i, j in zip(scores.index, scores[f"{dataset_label} Recall"])})
    logs.update({f"{i}_F1_score": j for i, j in zip(scores.index, scores[f"{dataset_label} F1_score"])})
    logs.update({f"{i}_IoU": j for i, j in zip(scores.index, scores[f"{dataset_label} IoU"])})
    logs.update({f"{i}_Acc_by_Class": j for i, j in zip(scores.index, scores[f"{dataset_label} Acc_by_Class"])})

    scores.loc[
        "",
        [
            f"{dataset_label} Precision",
            f"{dataset_label} Recall",
            f"{dataset_label} F1_score",
            f"{dataset_label} IoU",
        ],
    ] = ["", "", "", ""]
    scores.loc[
        "unweighted_Avg",
        [
            f"{dataset_label} Precision",
            f"{dataset_label} Recall",
            f"{dataset_label} F1_score",
            f"{dataset_label} IoU",
        ],
    ] = [
        unweighted_precision_avg,
        unweighted_recall_avg,
        unweighted_f1_score_avg,
        unweighted_iou,
    ]
    scores.loc[
        "weighted Avg",
        [
            f"{dataset_label} Precision",
            f"{dataset_label} Recall",
            f"{dataset_label} F1_score",
            f"{dataset_label} IoU",
        ],
    ] = [
        weighted_precision_avg,
        weighted_recall_avg,
        weighted_f1_score_avg,
        weighted_iou,
    ]

    scores.fillna("", inplace=True)

    logs.update(
        {
            f"{dataset_label} Global Accuracy:": overall_acc,
            f"Unweighted {dataset_label} F1_score": unweighted_f1_score_avg,
            f"Weighted {dataset_label} F1_score": weighted_f1_score_avg,
            f"Unweighted {dataset_label} Precision": unweighted_precision_avg,
            f"Weighted {dataset_label} Precision": weighted_precision_avg,
            f"Unweighted {dataset_label} Recall": unweighted_recall_avg,
            f"Weighted {dataset_label} Recall": weighted_recall_avg,
            f"Unweighted {dataset_label} MIou": unweighted_iou,
            f"Weighted {dataset_label} MIou": weighted_iou,
        }
    )

    return scores, logs


# precision recall curve
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
        print(select_classes)

    def metric_evaluation(self, y_true, y_score):

        result = []

        for y in np.unique(y_true):
            y_pred_proba = y_score[:, y].ravel()
            for t in np.arange(0.00, 1.0, 0.1):

                y_pred = (y_pred_proba >= t).astype(int)

                recall = recall_score((y_true == y) * 1, y_pred, pos_label=1, zero_division=0)

                precision = precision_score((y_true == y) * 1, y_pred, pos_label=1, zero_division=0)

                f1 = f1_score((y_true == y) * 1, y_pred, pos_label=1, zero_division=0)

                result.append(
                    {
                        "recall": recall,
                        "precision": precision,
                        "f1_score": f1,
                        "thresholds": round(t, 3),
                        "class": self.select_classes[y],
                    }
                )

            # if y == 2:
            #    print(
            #        {
            #            "recall": recall,
            #            "precision": precision,
            #            "f1_score": f1,
            #            "thresholds": round(t, 3),
            #            "class": self.select_classes[y],
            #        }
            #    )

        if self.epoch == 0:
            self.result = pd.DataFrame(result)
            self.epoch += 1
        else:
            self.result.recall += pd.DataFrame(result).recall
            self.result.precision += pd.DataFrame(result).precision
            self.result.f1_score += pd.DataFrame(result).f1_score
            self.epoch += 1

        print(self.result)

    def plot_PR_curve(self, color="class"):

        result = self.result.copy()

        if self.epoch != 0:
            result.recall /= self.epoch
            result.precision /= self.epoch
            result.f1_score /= self.epoch

        fig = px.line(
            data_frame=result,
            x="recall",
            y="precision",
            color=color,
            markers=True,
            hover_data=["thresholds"],
        )
        fig.update_layout(yaxis_title=f"Precision", xaxis_title="Recall", font={"size": 14})

        return fig

    def get_bar_plot(self, metric="Recall", color="class"):

        result = self.result.copy()

        if self.epoch != 0:
            result.recall /= self.epoch
            result.precision /= self.epoch
            result.f1_score /= self.epoch

        fig = px.line(data_frame=result, x="thresholds", y=f"{metric}", color=color, markers=True)
        fig.update_layout(yaxis_title=f"{metric}", xaxis_title="Threshold", font={"size": 14})

        return fig

    def get_table(self):

        result = self.result.copy()

        if self.epoch != 0:
            result.recall /= self.epoch
            result.precision /= self.epoch
            result.f1_score /= self.epoch

        return result


def plot_metrics_from_logs(logs, metric="F1_score"):

    """
    Helper function to plot metrics from wandb logs

    Argumetns:
    ----------
        logs: dict
            WandB dictionary with logs
        metric: str:
            one of the follow metrics 'F1_score', 'Recall',
            'Precision' or 'Acc_by_Class'
    """

    assert metric in [
        "F1_score",
        "Recall",
        "Precision",
        "Acc_by_Class",
    ], "The parameter metric must be one of the metrics F1_score, Recall or Precision"

    metric_name = []
    metric_score = []

    for i, j in logs.items():
        if (f"{metric}" in i) and ("weighted" not in i.lower() and (type(j) == float)):
            metric_name.append(i.replace("_" + f"{metric}", ""))
            metric_score.append(j)

    fig = px.bar(x=metric_name, y=metric_score)
    fig.update_layout(xaxis_title="Labels", yaxis_title=f"{metric}", font={"size": 15})

    return fig
