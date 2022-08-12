import numpy as np
import pandas as pd
import plotly.express as px
from typing import List
import wandb


def lineplot_metrics_from_wandb(
    wandb,
    project="MasterThesis",
    entity="omar_castno",
    version=["baseline"],
    train_size=[0.02],
    metric="F1_score",
):

    """
    Returns a linplot where x-axis is the amount of training data and the y-axis
    is the mean score for a given metric across different runs (corss-validation).

    Argumetns:
        wandb: wandb module
        project: wandb project name
        entity: name of the wandb entity
        version: identifiers to filter wandb runs. This parametere is used to
            select the wandb runs that will be use to in the final linplot. This parameter is
            also used to group and aggregate results.
        train_size: train size used to train the model. This parametere is used to
            select the wandb runs that will be use to in the final lineplot
        metric: one of the follow metrics 'F1_score', 'Recall', 'Precision' or 'Acc_by_Class'
        line_plot: If true
    """

    assert metric in [
        "F1_score",
        "Recall",
        "Precision",
        "Acc_by_Class",
    ], "The parameter metric must be one of the metrics F1_score, Recall or Precision"

    api = wandb.Api()
    runs = api.runs(entity + "/" + project)

    summary_runs = []
    for run in runs:
        summary = run.summary._json_dict
        summary.update(run.config)
        summary.update({"id": run.id})
        summary_runs.append(summary)

    metric_name = []
    metric_score = []
    metric_version = []
    metric_train_size = []

    for logs in summary_runs:
        if (logs["version"] in version) and (logs["amount_of_ft_data"] in train_size):
            for i, j in logs.items():
                if (f"{metric}" in i) and (
                    "weighted" not in i.lower() and isinstance(j, float)
                ):
                    metric_name.append(i.replace("_" + f"{metric}", ""))
                    metric_score.append(j)
                    metric_version.append(logs["version"])
                    metric_train_size.append(logs["amount_of_ft_data"])

    df = pd.DataFrame(
        {
            "metrics": metric_score,
            "class": metric_name,
            "version": metric_version,
            "train_size": np.array(metric_train_size) * 100,
        }
    )

    df = df.groupby(["version", "class", "train_size"], as_index=False).agg(
        mean=("metrics", "mean"), std=("metrics", "std")
    )
    fig = px.line(
        data_frame=df,
        y="mean",
        x="train_size",
        color="class",
        line_dash="version",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Amount of Labeled Data (%)",
        yaxis_title=f"{metric}",
        font={"size": 15},
    )

    return fig


def barplot_metrics_from_wandb(
    wandb,
    project: str = "MasterThesis",
    entity: str = "omar_castno",
    version: List[str] = ["baseline"],
    train_size: List[float] = [0.02],
    metric: str = "F1_score",
    return_table: bool = False,
):

    """
    Returns a barplot with the mean score for a given metric across several runs.

    Argumetns:
        wandb: wandb module
        project: wandb project name
        entity: name of the wandb entity
        version: identifiers to filter wandb runs. This parametere is used to
            select the wandb runs that will be use to in the final barplot. This parameter is
            also used to group and aggregate results.
        train_size: train size used to train the model. This parametere is used to
            select the wandb runs that will be use to in the final barplot
        metric: one of the follow metrics 'F1_score', 'Recall', 'Precision' or 'Acc_by_Class'
        return_table: If true returns the table used in the barplot
    """

    assert metric in [
        "F1_score",
        "Recall",
        "Precision",
        "Acc_by_Class",
    ], "The parameter metric must be one of the metrics F1_score, Recall or Precision"

    api = wandb.Api()
    runs = api.runs(entity + "/" + project)

    summary_runs = []
    for run in runs:
        summary = run.summary._json_dict
        summary.update(run.config)
        summary.update({"id": run.id})
        summary_runs.append(summary)

    metric_name = []
    metric_score = []
    metric_version = []
    metric_train_size = []

    for logs in summary_runs:
        if (logs["version"] in version) and (logs["amount_of_ft_data"] in train_size):
            for i, j in logs.items():
                if (f"{metric}" in i) and (
                    "weighted" not in i.lower() and isinstance(j, float)
                ):
                    metric_name.append(i.replace("_" + f"{metric}", ""))
                    metric_score.append(j)
                    metric_version.append(logs["version"])
                    metric_train_size.append(logs["amount_of_ft_data"])

    df = (
        pd.DataFrame(
            {
                "metrics": metric_score,
                "class": metric_name,
                "version": metric_version,
                "train_size": np.array(metric_train_size) * 100,
            }
        )
        .astype({"train_size": str})
        .assign(train_size=lambda x: x["train_size"] + "%")
    )

    color = "train_size" if len(version) == 1 else "version"

    df = df.groupby(["class", color], as_index=False).agg(
        mean=("metrics", "mean"), std=("metrics", "std")
    )

    fig = px.bar(
        data_frame=df,
        y="mean",
        x="class",
        error_y="std",
        color=color,
        barmode="group",
    )

    fig.update_layout(xaxis_title="Labels", yaxis_title=f"{metric}", font={"size": 15})

    if return_table:
        return df

    return fig


def get_table(
    wandb: wandb,
    table_name: str = "Table_Metrics",
    project: str = "MasterThesis",
    entity: str = "omar_castno",
    version: List[str] = "baseline",
    train_size: List[float] = 0.02,
):

    """
    Helper funtion to get tables which are stored as wandb artifacts


    Argumetns:
        wandb: wandb module
        table_name: Table artifact name
        project: wandb project name
        entity: name of the wandb entity
        version: version of the run
        train_size: amount of data used to train the model
    """
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)

    summary_runs = []
    for run in runs:
        summary = run.summary._json_dict
        summary.update(run.config)
        summary.update({"id": run.id})
        summary_runs.append(summary)

    run = wandb.init()

    result = []

    for vs, ts in zip(version, train_size):

        precision = 0
        recall = 0
        f1_score = 0
        train_loss = 0
        test_loss = 0

        ids = [
            j["id"]
            for j in summary_runs
            if (j["version"] == vs) and (j["amount_of_ft_data"] == ts)
        ]

        for n, id in enumerate(ids, 1):
            my_table = run.use_artifact(
                f"{entity}/{project}/run-{id}-{table_name}:v0"
            ).get(table_name)
            my_table = pd.DataFrame(data=my_table.data, columns=my_table.columns)

            if table_name == "Loss":
                train_loss += my_table.train_loss
                test_loss += my_table.test_loss
            else:
                precision += my_table.precision
                recall += my_table.recall
                f1_score += my_table.f1_score

        print(
            "-------------------------------------------------------------------------------------"
        )
        print(f"The number of runs found with train_size={ts} and version={vs} is {n}")
        print(
            "-------------------------------------------------------------------------------------"
        )

        if table_name == "Loss":
            my_table["train_loss"] = train_loss / n
            my_table["test_loss"] = test_loss / n
        else:
            my_table["recall"] = recall / n
            my_table["precision"] = precision / n
            my_table["f1_score"] = f1_score / n

        my_table["version"] = vs
        my_table["train_size"] = ts

        result.append(my_table)
        del my_table

    wandb.finish()

    if table_name == "Loss":
        return pd.concat(result)
    else:
        return pd.concat(result).rename({"class": "label"}, axis=1)


def plot_pr_curve(table: pd.DataFrame, color: str = None, line_dash: str = None):

    """
    Return precision recall curve

    Argumetns:
    ----------
        table: dataframe with metrics
        color: Values from this column are used to assign color to marks
        line_dash: values from this column are used to assing line style

    """

    fig = px.line(
        data_frame=table,
        x="recall",
        y="precision",
        color=color,
        line_dash=line_dash,
        markers=True,
        hover_data=["thresholds"],
    )

    fig.update_layout(yaxis_title=f"Precision", xaxis_title="Recall", font={"size": 14})

    return fig


def plot_loss_curves(table, color, line_dash):

    """
    Return precision recall curve

    Argumetns:
    ----------
        table: dataframe with metrics
        color: Values from this column are used to assign color to marks
        line_dash: values from this column are used to assing line style

    """

    fig = px.line(
        data_frame=table,
        x="epoch",
        y=["train_loss", "test_loss"],
        color=color,
        line_dash=line_dash,
        markers=True,
        hover_data={"version": False, "train_size": False, "epoch": False},
    )

    fig.update_layout(
        yaxis_title="Score", xaxis_title="Epoch", font={"size": 14}, hovermode="x"
    )

    return fig


def plot_metrics_by_threshold(table, metric="recall", color=None, line_dash=None):

    """
    Return precision recall curve

    Argumetns:
    ----------
        table: dataframe with metrics
        metric: metric that will be plot
        color: Values from this column are used to assign color to marks
        line_dash: values from this column are used to assing line style

    """

    fig = px.line(
        data_frame=table,
        x="thresholds",
        y=f"{metric}",
        color=color,
        line_dash=line_dash,
        markers=True,
    )

    fig.update_layout(
        yaxis_title=f"{metric}", xaxis_title="Threshold", font={"size": 14}
    )

    return fig
