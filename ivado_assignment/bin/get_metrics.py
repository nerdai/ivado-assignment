"""
This file produces the classification metrics report from a supplied labels
as well as predictions data files.
"""

import argparse
import pandas as pd
from ivado_assignment.settings.data import config
from sklearn import metrics

parser = argparse.ArgumentParser(
    prog='IVADO take home assignment',
    description='Takes a batch of predictions and its true labels and produces\
a classification report.'
)
parser.add_argument('--preds', type=str, required=True,
                    help='path to preds data csv')
parser.add_argument('--labels', type=str, required=True,
                    help='path to labels data csv')


def produce_report():
    """
    This function loads user specified labels and preds files and produces
    the sklearn metrics.
    """
    # load data
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)[['Unnamed: 0', 'target']]
    preds = pd.read_csv(args.preds)

    # metrics
    results = pd.merge(preds, labels, on="Unnamed: 0")
    print(metrics.classification_report(results[config.target], results.pred))
    print("Confusion:\n", metrics.confusion_matrix(results[config.target], results.pred), "\n")
    print("ROC-AUC: ", metrics.roc_auc_score(results[config.target], results.pred_proba))
    print("Log loss: ", metrics.log_loss(results[config.target], results.pred_proba))
    print("Brier: ", metrics.brier_score_loss(results[config.target], results.pred_proba))

if __name__ == "__main__":
    produce_report()