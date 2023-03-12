"""
This file produces the classification report from a supplied labels
as well as predictions data files. It also outputs in the console other 
suchs log-loss, Brier's score, F1 Score, and ROC-AUC Score.
"""

import argparse
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    log_loss
)
from ivado_assignment.settings.data import config

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
    the sklearn 
    """
    # load data
    args = parser.parse_args()
    labels = pd.read_csv(args.labels)[[config['id_col'], config['target']]]
    preds = pd.read_csv(args.preds)

    # metrics
    results = pd.merge(preds, labels, on=config['id_col'])
    print(classification_report(
        results[config['target']], results.pred))
    print("Confusion:\n", confusion_matrix(
        results[config['target']], results.pred), "\n")
    print("ROC-AUC: ",
          roc_auc_score(results[config['target']], results.pred_proba))
    print("Log loss: ", log_loss(
        results[config['target']], results.pred_proba))
    print("Brier: ", brier_score_loss(
        results[config['target']], results.pred_proba))
    print("F1 Score:", f1_score(
        results[config['target']], results.pred))


if __name__ == "__main__":
    produce_report()
