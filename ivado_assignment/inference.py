"""
This Python file is responsible for performing batch inference on new data
observations.
The predictions are saved as .csv in local.
"""

import argparse
import pandas as pd
from pathlib import Path
from joblib import load
from ivado_assignment.settings.data import config

parser = argparse.ArgumentParser(
    prog='IVADA take home assignment',
    description='Takes a batch of unseen observations and store its predictions\
in a file on local.'
)
parser.add_argument('--data', type=str, required=True,
                    help='path to test data csv')


def inference():
    """
    Function responsible for training the ML model and storing the artifact
    binary for later use.
    """
    # load data
    args = parser.parse_args()
    test_df = pd.read_csv(args.data)
    for feat in config.categorical:
        test_df[feat] = pd.Categorical(test_df[feat].astype(str))

    # load & run model
    parent_dir = Path().resolve()
    model = load(f'{parent_dir}/artifacts/models/model.joblib')
    preds = model.predict(test_df[config.categorical + config.numerical])
    preds_proba = model.predict_proba(test_df[config.categorical + config.numerical])[:,1]

    # save preds
    test_df["pred"] = preds
    test_df["pred_proba"] = preds_proba
    test_df[["Unnamed: 0", "pred", "pred_proba"]].to_csv(
        "./artifacts/preds/predictions.csv", index=False)


if __name__ == "__main__":
    inference()