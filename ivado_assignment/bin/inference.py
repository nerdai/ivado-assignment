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
from ivado_assignment.settings.models import model_settings
from ivado_assignment.utils.data_loader import load_and_prep

parser = argparse.ArgumentParser(
    prog='IVADA take home assignment',
    description='Takes a batch of unseen observations and store its predictions\
in a file on local.'
)
parser.add_argument('--data', type=str, required=True,
                    help='path to test data csv')
parser.add_argument('--setting', type=str, required=True,
                    choices=['complete', 'imputed'],
                    help='setting of complete-case or imputed analysis')


def inference():
    """
    Function responsible for training the ML model and storing the artifact
    binary for later use.
    """
    # load data
    args = parser.parse_args()
    test_df = load_and_prep(args.data)

    # load & run model
    parent_dir = Path().resolve()
    model = load(f'{parent_dir}/artifacts/models/{model_settings[args.setting].name}-model.joblib')
    preds = model.predict(test_df[config.categorical + config.numerical])
    preds_proba = model.predict_proba(
        test_df[config.categorical + config.numerical])[:, 1]

    # save preds
    test_df["pred"] = preds
    test_df["pred_proba"] = preds_proba
    test_df[[config.id_col] + ["pred", "pred_proba"]].to_csv(
        f"./artifacts/preds/{model_settings[args.setting].name}-predictions.csv", index=False)


if __name__ == "__main__":
    inference()
