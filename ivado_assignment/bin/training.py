"""
Module for performing lightweight autoML. That is automatic testing of
candidate models and their hyperparameter tuning. The best model is stored in
local.
"""

import argparse
import logging
import warnings
from pathlib import Path
from joblib import dump
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from ivado_assignment.settings.models import model_settings
from ivado_assignment.utils.data_loader import load_and_prep
from ivado_assignment.settings.data import config

parser = argparse.ArgumentParser(
    prog='IVADA take home assignment',
    description='Trains the model according to the configured settings.'
)
parser.add_argument('--setting', type=str, required=True,
                    choices=['complete', 'imputed'],
                    help='setting of complete-case or imputed analysis')


def train():
    """
    Function responsible for training the ML model and storing the artifact
    binary for later use.
    """
    args = parser.parse_args()
    setting = model_settings[args.setting]
    logging.basicConfig(
        filename=f'./artifacts/training_logs/{setting.name}.log',
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    logger = (logging
              .getLogger(f'lightweight autoML [{setting.name}]')
              )

    # load data
    train_df = load_and_prep(setting.train_path)
    logger.info("loaded training data with shape: %s", train_df.shape)

    # lightweight autoML
    best_model = None
    running_score = 0
    logger.info("beginning model selection")
    for clf, hyperparams in setting.classifiers_and_hyperparms:
        model = Pipeline(
            steps=[
                ('preprocessor', setting.preprocessing),
                ('clf', clf)
            ]
        )
        bayes = BayesSearchCV(
            model,
            search_spaces=hyperparams,
            scoring=setting.model_selection_critiera,
            n_iter=20, cv=4
        )
        bayes.fit(train_df[config['categorical'] + config['numerical']],
                  train_df[config['target']])
        logger.info(
            "clf: %s, best_score: %s, best_params: %s",
            clf.__class__.__name__,
            bayes.best_score_,
            bayes.best_params_
        )
        if running_score < bayes.best_score_:
            running_score = bayes.best_score_
            best_model = bayes.best_estimator_
    logger.info(
        "best model: %s",
        best_model.named_steps.clf.__class__.__name__
    )

    # metrics
    preds = best_model.predict(
        train_df[config['categorical'] + config['numerical']])
    logger.info(
        "\n%s",
        classification_report(train_df[config['target']], preds)
    )

    # save model
    parent_dir = Path().resolve()
    dump(best_model,
         f'{parent_dir}/artifacts/models/{setting.name}\
-model.joblib')
    logger.info("successfully saved best model")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    train()
