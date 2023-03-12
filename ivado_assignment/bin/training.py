"""
TODO
"""

import argparse
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

    # load data
    train_df = load_and_prep(model_settings[args.setting].train_path)

    # lightweight autoML
    best_model = None
    running_score = 0
    for clf in model_settings[args.setting].classifiers:
        model = Pipeline(steps=[('preprocessor', model_settings[args.setting].preprocessing),
                                ('clf', clf)])
        bayes = BayesSearchCV(model,
                              search_spaces=model_settings[args.setting].hyperparams,
                              scoring=model_settings[args.setting].model_selection_critiera,
                              n_iter=20, cv=4)
        bayes.fit(train_df[config['categorical'] + config['numerical']],
                  train_df[config['target']])
        print(
            f"clf: {clf.__class__.__name__}, \
best_score: {bayes.best_score_}, \
best_params: {bayes.best_params_}")
        if running_score < bayes.best_score_:
            running_score = bayes.best_score_
            best_model = bayes.best_estimator_
    print(best_model.named_steps.clf.__class__.__name__)

    # metrics
    preds = best_model.predict(
        train_df[config['categorical'] + config['numerical']])
    print(classification_report(train_df[config['target']], preds))

    # save model
    parent_dir = Path().resolve()
    dump(best_model,
         f'{parent_dir}/artifacts/models/{model_settings[args.setting].name}\
-model.joblib')


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    train()
