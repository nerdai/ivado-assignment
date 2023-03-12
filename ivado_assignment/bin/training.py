"""
TODO
"""

from pathlib import Path
from joblib import dump
import pandas as pd
from ivado_assignment.settings.data import config
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from ivado_assignment.settings.models import model_setting_1
from ivado_assignment.utils.data_loader import data_loader
import warnings


def train():
    # load data
    train = data_loader("./data/splits/incomplete_df/train.csv")

    # lightweight autoML
    best_model = None
    running_score = 0
    for clf in model_setting_1.classifiers:
        model = Pipeline(steps=[('preprocessor', model_setting_1.preprocessing),
                                ('clf', clf)])
        bayes = BayesSearchCV(model,
                              search_spaces=model_setting_1.hyperparams,
                              scoring=model_setting_1.model_selection_critiera,
                              n_iter=20, cv=4)
        bayes.fit(train[config.categorical + config.numerical],
                train[config.target])
        print(f"clf: {clf.__class__.__name__}, best_score: {bayes.best_score_}, best_params: {bayes.best_params_}")
        if running_score < bayes.best_score_:
            running_score = bayes.best_score_
            best_model = bayes.best_estimator_
    print(best_model.named_steps.clf.__class__.__name__)

    # metrics
    preds = best_model.predict(train[config.categorical + config.numerical])
    print(classification_report(train[config.target], preds))

    # save model
    parent_dir = Path().resolve()
    dump(best_model, f'{parent_dir}/artifacts/models/model.joblib')


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    train()
