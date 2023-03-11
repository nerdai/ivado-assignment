"""
TODO
"""

from pathlib import Path
from joblib import dump
import pandas as pd
from ivado_assignment.config import config
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from skopt import BayesSearchCV
from ivado_assignment.models import model_setting_1


def train():

    # load data
    train = pd.read_csv("./data/splits/incomplete_df/train.csv")
    for feat in config.categorical:
        train[feat] = pd.Categorical(train[feat].astype(str))

    model = Pipeline(steps=[('preprocessor', model_setting_1.preprocessing),
                            ('clf', model_setting_1.models_to_test[0])])

    parameters = {
        "clf__n_estimators": [25, 50, 100]
    }

    bayes = BayesSearchCV(model,
                        search_spaces=parameters,
                        scoring=metrics.make_scorer(metrics.f1_score, pos_label=0),
                        n_iter=20, cv=4)

    bayes.fit(train[config.categorical + config.numerical], train[config.target])

    # metrics
    preds = bayes.predict(train[config.categorical + config.numerical])
    print(metrics.classification_report(train[config.target], preds))

    # save model
    parent_dir = Path().resolve()
    dump(bayes, f'{parent_dir}/artifacts/models/model.joblib')


if __name__ == "__main__":
    train()