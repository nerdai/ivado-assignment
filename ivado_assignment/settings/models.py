"""
This module stores the model settings for both complete-case and imputed
analyses in a custom ModelSettings class. Here the preprocessing pipeline among
other admin items are defined for these settings.

Note that the features used in both settings are common are set in
ivado_assignment.settings.data.config
"""


from typing import Dict
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from ivado_assignment.settings.data import config


class ModelSettings:  # pylint: disable=too-few-public-methods
    """
    A custom class to hold the settings for the two kinds of analyses:
    complete-case and imputed. This is where to specify:
        - classifiers or models to choose from
        - params to put in the grid search
        - and preprocessing pipeline for categorical and numerical features
    """
    seed = 42
    classifiers_and_hyperparms = [
        (
            BalancedRandomForestClassifier(random_state=seed),
            {
                "clf__n_estimators": [25, 50, 100]
            }
        ),
        (
            RandomForestClassifier(random_state=seed),
            {
                "clf__n_estimators": [25, 50, 100]
            }
        ),
        (
            GradientBoostingClassifier(random_state=seed),
            {
                "clf__n_estimators": [25, 50, 100]
            }
        ),
    ]

    def __init__(
        self,
        transformers: Dict,
        selection_criteria: callable,
        train_path: str,
        name: str
    ):
        self.transformers = transformers
        self.model_selection_critiera = selection_criteria
        self.train_path = train_path
        self.name = name
        self.set_preprocessing()

    def set_preprocessing(self):
        """
        The method to set the preprocessing pipeline from the specified
        transformers for both numerical and categorical features.
        """
        if ((self.transformers['numerical'] is not None)
                and (self.transformers['categorical'] is not None)):
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'num',
                        self.transformers['numerical'],
                        config['numerical']
                    ),
                    (
                        'cat',
                        self.transformers['categorical'],
                        config['categorical']
                    )
                ]
            )
        elif self.transformers['numerical'] is None:
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'cat',
                        self.transformers['categorical'],
                        config['categorical']
                    )
                ],
                remainder='passthrough'
            )
        elif self.transformers['categorical'] is None:
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'num',
                        self.transformers['numerical'],
                        config['numerical']
                    )
                ],
                remainder='passthrough'
            )
        else:
            preprocessor = None
        self.preprocessing = preprocessor


imputed = ModelSettings(
    transformers={
        "numerical": Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ]),
        "categorical":  Pipeline(steps=[
            ('imputer',
                SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])
    },
    selection_criteria=make_scorer(f1_score, pos_label=0),
    train_path='./data/splits/incomplete_df/train.csv',
    name="imputed"
)

complete = ModelSettings(
    transformers={
        "numerical": None,
        "categorical":  Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])
    },
    selection_criteria=make_scorer(f1_score, pos_label=0),
    train_path='./data/splits/complete_df/train.csv',
    name="complete"
)

model_settings = {
    'imputed': imputed,
    'complete': complete
}
