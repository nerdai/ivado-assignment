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
from sklearn.linear_model import LogisticRegression
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
            LogisticRegression(random_state=seed, penalty='l2'),
            {
                "clf__C": [1., 0.99, 0.98, 0.97, 0.96, 0.95]
            }
        ),
        (
            BalancedRandomForestClassifier(random_state=seed),
            {
                "clf__n_estimators": [25, 50, 100],
                "clf__ccp_alpha": [0., 0.01, 0.02, 0.03, 0.04, 0.05]
            }
        ),
        (
            RandomForestClassifier(random_state=seed),
            {
                "clf__n_estimators": [25, 50, 100],
                "clf__ccp_alpha": [0., 0.01, 0.02, 0.03, 0.04, 0.05]
            }
        ),
        (
            GradientBoostingClassifier(random_state=seed),
            {
                "clf__n_estimators": [25, 50, 100],
                "clf__ccp_alpha": [0., 0.01, 0.02, 0.03, 0.04, 0.05]
            }
        ),
    ]

    def __init__(
        self,
        transformers: Dict,
        selection_criteria: callable,
        selection_criteria_comparator: callable,  # min or max
        selection_criteria_default: int,
        train_path: str,
        name: str
    ):
        self.transformers = transformers
        self.model_selection_critiera = selection_criteria
        self.selection_comparator = selection_criteria_comparator
        self.selection_default = selection_criteria_default
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
    selection_criteria_comparator=max,
    selection_criteria_default=float('-inf'),
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
    selection_criteria_comparator=max,
    selection_criteria_default=float('-inf'),
    train_path='./data/splits/complete_df/train.csv',
    name="complete"
)

model_settings = {
    'imputed': imputed,
    'complete': complete
}
