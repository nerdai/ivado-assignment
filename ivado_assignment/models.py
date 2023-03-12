
from ivado_assignment.config import config
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import SimpleImputer
from ivado_assignment.dot_dict import DotDict


class ModelSettings:
    seed = 42
    classifiers = [
        BalancedRandomForestClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed),
        GradientBoostingClassifier(random_state=seed)
    ]
    hyperparams = {
        "clf__n_estimators": [25, 50, 100]
    }

    def __init__(self, transformers, selection_criteria):
        self.transformers = transformers
        self.model_selection_critiera = selection_criteria
        self.set_preprocessing()

    def set_preprocessing(self):
        if ((self.transformers.numerical is not None)
                and (self.transformers.categorical is not None)):
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', self.transformers.numerical, config.numerical),
                    ('cat', self.transformers.categorical, config.categorical)])
        elif self.transformers.numerical is None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', self.transformers.categorical, config.categorical)],
                remainder='passthrough'
            )
        elif self.transformers.categorical is None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', self.transformers.numerical, config.numerical)],
                remainder='passthrough'
            )
        else:
            preprocessor = None
        self.preprocessing = preprocessor


model_setting_1 = ModelSettings(
    transformers=DotDict({
        "numerical": Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ]),
        "categorical":  Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])
    }),
    selection_criteria=metrics.make_scorer(metrics.f1_score, pos_label=0)
)
