
from ivado_assignment.config import config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import SimpleImputer
from ivado_assignment.dot_dict import DotDict


class ModelSettings:
    models_to_test = [
        BalancedRandomForestClassifier(random_state=42)
    ]

    def __init__(self, transformers):
        self.transformers = transformers
        self.set_preprocessing()

    def set_preprocessing(self):
        if (self.transformers.numerical is not None) and (self.transformers.categorical is not None):
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
    })
)
