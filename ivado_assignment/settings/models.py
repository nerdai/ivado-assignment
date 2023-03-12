
from ivado_assignment.settings.data import config
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import SimpleImputer
from ivado_assignment.utils.dot_dict import DotDict


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

    def __init__(self, transformers, selection_criteria, train_path, name):
        self.transformers = transformers
        self.model_selection_critiera = selection_criteria
        self.train_path = train_path
        self.name = name 
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


imputed = ModelSettings(
    transformers=DotDict({
        "numerical": Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ]),
        "categorical":  Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])
    }),
    selection_criteria=metrics.make_scorer(metrics.f1_score, pos_label=0),
    train_path='./data/splits/incomplete_df/train.csv',
    name="imputed"
)

complete = ModelSettings(
    transformers=DotDict({
        "numerical": None,
        "categorical":  Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])
    }),
    selection_criteria=metrics.make_scorer(metrics.f1_score, pos_label=0),
    train_path='./data/splits/complete_df/train.csv',
    name="complete"
)

model_settings = DotDict({
    'imputed': imputed,
    'complete': complete
})
