"""
Configuration for the assignment.
"""


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


config = DotDict({
    'data_file': "./data/raw/2021-10-19_14-11-08_val_candidate_data.csv",
    'numeric': ['feature_5', 'feature_6', 'feature_7'],
    'categorical': ['feature_0',
                    'feature_1',
                    'feature_2',
                    'feature_3',
                    'feature_4',
                    'feature_8',
                    'feature_9'
                    'feature_10',
                    'feature_11',
                    'feature_12',
                    ],
    'target': 'target'
})
