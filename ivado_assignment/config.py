"""
Configuration for the assignment.
"""

from ivado_assignment.dot_dict import DotDict


config = DotDict({
    'data_file': "./data/raw/2021-10-19_14-11-08_val_candidate_data.csv",
    'numerical': ['feature_5', 'feature_6', 'feature_7'],
    'categorical': ['feature_0',
                    'feature_1',
                    'feature_2',
                    'feature_3',
                    'feature_4',
                    'feature_8',
                    'feature_9',
                    'feature_10',
                    'feature_11',
                    'feature_12',
                    ],
    'target': 'target'
})
