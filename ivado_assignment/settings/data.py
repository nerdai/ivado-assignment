"""
Configuration for the assignment.
"""

config = {
    'data_file': "./data/raw/2021-10-19_14-11-08_val_candidate_data.csv",
    'id_col': 'Unnamed: 0',
    'numerical': ['feature_5', 'feature_6', 'feature_7'],
    'categorical': ['feature_0',
                    'feature_1',
                    'feature_2',
                    'feature_3',
                    'feature_4',
                    'feature_8',
                    'feature_9',
                    'feature_10',
                    ],
    'categorical_levels': {
        'feature_0': [
            'Female',
            'Male',
            'missing'
        ],
        'feature_1': [
            'No',
            'Yes',
            'missing'
        ],
        'feature_2': [
            '0',
            '1',
            '2',
            '3+',
            'missing'
        ],
        'feature_3': [
            'Graduate',
            'Not Graduate',
            'missing'
        ],
        'feature_4': [
            'No',
            'Yes',
            'missing'
        ],
        'feature_8': [
            '12.0',
            '36.0',
            '60.0',
            '84.0',
            '120.0',
            '180.0',
            '240.0',
            '300.0',
            '360.0',
            '480.0',
            'missing'
        ],
        'feature_9': [
            '0.0',
            '1.0',
            'missing'
        ],
        'feature_10': [
            'Rural',
            'Semiurban',
            'Urban'
        ],
    },
    'target': 'target'
}
