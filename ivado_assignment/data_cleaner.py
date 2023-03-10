"""
This module takes the raw data, applies the data cleaning steps identified
by the EDA. In particular:
- drop the two records that erronoeuous entries for `feature_6`
    + cast feature_6 as a numerical feature
- `feature_8` and `feature_9` treated as categorical features
- create and store complete_df, which removes the 150 records which have a
missing value for one of the features
- create and store imputed_df, which instead imputes the missing values
according to the below strategy:
    + categorical: add a `missing` category level
    + numerical: impute with `mean`
"""

from typing import Tuple
import numpy as np
import pandas as pd


def data_cleaner() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This module method takes the raw data, applies the cleaning steps
    and returns two pandas DataFrames:
        1. complete_df: for a complete-case analysis
        2. imputed_df: which imputes the missing variables
    """
    raw_df = pd.read_csv("./data/2021-10-19_14-11-08_val_candidate_data.csv")
    conts = ['feature_5', 'feature_6', 'feature_7']
    cats = ['feature_0',
            'feature_1',
            'feature_2',
            'feature_3',
            'feature_4',
            'feature_10',
            'feature_11',
            'feature_12',
            'feature_8',
            'feature_9'
            ]

    # drop the two records with erroneous entry for feature_6
    raw_df = raw_df[~raw_df['feature_6'].apply(
        lambda x: '.' in x)].reset_index(drop=True)

    # complete df
    complete_df = raw_df[~raw_df.isna().any(axis=1)].reset_index(drop=True)

    # imputed df
    imputed_df = raw_df.copy()
    # imputing missing categorical features
    for f in cats:
        imputed_df[f] = pd.Categorical(imputed_df[f].fillna('missing'))
    # imputing missing numerical features
    for f in conts:
        mean = np.mean(imputed_df[f])
        imputed_df[f] = imputed_df[f].fillna(mean)

    return complete_df, imputed_df


def store_cleaned_data(complete_df: pd.DataFrame, imputed_df: pd.DataFrame):
    pass


if __name__ == "__main__":
    complete, imputed = data_cleaner()
    store_cleaned_data(complete, imputed)
