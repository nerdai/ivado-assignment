"""
This module takes the raw data, applies the data cleaning steps identified
by the EDA. In particular:
- drop the two records that erronoeuous entries for `feature_6`
    + cast feature_6 as a numerical feature
"""

from typing import Tuple
import numpy as np
import pandas as pd


RAW_DATA_FILE = "./data/raw/2021-10-19_14-11-08_val_candidate_data.csv"
CONTS = ['feature_5', 'feature_6', 'feature_7']
CATS = ['feature_0',
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
TARGET = 'target'


def data_cleaner() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This module method takes the raw data, applies the cleaning steps
    and returns two pandas DataFrames:
        1. complete_df: for a complete-case analysis
        2. incomplete_df: contains incomplete observations to be imputed
    """
    incomplete_df = pd.read_csv(RAW_DATA_FILE)

    # drop the two records with erroneous entry for feature_6
    incomplete_df = incomplete_df[~incomplete_df['feature_6'].apply(
        lambda x: '.' in x)].reset_index(drop=True)

    # cast to correct data types
    for feat in CATS + [TARGET]:
        incomplete_df[feat] = pd.Categorical(incomplete_df[feat])
    incomplete_df[TARGET] = incomplete_df[TARGET].cat.codes  # Y = 1; N = 0

    # complete df
    complete_df = incomplete_df[~incomplete_df.isna().any(
        axis=1)].reset_index(drop=True)

    return complete_df, incomplete_df


def main():
    """
    Main method containing logic for this script.
    """
    complete, incomplete_df = data_cleaner()

    # store dfs to local
    complete.to_csv("./data/processed/complete_df.csv")
    incomplete_df.to_csv("./data/processed/incomplete_df.csv")


if __name__ == "__main__":
    main()
