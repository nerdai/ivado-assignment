"""
This module takes the raw data, applies the data cleaning steps identified
by the EDA. In particular:
- drop the two records that erronoeuous entries for `feature_6`
    + cast feature_6 as a numerical feature
"""

from typing import Tuple
import numpy as np
import pandas as pd
from ivado_assignment.settings.data import config


def data_cleaner() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This module method takes the raw data, applies the cleaning steps
    and returns two pandas DataFrames:
        1. complete_df: for a complete-case analysis
        2. incomplete_df: contains incomplete observations to be imputed
    """
    incomplete_df = pd.read_csv(config.data_file)

    # drop the two records with erroneous entry for feature_6
    incomplete_df = incomplete_df[~incomplete_df['feature_6'].apply(
        lambda x: '.' in x)].reset_index(drop=True)

    # cast to correct data types
    for feat in config.categorical + [config.target]:
        incomplete_df[feat] = pd.Categorical(incomplete_df[feat])
    incomplete_df[config.target] = incomplete_df[config.target].cat.codes
                                                            # Y = 1; N = 0

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
