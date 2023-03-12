"""
Module for loading data and performing any necessary preparation steps.
"""

import pandas as pd
from ivado_assignment.settings.data import config


def load_and_prep(csv_path: str) -> pd.DataFrame:
    """
    Utility function for loading a csv into a pd.DataFrame and casting the
    categorical fields as string, which is necessary prior to passing it to
    downstream sklearn pipeline.
    """
    df = pd.read_csv(csv_path)
    for feat in config['categorical']:
        df[feat] = pd.Categorical(df[feat].astype(str))
    return df
