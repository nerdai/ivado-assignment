"""
Utils module containing utility functions used by other modules.
"""

import pandas as pd
from ivado_assignment.settings.data import config

def load_and_prep(csv_path: str) -> pd.DataFrame:
    """
    Utility function for loading a csv into a pd.DataFrame
    params:
    [] train_csv_path: str
    returns:
    [] pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    for feat in config.categorical:
        df[feat] = pd.Categorical(df[feat].astype(str))
    return df