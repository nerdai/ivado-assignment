"""
This module takes an input df and splits the df into 3 sets:
    1. train
    2. val
    3. test
The three sets are stored in local.
"""

import argparse
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
from joblib import load

parser = argparse.ArgumentParser(
    prog='ivado data_splitter',
    description='Takes an input DataFrame and splits it into 3 DFs for\
training, testing, and validation.'
)
parser.add_argument('--data', type=str, required=True,
                    help='path to input data csv')
parser.add_argument('--output', type=str, required=True,
                    help='path to output splits as csv')


def split(df: pd.DataFrame, split_fracs=[0.6, 0.2, 0.2], seed=42) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert sum(split_fracs) == 1
    train, val, test = \
        np.split(df.sample(frac=1, random_state=seed),
                 [
            int(split_fracs[0]*len(df)),
            int(sum(split_fracs[:2])*len(df))
        ]
        )

    return train, test, val




if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    train, test, val = split(df, args.output)

    # store splits as csvs in local
    train.to_csv(f"{args.output}/train.csv")
    test.to_csv(f"{args.output}/test.csv")
    val.to_csv(f"{args.output}/val.csv")
