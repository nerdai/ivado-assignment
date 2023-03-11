"""
This module takes an input df and splits the df into 2 sets:
    1. train
    2. test
The three sets are stored in local.
"""

import argparse
from typing import Tuple
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(
    prog='ivado data_splitter',
    description='Takes an input DataFrame and splits it into 2 DFs for\
training and testing'
)
parser.add_argument('--data', type=str, required=True,
                    help='path to input data csv')
parser.add_argument('--output', type=str, required=True,
                    help='path to output splits as csv')


def split(df: pd.DataFrame, split_fracs=(0.75, 0.25), seed=42) -> \
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function for taking an input DF and splitting it into 2 sets.

    Returns two pd.DataFrames: train, test
    """
    assert sum(split_fracs) == 1
    train, test = np.split(df.sample(frac=1, random_state=seed),  # pylint: disable=unbalanced-tuple-unpacking
                                [int(split_fracs[0]*len(df))])

    return train, test


def main():
    """
    Main method containing logic for this script.
    """
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    train, test = split(df)

    # store splits as csvs in local
    train.to_csv(f"{args.output}/train.csv")
    test.to_csv(f"{args.output}/test.csv")


if __name__ == "__main__":
    main()
