"""
Unit tests data_splitter script
"""

import numpy as np
import pandas as pd
from ivado_assignment.data_splitter import split


def test_result_shapes():
    """
    Unit test for shapes of train, test, val 
    """
    dummy = pd.DataFrame({
       'A': np.arange(1,101),
       'B': np.arange(1,101)*2
    })
    train, test, val = split(dummy)
    print(train.head())
    exp = [(60, 2), (20, 2), (20, 2)]
    assert train.shape == exp[0]
    assert test.shape == exp[1]
    assert val.shape == exp[2]
