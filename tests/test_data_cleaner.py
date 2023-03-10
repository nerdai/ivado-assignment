"""
Unit tests data_cleaner script
"""

from ivado_assignment.data_cleaner import data_cleaner


def test_result_shapes():
    """
    Unit test for shapes of complete_df and imputed_df
    """
    complete_df, imputed_df = data_cleaner()
    exp = [(428, 15), (578, 15)]
    assert complete_df.shape == exp[0]
    assert imputed_df.shape == exp[1]
