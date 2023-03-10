"""
Unit tests for the ivado_assignment package and its modules.
"""

from ivado_assignment import __version__
from ivado_assignment.data_cleaner import data_cleaner
from ivado_assignment.get_metrics import produce_report
from ivado_assignment.training import train
from ivado_assignment.inference import inference


def test_version():
    """
    Unit test for version.
    """
    assert __version__ == '0.1.0'


def test_callables():
    """
    Unit test for required callables.
    """
    assert callable(data_cleaner)
    assert callable(produce_report)
    assert callable(train)
    assert callable(inference)