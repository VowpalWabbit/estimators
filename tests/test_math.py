from re import L
from estimators.math import IncrementalFsum
from utils import Helper
import math


def test_incremental_fsum_simple():
    fsum = IncrementalFsum()
    assert float(fsum) == 0

    fsum += 1
    assert float(fsum) == 1

    fsum += 2
    assert float(fsum) == 3


def test_incremental_fsum_is_better_than_naive_one():
    compensated_sum = IncrementalFsum()
    naive_sum = 0

    large = 2**50

    naive_sum += large
    compensated_sum += large

    small = 0.5**15
    for _ in range(2**15):
        compensated_sum += small
        naive_sum += small

    expected = large + 1
    assert math.fabs(expected - naive_sum) > math.fabs(
        expected - float(compensated_sum)
    )


def test_incremental_fsum_summation():
    large = 2**50

    first = IncrementalFsum()
    second = IncrementalFsum()

    first += large
    second += large

    small = 0.5**15
    for _ in range(2**15):
        first += small
        second += small

    first_plus_second = IncrementalFsum.merge(first, second)
    expected = 2 * large + 2
    assert math.isclose(float(first_plus_second), expected, rel_tol=0.5**52)

    first_plus_second += large
    for _ in range(2**15):
        first_plus_second += small

    expected = 3 * large + 3
    assert math.isclose(float(first_plus_second), expected, rel_tol=0.5**52)
