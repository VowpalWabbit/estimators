from re import L
from estimators.math import IncrementalFsum
from estimators.test.utils import Helper
import math

def test_incremental_fsum_simple():
    fsum = IncrementalFsum()
    assert (float)(fsum) == 0

    fsum += 1
    assert (float)(fsum) == 1

    fsum += 2
    assert (float)(fsum) == 3

def test_incremental_fsum_is_better_than_naive_one():
    compensated_sum = IncrementalFsum()
    naive_sum = 0

    large = 2 ** 40

    naive_sum += large
    compensated_sum += large

    d = 0.5 ** 20
    for _ in range(2 ** 20):
        compensated_sum += d
        naive_sum += d

    expected = large + 1
    assert math.fabs(expected - naive_sum) > math.fabs(expected - (float)(compensated_sum))

def test_incremental_fsum_summation():
    large = 2 ** 40

    first = IncrementalFsum()
    second = IncrementalFsum()

    first += large
    second += large

    d = 0.5 ** 20
    for _ in range(2 ** 20):
        first += d
        second += d

    first_plus_second = first + second
    expected = 2 * large + 2
    Helper.assert_is_close((float)(first_plus_second), expected)

    for _ in range(2 ** 20):
        first_plus_second += d

    expected = 2 * large + 3
    Helper.assert_is_close((float)(first_plus_second), 2 * large + 3)
