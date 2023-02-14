from estimators.bandits import ips


def test_single_example():
    estimator = ips.Estimator()
    estimator.add_example(0.3, 1, 0.6)
    assert estimator.get() == 2.0
