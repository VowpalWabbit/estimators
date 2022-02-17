from estimators.bandits import snips


def test_single_example():
    estimator = snips.Estimator()
    estimator.add_example(0.3, 1, 0.6)
    assert estimator.get() == 1.0