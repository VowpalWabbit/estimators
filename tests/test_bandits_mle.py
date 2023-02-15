from estimators.bandits import mle


def test_single_example():
    estimator = mle.Estimator()
    estimator.add_example(0.3, 1, 0.6)
    assert estimator.get() == 1.0
