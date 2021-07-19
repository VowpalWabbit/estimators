import os, sys, random, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimators.bandits import ips
from estimators.bandits import snips
from estimators.bandits import mle
from estimators.bandits import cressieread
from estimators.bandits import gaussian
from estimators.bandits import clopper_pearson
from estimators.ccb import ccb
from estimators.test.utils import Helper

def test_bandits_unit_test():
    listofestimators = [(ccb.Estimator(ips.Estimator()), 2.0),
                        (ccb.Estimator(snips.Estimator()), 1.0),
                        (ccb.Estimator(mle.Estimator()), 1.0),
                        (ccb.Estimator(cressieread.Estimator()), 1.0)]
    
    p_log = [0.3]
    p_pred = [0.6]
    reward = [1]

    for Estimator in listofestimators:
        Estimator[0].add_example(p_log, reward, p_pred)
        assert Estimator[0].get() == Estimator[1]
