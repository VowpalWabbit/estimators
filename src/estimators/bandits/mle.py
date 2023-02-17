# Empirical likehood estimator

from __future__ import annotations

from math import fsum, inf
from estimators.bandits import base
from typing import List, Optional, Tuple


class Estimator(base.Estimator):
    # NB: This works better you use the true wmin and wmax
    #     which is _not_ the empirical minimum and maximum
    #     but rather the actual smallest and largest possible values
    def __init__(self, wmin: float = 0, wmax: float = inf):
        assert wmin < 1
        assert wmax > 1

        self.wmin = wmin
        self.wmax = wmax

        self.data: List[Tuple[int, float, float]] = []

    def add_example(self, p_log: float, r: float, p_pred: float) -> None:
        w = p_pred / p_log
        assert w >= 0, "Error: negative importance weight"

        self.data.append((1, w, r))
        self.wmax = max(self.wmax, w)
        self.wmin = min(self.wmin, w)

    def graddualobjective(self, n: float, beta: float) -> float:
        return fsum(c * (w - 1) / ((w - 1) * beta + n) for c, w, _ in self.data)

    def get(self) -> Optional[float]:
        from scipy.optimize import brentq  # type: ignore

        n = fsum(c for c, _, _ in self.data)
        if n == 0:
            return None

        betaub = n / (1 - self.wmin)
        betamax = min(
            betaub,
            min(((n - c) / (1 - w) for c, w, _ in self.data if w < 1), default=betaub),
        )

        betalb = 0.0 if self.wmax == inf else n / (1 - self.wmax)
        betamin = max(
            betalb,
            max(((n - c) / (1 - w) for c, w, _ in self.data if w > 1), default=betalb),
        )

        gradmin = self.graddualobjective(n, betamin)
        gradmax = self.graddualobjective(n, betamax)

        if gradmin * gradmax < 0:
            betastar = brentq(
                f=lambda x: self.graddualobjective(n, x), a=betamin, b=betamax
            )
        elif gradmin < 0:
            betastar = betamin
        else:
            betastar = betamax

        sumofw = fsum(c * w / ((w - 1) * betastar + n) for c, w, _ in self.data)
        missing = max(0.0, 1.0 - sumofw)

        vhat = fsum(c * w * r / ((w - 1) * betastar + n) for c, w, r in self.data)
        rhatmissing = fsum(c * r for c, _, r in self.data) / n
        vhat += missing * rhatmissing

        return vhat

    def __add__(self, other: Estimator) -> Estimator:
        raise NotImplementedError()
