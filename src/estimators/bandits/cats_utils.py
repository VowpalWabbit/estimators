import math
from typing import Any, Dict


class CatsTransformer:
    def __init__(
        self, num_actions: int, bandwidth: float, max_value: float, min_value: float
    ) -> None:
        self.num_actions = num_actions
        self.max_value = max_value
        self.min_value = min_value
        self.bandwidth = bandwidth

        self.continuous_range = self.max_value - self.min_value
        self.unit_range = self.continuous_range / float(self.num_actions)

    def get_baseline1_prediction(self) -> float:
        return self.min_value + (self.unit_range / 2.0)

    def transform(self, data: Dict[str, Any], pred_a: float) -> Dict[str, Any]:
        logged_a = data["a"]

        ctr = min(
            (self.num_actions - 1),
            math.floor((pred_a - self.min_value) / self.unit_range),
        )
        centre = self.min_value + ctr * self.unit_range + (self.unit_range / 2.0)

        if math.isclose(centre, logged_a, abs_tol=self.bandwidth):
            b = min(self.max_value, centre + self.bandwidth) - max(
                self.min_value, centre - self.bandwidth
            )
            data["pred_p"] = 1.0 / b
        else:
            data["pred_p"] = 0.0

        return data
