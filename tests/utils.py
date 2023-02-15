import copy


class Scenario:
    def __init__(self, simulator, estimator, alpha=0.05):
        self.simulator = simulator
        self.estimator = estimator
        self.result = None
        self.alpha = alpha

    def aggregate(self):
        for example in self.simulator():
            self.estimator.add_example(**example)

    def get_estimate(self):
        self.aggregate()
        self.result = self.estimator.get()

    def get_interval(self):
        self.aggregate()
        self.result = self.estimator.get(self.alpha)

    def get_r_estimate(self):
        self.aggregate()
        self.result = self.estimator.get_r()

    def get_r_overall_estimate(self):
        self.aggregate()
        self.result = self.estimator.get_r_overall()

    def get_r_interval(self):
        self.aggregate()
        self.result = self.estimator.get_r(self.alpha)

    def get_r_overall_interval(self):
        self.aggregate()
        self.result = self.estimator.get_r_overall(self.alpha)


def get_estimates(scenarios):
    for scenario in scenarios:
        scenario.get_estimate()


def get_intervals(scenarios):
    for scenario in scenarios:
        scenario.get_interval()


def get_r_estimates(scenarios):
    for scenario in scenarios:
        scenario.get_r_estimate()


def get_r_overall_estimates(scenarios):
    for scenario in scenarios:
        scenario.get_r_overall_estimate()


def get_r_intervals(scenarios):
    for scenario in scenarios:
        scenario.get_r_interval()


def get_r_overall_intervals(scenarios):
    for scenario in scenarios:
        scenario.get_r_overall_interval()


class Helper:
    """Helper Class for tests"""

    @staticmethod
    def run_add_example(datagen, estimator, num_examples):
        # class_object is the object of class Estimator() or class Interval()
        Estimator = copy.deepcopy(estimator)

        for n in range(0, num_examples):
            data = datagen()
            Estimator.add_example(data["p_log"], data["r"], data["p_pred"])

        return Estimator

    @staticmethod
    def get_estimate(datagen, estimators, num_examples):
        estimates = []
        for Estimator in estimators:
            estimator = Helper.run_add_example(datagen, Estimator, num_examples)
            estimates.append(estimator.get())

        return estimates
