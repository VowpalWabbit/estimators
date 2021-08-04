import random, copy

class Helper():
    ''' Helper Class for tests '''

    @staticmethod
    def assert_is_close(n1, n2):
        ''' Function to check if two numbers n1 and n2 are nearly equal'''

        assert abs(n1 - n2) <= 1e-6 * (1 + abs(n1) + abs(n2))

    @staticmethod
    def run_add_example(datagen, estimator, num_examples):
        # class_object is the object of class Estimator() or class Interval()
        Estimator = copy.deepcopy(estimator)

        for n in range(0,num_examples):
            data = datagen()
            Estimator.add_example(data['p_log'], data['r'], data['p_pred'])

        return Estimator

    @staticmethod
    def get_estimate(datagen, estimators, num_examples):
        estimates = []
        for Estimator in estimators:
            
            estimator = Helper.run_add_example(datagen, Estimator, num_examples)
            estimates.append(estimator.get())

        return estimates
