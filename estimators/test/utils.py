import random, copy

class Helper():
    ''' Helper Class for tests '''

    @staticmethod
    def assert_is_close(n1, n2):
        ''' Function to check if two numbers n1 and n2 are nearly equal'''

        assert abs(n1 - n2) <= 1e-6 * (1 + abs(n1) + abs(n2))

    @staticmethod
    def run_add_example(datagen, class_object, num_examples):
        # class_object is the object of class Estimator() or class Interval()
        Object = copy.deepcopy(class_object)

        for n in range(0,num_examples):
            data = datagen()
            if type(list(data.values())[0]) is list:
                # if data['p_logs'] is a list then it is slates
                Object.add_example(p_logs=data['p_logs'], r=data['r'], p_preds=data['p_preds'])
            else:
                Object.add_example(p_log=data['p_log'], r=data['r'], p_pred=data['p_pred'])

        return Object

    @staticmethod
    def get_estimate(datagen, listofestimators, num_examples):
        estimates = []
        for Estimator in listofestimators:
            # Estimator is a tuple
            # Estimator[0] is object of class Estimator()
            # Estimator[1] is expected value of the estimator
            
            estimator = Helper.run_add_example(datagen, Estimator[0], num_examples)
            estimates.append(estimator.get())

        return estimates

    @staticmethod
    def calc_CI_width(datagen, listofintervals, n1, n2):
        ''' Note: n1 is smaller than n2 '''

        width_n1 = []
        width_n2 = []
        for interval in listofintervals:

            # For n1 number of examples
            interval_n1 = Helper.run_add_example(datagen, interval, n1)
            result_n1 = interval_n1.get()
            width_n1.append(result_n1[1]-result_n1[0])

            # For n2 number of examples
            interval_n2 = Helper.run_add_example(datagen, interval, n2)
            result_n2 = interval_n2.get()
            width_n2.append(result_n2[1]-result_n2[0])

        return width_n1, width_n2
