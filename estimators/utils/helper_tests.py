import random, copy

class Helper():
    ''' Helper Function for tests '''

    def run_estimator(datagen, listofestimators, num_examples):
        is_close = lambda a, b: abs(a - b) <= 1e-6 * (1 + abs(a) + abs(b))
        for Estimator in listofestimators:
            # Estimator is a tuple
            # Estimator[0] is object of class Estimator()
            # Estimator[1] is expected value of the estimator
            for n in range(0,num_examples):
                data = datagen()
                if type(list(data.values())[0]) is list:
                    # if data['p_logs'] is a list then it is slates
                    Estimator[0].add_example(p_logs=data['p_logs'], r=data['r'], p_preds=data['p_preds'])
                else:
                    Estimator[0].add_example(p_log=data['p_log'], r=data['r'], p_pred=data['p_pred'])
            assert is_close(Estimator[0].get(), Estimator[1])

    def run_interval(datagen, listofintervals, n1, n2):
        """ n1 is smaller than n2; Number of examples increase => narrowing CI"""

        for interval in listofintervals:

            # For n1 number of examples
            interval_n1 = copy.deepcopy(interval)
            for i in range(n1):
                data = datagen(i)
                interval_n1.add_example(p_log=data['p_log'], r=data['r'], p_pred=data['p_pred'])
            result_n1 = interval_n1.get()
            width_n1 = result_n1[1]-result_n1[0]
            assert width_n1 > 0

            # For n2 number of examples
            interval_n2 = copy.deepcopy(interval)
            for i in range(n2):
                data = datagen(i)
                interval_n2.add_example(p_log=data['p_log'], r=data['r'], p_pred=data['p_pred'])
            result_n2 = interval_n2.get()
            width_n2 = result_n2[1]-result_n2[0]
            assert width_n2 > 0

            assert width_n2 < width_n1

class SlatesHelper():
    ''' Helper Function for slates tests '''

    def run_interval(datagen, listofintervals, n1, n2):
        """ n1 is smaller than n2; Number of examples increase => narrowing CI"""

        for interval in listofintervals:

            # For n1 number of examples
            interval_n1 = copy.deepcopy(interval)
            for i in range(n1):
                data = datagen(i)
                interval_n1.add_example(p_logs=data['p_logs'], r=data['r'], p_preds=data['p_preds'])
            result_n1 = interval_n1.get()
            width_n1 = result_n1[1]-result_n1[0]
            assert width_n1 > 0

            # For n2 number of examples
            interval_n2 = copy.deepcopy(interval)
            for i in range(n2):
                data = datagen(i)
                interval_n2.add_example(p_logs=data['p_logs'], r=data['r'], p_preds=data['p_preds'])
            result_n2 = interval_n2.get()
            width_n2 = result_n2[1]-result_n2[0]
            assert width_n2 > 0

            assert width_n2 < width_n1
