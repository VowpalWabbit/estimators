import random, copy

class BanditsHelper():
    ''' Helper Function for bandit tests '''

    def example_generator1():
        return  {'p_log': 1,
                'r': 1,
                'p_pred': 1}

    def example_generator2(i, epsilon, p_pred=0.5):
        # Logged Policy
        # 0 - (1-epsilon) : Reward is always zero
        # 1 - epsilon : Reward is always 1

        # p_pred: Fixed at p_pred parameter

        # policy to estimate
        # 0, 1 reward- for probability p_pred

        chosen = int(random.random() < epsilon)
        return {'p_log': epsilon if chosen == 1 else 1 - epsilon,
                'r': 1 if chosen == 1 else 0,
                'p_pred':p_pred}

    def example_generator3(i, epsilon, delta=0.5):
        # Logged Policy
        # 0 - (1-epsilon) : Reward is Bernoulli(delta)
        # 1 - epsilon : Reward is Bernoulli(1-delta)

        # p_pred: 1 if action is chosen, 0 if action not chosen

        # policy to estimate
        # (delta), (1-delta) reward from a Bernoulli distribution - for probability p_pred

        chosen = int(random.random() < epsilon)
        return {'p_log': epsilon if chosen == 1 else 1 - epsilon,
                'r': int(random.random() < 1-delta) if chosen == 1 else int(random.random() < delta),
                'p_pred': int(chosen==1)}

    def run_estimator(function, listofestimators, num_examples):
        is_close = lambda a, b: abs(a - b) <= 1e-6 * (1 + abs(a) + abs(b))
        for Estimator in listofestimators:
            # Estimator is a tuple
            # Estimator[0] is object of class Estimator()
            # Estimator[1] is expected value of the estimator
            for index in range(0,num_examples):
                data = function()
                Estimator[0].add_example(p_log=data['p_log'], r=data['r'], p_pred=data['p_pred'])
            assert is_close(Estimator[0].get(), Estimator[1])

    def run_interval(function, listofintervals, n1, n2):
        """ n1 is smaller than n2; Number of examples increase => narrowing CI"""

        datagen = lambda i: function(i, epsilon=0.5)

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

    def example_generator1(num_slots):
        # num_slots represents the len(p_logs) or len(p_pred) for each example
        data = {'p_logs': [], 'r': 0.0, 'p_preds': []}
        for s in range(num_slots):
            data['p_logs'].append(1)
            data['p_preds'].append(1)
        data['r'] = 1
        return  data

    def example_generator2(i, num_slots, epsilon, p_pred=0.5):

        data = {'p_logs': [], 'r': 0.0, 'p_preds': []}

        for s in range(num_slots):
            # Logged Policy for each slot s
            # 0 - (1-epsilon) : Reward is always zero
            # 1 - epsilon : Reward is always 1

            # p_pred: Fixed at p_pred parameter

            # policy to estimate
            # 0, 1 reward- for probability p_pred; looking at the matches per slot s

            chosen = int(random.random() < epsilon)
            data['p_logs'].append(epsilon if chosen == 1 else 1 - epsilon)
            data['r'] += 1 if chosen == 1 else 0
            data['p_preds'].append(p_pred)

        return data

    def example_generator3(i, num_slots, epsilon, delta=0.5):

        data = {'p_logs': [], 'r': 0.0, 'p_preds': []}

        for s in range(num_slots):
            # Logged Policy for each slot s
            # 0 - (1-epsilon) : Reward is Bernoulli(delta)
            # 1 - epsilon : Reward is Bernoulli(1-delta)

            # p_pred: 1 if action is chosen, 0 if action not chosen

            # policy to estimate
            # (delta), (1-delta) reward from a Bernoulli distribution - for probability p_pred; looking at the matches per slot s

            chosen = int(random.random() < epsilon)
            data['p_logs'].append(epsilon if chosen == 1 else 1 - epsilon)
            data['r'] += int(random.random() < 1-delta) if chosen == 1 else int(random.random() < delta)
            data['p_preds'].append(int(chosen==1))

        return data

    def run_estimator(function, listofestimators, num_examples, num_slots):
        is_close = lambda a, b: abs(a - b) <= 1e-6 * (1 + abs(a) + abs(b))
        for Estimator in listofestimators:
            # Estimator is a tuple
            # Estimator[0] is object of class Estimator()
            # Estimator[1] is expected value of the estimator
            for n in range(num_examples):
                data = function(num_slots)
                Estimator[0].add_example(p_logs=data['p_logs'], r=data['r'], p_preds=data['p_preds'])
            assert is_close(Estimator[0].get(), Estimator[1])

    def run_interval(function, listofintervals, n1, n2, num_slots):
        """ n1 is smaller than n2; Number of examples increase => narrowing CI"""

        datagen = lambda i: function(i, num_slots, epsilon=0.5)

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
