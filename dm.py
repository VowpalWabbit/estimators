from vowpalwabbit import pyvw



class Estimator:
    def __init__(self):
        ############################### Aggregates quantities ######################################
        #
        # 'pred':   probabilities
        # 'N':   total number of samples
        # 'r':   costs
        #
        #################################################################################################

        self.data = {'pred': [], 'r': [], 'N': 0}
        self.format = []
        self.test = ["| r: 0.", "| r: 1."]


    def add_example(self, p_log, r, p_pred, count=1):
        self.data['N'] += count
        if count > 1:
            self.data['pred'].extend(p_pred)
            self.data['r'].extend(r)
            for i, p in enumerate(p_pred):
                self.format.append(str(p)+" | r: "+str(float(r[i])))
        elif count == 1:
            self.data['pred'].append(p_pred)
            self.data['r'].append(r)
            self.format.append(str(p_pred)+" | r: "+str(float(r)))
        
    def get_estimate(self):
        if self.data['N'] == 0:
            raise('Error: No data point added')

        model = pyvw.vw(quiet=True)

        for example in self.format:
            model.learn(example)

        return model.predict(self.test[0]), model.predict(self.test[1])
