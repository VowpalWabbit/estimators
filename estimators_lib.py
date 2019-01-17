class IPS_SNIPS_Estimators:
    def __init__(self):
        ############################### Aggregates quantities ######################################
        #
        # 'n':   IPS of numerator
        # 'N':   total number of samples in bin from log (IPS = n/N)
        # 'd':   IPS of denominator (SNIPS = n/d)
        # 'Ne':  number of samples in bin when off-policy agrees with log policy
        # 'c':   max abs. value of numerator's items (needed for Clopper-Pearson confidence intervals)
        # 'SoS': sum of squares of numerator's items (needed for Gaussian confidence intervals)
        #
        #################################################################################################
    
        self.data = {'n':0.,'N':0,'d':0.,'Ne':0,'c':0.,'SoS':0}

    def add_example(self, p_log, r, p_pred):
        self.data['N'] += 1
        if p_pred > 0:
            p_over_p = p_pred/p_log
            self.data['d'] += p_over_p
            self.data['Ne'] += 1
            if r != 0:
                self.data['n'] += r*p_over_p
                self.data['c'] = max(self.data['c'], r*p_over_p)
                self.data['SoS'] += (r*p_over_p)**2
                
    def get_estimate(self, type):
        if self.data['N'] == 0:
            raise('Error: No data point added')
        
        if type == 'ips':
            return self.data['n']/self.data['N']
        elif type == 'snips':
            return self.data['n']/self.data['d']
        else:
            raise('Error: Incorrect estimator type {}. Supported options are ips or snips'.format(type))
