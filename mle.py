# Code by pminero (https://github.com/pmineiro)

class Estimator:
    def __init__(self):
        self.data = []
        self.n = 0
        self.wmax = 0
        self.wmin = 1
        
    def add_example(self, p_log, r, p_pred, count=1):
        self.data.append((count, p_pred/p_log, r))
        self.n += count
        self.wmax = max(self.wmax, p_pred/p_log)
        self.wmin = min(self.wmin, p_pred/p_log)
        
    @staticmethod
    def qbeta(beta, w, r, n):
        return 1 / ((w - 1) * beta + n)
    
    def estimatebeta(self):
        from collections import defaultdict
        import numpy as np
        from math import log
        from scipy.optimize import brentq
                
        wmin = self.wmin
        wmax = self.wmax
        n = self.n
        
        assert wmax >= 1
        assert wmin <= 1
        assert n > 0
        
        # alpha > 0
        
        betamax = (n-1) * min((1 / (1 - w) for (c, w, r) in self.data if w < 1), default=0)
        betamin = (n-1) * max((1 / (1 - w) for (c, w, r) in self.data if w > 1), default=0)
        
        def xi(beta):
            return sum(c*(w - 1) / ( (w - 1) * beta + n ) for (c, w, r) in self.data)
        
        fmin = xi(betamin)
        fmax = xi(betamax)
        
        if np.allclose(fmin, 0):
            beta = betamin
        elif np.allclose(fmax, 0):
            beta = betamax
        elif fmin * fmax < 0:
            beta = brentq(xi, a=betamin, b=betamax)
        else:
            beta = None
            
        likelyq = None
        betastar = None
        missingw = None
        
        if beta is not None:
            sumofone = sum(c * Estimator.qbeta(beta, w, r, n) for (c, w, r) in self.data)
            sumofw = sum(c * w * Estimator.qbeta(beta, w, r, n) for (c, w, r) in self.data)
            
            if np.allclose(sumofone, 1) and np.allclose(sumofw, 1):
                likelyq = sum((c/n) * log(c * Estimator.qbeta(beta, w, r, n)) for (c, w, r) in self.data)
                betastar = beta
                
        # alpha = 0
        
        def phi(wj):
            return sum(c * (w - 1) * (wj - 1) / (w - wj) for (c, w, r) in self.data)
        
        walpha = set()
        if wmin < wmax:
            if n < 2:
                uniquew = min(w for (c, w, r) in self.data)
                if wmin < uniquew and uniquew > 1:
                    walpha.add(wmin)
                if wmax > uniquew and uniquew < 1:
                    walpha.add(wmax)
            else:
                wjmaxlt1 = min(((n * w - 1) / (n - 1) for (c, w, r) in self.data if w < 1), default=wmin) 
                
                if wmin <= wjmaxlt1:
                    phimin = phi(wmin)
                    
                    if phimin <= 0:
                        walpha.add(wmin)
                    walpha.add(wjmaxlt1)
                               
                wjmingt1 = max(((n * w - 1) / (n - 1) for (c, w, r) in self.data if w > 1), default=wmax)

                if wjmingt1 <= wmax:
                    phimin = phi(wjmingt1)
                    
                    if phimin <= 0:
                        walpha.add(wjmingt1)
                    walpha.add(wmax)
        
        for wj in walpha:
            beta = -n / (wj - 1)
            sumofone = sum(c * Estimator.qbeta(beta, w, r, n) for (c, w, r) in self.data)
            sumofw = sum(c * w * Estimator.qbeta(beta, w, r, n) for (c, w, r) in self.data)
            missing = 1 - sumofone
            
            if sumofone < 1 and np.allclose(sumofw + wj * missing, 1):
                likelyqbeta = sum((c/n) * log(c * Estimator.qbeta(beta, w, r, n)) for (c, w, r) in self.data)
                
                if likelyq is None or likelyq < likelyqbeta:
                    betastar = beta
                    likelyq = likelyqbeta
                    missingw = wj
                
        assert betastar is not None
        return betastar, missingw
    
    def get_estimate(self, rmin=0, rmax=1):
    
        betastar, missingw = self.estimatebeta()
                
        vhat = 0
        sumofone  = 0
        for (c, w, r) in self.data:
            vhat += c * w * Estimator.qbeta(betastar, w, r, self.n) * r
            sumofone += c * Estimator.qbeta(betastar, w, r, self.n)
        del c, w, r
                        
        if missingw is not None:
            print('missingw')
            baseline = 0.5*(rmax - rmin)
            vhat += missingw*(1-sumofone)*baseline
        
        return vhat