# Incremental version of https://en.wikipedia.org/wiki/Kahan_summation_algorithm

class IncrementalFsum:
    def __init__(self):
        self.partials = []

    def __iadd__(self, x):
        i = 0
        for y in self.partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                self.partials[i] = lo
                i += 1
            x = hi
        self.partials[i:] = [x]
        return self
        
    def __float__(self):
        return sum(self.partials, 0.0)