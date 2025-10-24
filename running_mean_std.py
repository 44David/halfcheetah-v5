import numpy as np

class RunningMeanStd:
    def __init__(self, eps=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64) 
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        n_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        n_var = m2 / total_count
        
        self.mean = n_mean
        self.var = n_var
        self.count = total_count
        