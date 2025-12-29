import numpy as np
from numpy import zeros,ones
class BaseOptimizer:

    def __init__(self, func, n_dim, size_pop, max_iter, lb, ub):
        self.func = func       # Objective function
        self.n_dim = n_dim     # Number of dimensions
        self.pop = size_pop         # Population size
        self.max_iter = max_iter # Max iterations
        self.lb = lb           # Lower bounds
        self.ub = ub           # Upper bounds


        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.Y = np.zeros((self.pop, 1))
        for i in range(self.pop):
            self.Y[i] = self.func(self.X[i])

        # Initialize best positions and fitnesses
        self.pbest_X = self.X.copy()  # Personal Best Position 
        self.pbest_Y = self.Y.copy()  # Personal Best Fitness
        self.gbest_Y = self.pbest_Y.min()
        self.gbest_X = self.X[self.pbest_Y.argmin(), :].copy()

    def _check_bound(self, X):
        X = np.clip(X, self.lb, self.ub)
        return X
     
    def run(self):
        raise 
class GEO(BaseOptimizer):
    def __init__(self, func, n_dim, pop=40, max_iter=50, lb=-1, ub=1,
                 p_a0=0.5, p_c0=1, p_aT=2, p_cT=0.5):
        # Initialize the base components (X, Y, pbest, gbest)
        super().__init__(func, n_dim, pop, max_iter, lb, ub)

        # GEO-specific parameters (Propensity initial/final values)
        self.p_a0 = p_a0
        self.p_c0 = p_c0
        self.p_aT = p_aT
        self.p_cT = p_cT
        self.gbest_y_hist = []

    def _get_propensities(self, t):

        T = self.max_iter
        # Pa = pa0 + (t/T) * |paT - pa0|
        pa = self.p_a0 + (t / T) * abs(self.p_aT - self.p_a0)

        # Pc = pc0 - (t/T) * |pc0 - pcT|
        pc = self.p_c0 - (t / T) * abs(self.p_c0 - self.p_cT)
        return pa, pc

    def _get_cruise_vector(self, A):
        C = 2 * np.random.rand(self.n_dim) -1
        k = np.random.randint(0,self.n_dim)
        while A[k] == 0 :
            k = np.random.randint(0,self.n_dim)
        d = np.sum(A * self.X)
        C[k] = -(np.sum(A * C) - A[k] * C[k]) / A[k]
        C_n = np.linalg.norm(C)
        return C / C_n

    def _update_position(self, pa, pc):
        X_new = np.zeros_like(self.X)
        per = np.random.permutation(self.pop)
        for i in range(self.pop):
            A_i = self.gbest_X - self.X[i]
            A_i = self.pbest_X[per[i]] - self.X[i]
            if np.linalg.norm(A_i) == 0:
               continue

            C_i = self._get_cruise_vector(A_i)
            r = np.linalg.norm(A_i)
            A_i = A_i/np.linalg.norm(A_i)
            r1 = np.random.rand(self.n_dim)
            r2 = np.random.rand(self.n_dim)
            M_i = np.copy(r1 * (pa * A_i) * r + r2 *(pc * C_i) * r)
            X_new[i] = np.copy(self.X[i]) + M_i
            for j in range(X_new[i].size):
                if X_new[i][j] > self.ub[j] :
                   X_new[i][j] = self.ub[j]     
                if X_new[i][j] < self.lb[j] :
                   X_new[i][j] = self.lb[j]
        return X_new

    def run(self,log_file="geo_log.txt"):

        for t in range(self.max_iter):
            pa, pc = self._get_propensities(t+1)
            self.X = self._update_position(pa, pc)
            self.X = self._check_bound(self.X)
            Y_new = np.zeros((self.pop, 1))
            for i in range(self.pop):
                Y_new[i] = self.func(self.X[i])
                if Y_new[i] < self.pbest_Y[i]:
                    self.pbest_Y[i] = np.copy(Y_new[i])
                    self.pbest_X[i] = np.copy(self.X[i])

            current_min_y = self.pbest_Y.min()
           
            if current_min_y < self.gbest_Y:
                self.gbest_Y = current_min_y
                self.gbest_X = self.pbest_X[self.pbest_Y.argmin(), :].copy()

            self.gbest_y_hist.append(self.gbest_Y)
        return self.gbest_X, self.gbest_Y    