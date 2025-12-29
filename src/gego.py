from sko.GA import GA
import numpy as np
class GEGO(GA):
    def __init__(self, func, n_dim, size_pop=40, max_iter=50, lb=-1, ub=1,
                 p_a0=0.5, p_c0=1, p_aT=2, p_cT=0.5,prob_mut=0.001, constraint_eq=tuple(), constraint_ueq=tuple(),precision=1e-7, early_stop=None, n_processes=0,frequancy=5):
        # Initialize the base components (X, Y, pbest, gbest)

        super().__init__(func, n_dim, size_pop, max_iter, prob_mut,lb,ub, constraint_eq, constraint_ueq,precision, early_stop)
        self.func2 = func
        # GEO-specific parameters (Propensity initial/final values)
        self.p_a0 = p_a0
        self.p_c0 = p_c0
        self.p_aT = p_aT
        self.p_cT = p_cT
        self.freq = frequancy
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        self.Y = np.zeros((self.size_pop, 1))
        for i in range(self.size_pop):
            self.Y[i] = self.func2(self.X[i])
        # Initialize best positions and fitnesses
        self.pbest_x = self.X.copy()  # Personal Best Position (Pi)
        self.pbest_y = self.Y.copy()  # Personal Best Fitness

        # Initialize Global Best (gbest_X is Xf, gbest_Y is f(Xf))
        self.gbest_Y = self.pbest_y.min()
        self.gbest_X = self.X[self.pbest_y.argmin(), :].copy()
        # Variables to store the history of global best fitness
        self.gbest_y_hist = []

    def _check_bound(self, X):
        X = np.clip(X, self.lb, self.ub)
        return X

    def _get_propensities(self, t):

        T = self.max_iter
        # Pa = pa0 + (t/T) * |paT - pa0|
        pa = self.p_a0 + (t / T) * abs(self.p_aT - self.p_a0)

        # Pc = pc0 - (t/T) * |pc0 - pcT|
        pc = self.p_c0 - (t / T) * abs(self.p_c0 - self.p_cT)
        return pa, pc
     
    def _get_cruise_vector(self, A):
        C = 2 * np.random.rand(self.n_dim) - 1
        k = np.random.randint(0,self.n_dim)
        while A[k] == 0 :
            k = np.random.randint(0,self.n_dim)
        d = np.sum(A * self.X)
        C[k] = -(np.sum(A * C) - A[k] * C[k]) / A[k]
        C_n = np.linalg.norm(C)
        return C / C_n
   
    def _update_position(self, pa, pc):

        X_new = np.zeros_like(self.X)
        per = np.random.permutation(self.size_pop)
        for i in range(self.size_pop):
            A_i = np.copy(self.pbest_x[per[i]] - self.X[i])
            r = np.linalg.norm(A_i)
            if np.linalg.norm(A_i) == 0:
               continue
            C_i = self._get_cruise_vector(A_i)
            A_i = A_i/np.linalg.norm(A_i)
            r1 = np.random.rand(self.n_dim) 
            r2 = np.random.rand(self.n_dim)
            M_i = r1 * (pa * A_i) * r + r2 *(pc * C_i) *r
            X_new[i] = np.copy(self.X[i]) + M_i
            for j in range(X_new[i].size):
                if X_new[i][j] > self.ub[j] :
                   X_new[i][j] = self.ub[j]     
                if X_new[i][j] < self.lb[j] :
                   X_new[i][j] = self.lb[j]
        return X_new
    def rv2gray(self, real_vals, len_gray_code):

        real_vals = np.asarray(real_vals).reshape(-1)
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        mask_sum = mask.sum()
        weighted = real_vals[:, None] * mask_sum
        binary = np.zeros((len(real_vals), len_gray_code))
        for j in range(len_gray_code):
            bit = (weighted >= mask[j]).astype(int).ravel()
            binary[:, j] = bit
            weighted -= bit[:, None] * mask[j]
        gray = np.zeros_like(binary)
        gray[:, 0] = binary[:, 0]
        for j in range(1, len_gray_code):
            gray[:, j] = (binary[:, j].astype(int) ^ binary[:, j - 1].astype(int))

        return gray.astype(int)

    def x2chrom(self, X): 
        Chrom = np.zeros((self.size_pop, self.len_chrom), dtype=int)
        if self.int_mode:
            X_norm = (X - self.lb) / (self.ub_extend - self.lb)
        else:
            X_norm = (X - self.lb) / (self.ub - self.lb)
        X_norm = np.clip(X_norm, 0, 1)
        cumsum_len = self.Lind.cumsum()

        start_bit = 0
        for i in range(self.n_dim):
            bit_len = self.Lind[i]
            end_bit = cumsum_len[i]
            Xi = X_norm[:, i]
            gray = self.rv2gray(Xi, bit_len)
            Chrom[:, start_bit:end_bit] = gray
            start_bit = end_bit
        return Chrom
    
    def run(self):
        best = []
        # frequancy of genetic mutation
        freq = self.freq
        for t in range(1, self.max_iter + 1):
            
            if (t) % freq == 0 :
                
                self.Chrom = self.x2chrom(self.X)
                self.Y = self.x2y()
                self.ranking()
                self.selection()
                self.crossover()
                self.mutation()

                # record the best ones
                generation_best_index = self.FitV.argmax()
                self.generation_best_X.append(self.X[generation_best_index, :])
                self.generation_best_Y.append(self.Y[generation_best_index])
                self.all_history_Y.append(self.Y)
                self.all_history_FitV.append(self.FitV)

                if self.early_stop:
                    best.append(min(self.generation_best_Y))
                    if len(best) >= self.early_stop:
                        if best.count(min(best)) == len(best):
                            break
                        else:
                            best.pop(0)
                
                global_best_index = np.array(self.generation_best_Y).argmin()
                self.best_x = self.generation_best_X[global_best_index]
                self.best_y = self.func(np.array([self.best_x]))
                
                self.X = self.chrom2x(self.Chrom)
                self.Y = self.x2y()
                for i in range(self.size_pop):
                    if self.Y[i] < self.pbest_y[i]:
                        self.pbest_y[i] = self.Y[i]
                        self.pbest_x[i] = self.X[i].copy()
               
                if self.best_y < self.gbest_Y:
                    self.gbest_Y = self.best_y
                    self.gbest_X = self.best_x.copy()
            
            pa, pc = self._get_propensities(t)
            self.X = self._update_position(pa, pc)
            self.X = self._check_bound(self.X)
            Y_new = np.zeros((self.size_pop, 1))
            for i in range(self.size_pop):
                Y_new[i] = self.func2(self.X[i])
                if Y_new[i] < self.pbest_y[i]:
                    self.pbest_y[i] = np.copy(Y_new[i])
                    self.pbest_x[i] = np.copy(self.X[i])

            current_min_y = self.pbest_y.min()
            if current_min_y < self.gbest_Y:
                self.gbest_Y = current_min_y
                self.gbest_X = self.pbest_x[self.pbest_y.argmin(), :].copy()
            self.gbest_y_hist.append(self.gbest_Y)

        del self.X,self.Y,self.pbest_y,self.pbest_x
        return self.gbest_X, self.gbest_Y