__version__ = "0.1"

import numpy as np
import joblib

class RecommendationEngine:
    '''
    Recommendation model designed specially for the Sako mobile app.

    hyper parameters:

        learining_rate (float): the speed at which the model learns. The higher the value less likely it is to reach a global minima.

        l1 (float): the regularization parameter used to prevent overfitting of the model.

        gradient ("stoich",): the method of gradient calculation.

        tol (int): the number of iteration without improvements before the model stops learning.

        max_iter (int): the maximum number of iteration the model is allowed to learn for.

        min_M (int): the minimum number of columns (users) in the training matrix.

        min_N (int): the minimum number of rows (products) in the training matrix. 

        R (2): the matrix factorization size.

    '''

    def __init__(self, learning_rate=0.01, l1=0.01, gradient="stoch", tol=500 , max_iter=50000, min_M=3, min_N=3, R=2) -> None:
        self.learning_rate = learning_rate
        self.l1 = l1
        self.gradient = gradient
        self.tol = tol
        self.max_iter = max_iter
        self.min_M = min_M
        self.min_N = min_N
        self.R = R
    
    def train(self, matrix:np.ndarray, matrix_init = "random") -> None:
        '''
        Train the model with fresh data (matrix).

        Parameters:

            matrix (numpy array): the training data (matrix).

            matrix_init ("random", "ones"): initializes the matrices randomly or with 1's.
        '''
        M, N = matrix.shape
        self.no_of_ratings = no_of_ratings = (matrix > 0).sum()
        if M < self.min_M or N < self.min_N or no_of_ratings < 10:
            return
        rng = np.random.default_rng()
        if matrix_init == "random":
            matrix_1 = rng.uniform(1, 2, size=(M, self.R))
            matrix_2 = rng.uniform(1, 2, size=(self.R, N))
        elif matrix_init == "ones":
            matrix_1 = np.ones((M, self.R), dtype=np.float16)
            matrix_2 = np.ones((self.R, N), dtype=np.float16)
        self.matrix_1 = np.ones((M, self.R), dtype=np.float16)
        self.matrix_2 = np.ones((self.R, N), dtype=np.float16)
        if self.gradient == "stoch":
            self._stochastic_gradient_decent(matrix, matrix_1, matrix_2)
    
    def _stochastic_gradient_decent(self, matrix:np.ndarray, matrix_1:np.ndarray, matrix_2:np.ndarray):
        '''
        The stochastic gradient decent used to find the global minima of the.
        '''
        loss = np.zeros(self.no_of_ratings)
        learning_rate = self.learning_rate
        l1 = self.l1
        R = self.R
        old_total_loss = 1
        total_loss = y = e = p = q = tol = 0
        _tol = self.tol
        for count in range(self.max_iter):
            rating_no = 0
            for i in range(matrix_1.shape[0]):
                for j in range(matrix_2.shape[1]):
                    if matrix[i, j] == 0:
                        continue
                    y = matrix_1[i,:].dot(matrix_2[:, j])
                    e = matrix[i, j] - y
                    for k in range(R):
                        p = (learning_rate*(2.*e*matrix_2[k, j] - l1*matrix_1[i, k]))
                        q = (learning_rate*(2.*e*matrix_1[i, k] - l1*matrix_2[k, j]))
                        matrix_1[i, k] += p
                        matrix_2[k, j] += q
                    y = matrix_1[i,:].dot(matrix_2[:, j])
                    loss[rating_no] = (matrix[i, j] - y)**2
                    rating_no += 1
            total_loss = np.average(loss)
            print("Loss:", total_loss)

            if tol >= _tol:
                break
            if total_loss >= old_total_loss:
                tol += 1
                if tol == 1:
                    self.matrix_1[:] = matrix_1
                    self.matrix_2[:] = matrix_2
            else:
                tol = 0
                old_total_loss = total_loss
        print(count)
    

    def recommend(self):
        pass
            
if __name__ == "__main__":
    r = RecommendationEngine()
    m = np.random.default_rng().integers(0, 5, size=(50, 50), dtype=np.int8)
    r.train( m)
    print(r.matrix_1.dot(r.matrix_2).round())
    print(m)
   
