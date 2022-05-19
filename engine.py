__version__ = "0.1"

import numpy as np
import joblib

class RecommendationEngine:
    '''
    Recommendation model designed specially for the Sako mobile app.
    '''

    def __init__(self, learning_rate=0.01, l1=0.01, gradient="stoch", tol=50 , max_iter=50000, min_M=3, min_N=3, R=2) -> None:
        self.learning_rate = learning_rate
        self.l1 = l1
        self.gradient = gradient
        self.tol = tol
        self.max_iter = max_iter
        self.min_M = min_M
        self.min_N = min_N
        self.R = R
    
    def train(self, matrix:np.ndarray) -> None:

        M, N = matrix.shape
        self.no_of_ratings = no_of_ratings = (matrix > 0).sum()
        if M < self.min_M or N < self.min_N or no_of_ratings < 10:
            return
        rng = np.random.default_rng()
        matrix_1 = rng.uniform(1, 2, size=(M, self.R))
        matrix_2 = rng.uniform(1, 2, size=(self.R, N))
        if self.gradient == "stoch":
            self._stochastic_gradient_decent(matrix, matrix_1, matrix_2)
    
    def _stochastic_gradient_decent(self, matrix:np.ndarray, matrix_1:np.ndarray, matrix_2:np.ndarray):
        loss = np.zeros(self.no_of_ratings)
        learning_rate = self.learning_rate
        l1 = self.l1
        R = self.R
        old_total_loss = 1
        total_loss = y = e = tol = 0
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
                        matrix_1[i, k] += (learning_rate*(2*e*matrix_2[k, j] - l1*matrix_1[i, k]))
                        matrix_2[k, j] += (learning_rate*(2*e*matrix_1[i, k] - l1*matrix_2[k, j]))
                    y = matrix_1[i,:].dot(matrix_2[:, j])
                    loss[rating_no] = (matrix[i, j] - y)**2
                    rating_no += 1
            total_loss = loss.sum()
            print("Loss:", total_loss)

            if tol >= _tol:
                break
            if total_loss >= old_total_loss:
                print("critical", total_loss>old_total_loss)
                tol += 1
            else:
                tol = 0
            old_total_loss = total_loss

        print(count)
        self.matrix_1 = matrix_1
        self.matrix_2 = matrix_2
    

    def recommend(self):
        pass
            
if __name__ == "__main__":
    r = RecommendationEngine()
    m = np.random.default_rng().integers(0, 5, size=(10, 10))
    r.train( m)
    print(r.matrix_1.dot(r.matrix_2).round())
    print(m)
