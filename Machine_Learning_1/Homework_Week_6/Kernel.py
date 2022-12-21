import numpy as np
import pandas as pd

class Kernel_Ridge_Regression:
    def __init__(self,kernel_type='linear', gamma=5.0, lamda = 0.1):
        self.theta = None
        self.gamma = gamma
        self.lamda = lamda
        self.kernels = {
            'linear': self.Linear_Kernel,
            'polynomial': self.Polynomial_Kernel,
            'RBF': self.RBF_kernel }
        self.kernel_type = kernel_type
        self.kernel = self.kernels[self.kernel_type]

    def transform(self,x, y):
        if x.ndim == 1:
            x = x.reshape(-1,1)

        one_col = np.ones((x.shape[0],1))
        self.X = np.hstack((one_col, x))
        self.Y = np.array(y).reshape(-1,1)

    def Linear_Kernel(self, x1, x2):
        return np.dot(x1,x2.T)

    def Polynomial_Kernel(self, x1, x2):
        return np.dot(x1,x2.T) ** 2

    def RBF_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (self.gamma ** 2)))

    def compute_kernel_matrix(self, X1, X2):
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(X1[i], X2[j])

        return K

    def fit(self, X, y):
        K = self.compute_kernel_matrix(X, X)
        self.theta = np.linalg.pinv(K + self.lamda * np.eye( K.shape[0]) ) * y.T

    def predict(self, X, y):
        K = self.compute_kernel_matrix(X, y)
        return np.dot(K, self.theta).T
