import numpy as np
import pandas as pd

class Ridge_Regression():
    def __init__(self, lamda= 2):
        self.theta = None
        self.lamda = lamda

    def fit(self, x, y):
        # create columns for bias feature ( m * 1 matrix with 1)
        one_col = np.ones((x.shape[0], 1))
        X = np.hstack((one_col, x))

        # create identity matrix
        identity_matrix = np.identity(X.shape[1])
        # calculate thetea
        self.theta = np.linalg.pinv(X.T @ X + self.lamda * identity_matrix) @ X.T @ y

    def predict(self,x):
        if x.ndim == 1:
            x = x.reshap(-1, 1)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x @ self.theta

    def get_model_theta(self):
        return self.theta



