import numpy as np
import pandas as pd

from Homework_Week_4 import Polynomial_Feature

class LinearRegression():
    def __init__(self):
        self.theta = None

    def fit(self, x, y):
        # create X matrix
        one_col = np.ones((x.shape[0],1)) # create columns for bias feature ( m * 1 matrix with 1)
        X = np.hstack((one_col,x))

        # calculate thetea
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshap(-1,1)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x @ self.theta

    def get_model_theta(self):
        return self.theta



