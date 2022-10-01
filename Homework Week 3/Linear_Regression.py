import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self):
        self.theta = None
        self.alpha = 0.01 # step size
        self.elpison = 0.001 # error after comma

    def fit(self, x, y):
        # create X matrix
        one_col = np.ones((x.shape[0],1)) # create columns for bias feature ( m * 1 matrix with 1)
        X = np.hstack((one_col,x))

        # calculate thetea
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    def predict(self, x):
        return np.sum(np.dot(x, self.theta[1:]) + self.theta[0])

    def get_model_theta(self):
        return self.theta
