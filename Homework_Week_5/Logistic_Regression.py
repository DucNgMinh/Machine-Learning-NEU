import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, step_size = 0.01, eps=1e-3):
        self.theta = None
        self.step_size = step_size
        self.eps = eps

    def transform(self,x, y):
        if x.ndim == 1:
            x = x.reshape(-1,1)

        one_col = np.ones((x.shape[0],1))
        self.X = np.hstack((one_col, x))
        self.Y = np.array(y).reshape(-1,1)

    def fit(self, X, y):
        self.transform(X, y)
        self.theta = np.zeros(self.X.shape[1]).reshape(-1, 1)

        for i in range(1000):
            y_pred = self.sigmoid()
            grad = np.dot(self.X.T , y_pred - self.Y)
            self.theta -= self.step_size * grad

    def sigmoid(self):
        z = np.dot(self.X, self.theta)
        return 1 / (1 + np.exp(-z))

    def get_theta(self):
        return self.theta

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        z = self.theta @ x
        return 1 / (1 + np.exp(-z))


