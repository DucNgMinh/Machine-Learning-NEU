import numpy as np
import pandas as pd
import warnings

#suppress warnings
warnings.filterwarnings('ignore')

class Lasso_Regression():
    def __init__(self, lamda= 1, learning_rate= 0.001, iteration= 100):
        self.theta = 2
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.iteration = iteration

    def transform(self,X, Y):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        # create bias columns
        one_col = np.ones((X.shape[0], 1))
        self.X = np.hstack((one_col, X))
        # reformat Y
        self.Y = np.array(Y).reshape(-1,1)


    def fit(self, X, Y):
        self.transform(X, Y)
        self.theta = np.zeros(self.X.shape[1])
        for i in range(self.iteration):
            self.calTheta()


    def calTheta(self):
        y_pred = np.matmul(self.X, self.theta).reshape(-1, 1)
        # Calculate gradients:
        dW = np.zeros(self.X.shape[1])
        for j in range(self.X.shape[1]):
            if self.theta[j] > 0:
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - y_pred) + self.lamda) / self.X.shape[0]
            else:
                dW[j] = (-2 * (self.X[:, j]).dot(self.Y - y_pred) - self.lamda) / self.X.shape[0]
        # update W
        self.theta -= self.learning_rate * dW

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshap(-1, 1)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x @ self.theta

    def get_model_theta(self):
        return self.theta


