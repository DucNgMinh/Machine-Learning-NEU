import numpy as np
import pandas as pd


class Polynomial_Feature():
    def __init__(self, degree):
        self.degree = self.is_valid_degree(degree)

    def is_valid_degree(self, degree):
        if degree < 0:
            raise "Invalid degree"
        return degree

    def fit_transform(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        feature_df = np.empty((len(X),0))
        if self.degree == 0:
            return feature_df
        else:
            for i in range(1, self.degree + 1):
                feature_df = np.append(feature_df, X ** i, axis=1)

        return feature_df


