import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, df):
        # mean of each column
        mean_col = np.mean(df, axis=0)

        # center matrix
        center_matrix = df - mean_col

        # covariance matrix of centered matrix
        cov_matrix = np.cov(center_matrix.T)

        # calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix.T)

        # sort eigenvalue and eigenvector in descending order
        sorted_index = np.argsort(eigenvalues)[::-1]
        self.sorted_eigenvalue = eigenvalues[sorted_index]
        self.sorted_eigenvector = eigenvectors[:, sorted_index]

        # create eigenvector subset
        self.eigenvector_subset = self.sorted_eigenvector[:, 0:self.n_components]

        # calculate reduced matrix
        Z = center_matrix @ self.eigenvector_subset
        return Z

    def explained_variance_raito(self):
        total_eigenvalues = np.sum(self.sorted_eigenvalue)
        var_explain = [(i / total_eigenvalues) for i in self.sorted_eigenvalue]
        cumulative_sum_var = np.cumsum(var_explain)
        return var_explain, cumulative_sum_var

    def get_components(self):
        return self.eigenvector_subset



