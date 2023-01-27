import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

class K_means():
    def __init__(self, K):
        self.K = K

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def kmean_init_centers(self,X):
        return X[np.random.choice(X.shape[0], self.K, replace= False)]

    def kmean_assign_labels(self, X, centers):
        D = cdist(X, centers)
        return np.argmin(D, axis= 1)

    def kmean_update_centers(self, X, labels):
        centers = np.zeros((self.K, X.shape[1]))

        for k in range(self.K):
            Xk = X[labels == k, :]
            centers[k, :] = np.mean(Xk, axis= 0)
        return centers

    def fit_transform(self, X):
        centers = self.kmean_init_centers(X)

        labels = []

        for i in range(500):
            labels.append(self.kmean_assign_labels(X, centers))

            centers = self.kmean_update_centers(X, labels[-1])

        return (centers, labels)

    def predict(self):
        pass


np.random.seed(11)
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

K_mean = K_means(3)
print(K_mean.fit_transform(X))


print(cdist(X, [[1.95180248, 6.72897643],
       [1.9444814 , 3.06716532],
       [3.42144011, 0.17309469]]))
print(np.linalg.norm(np.array([3.74945474,1.713927]) - np.array([1.95180248, 6.72897643])))
print(np.linalg.norm(np.array([3.74945474,1.713927]) - np.array([1.9444814 , 3.06716532])))
print(np.linalg.norm(np.array([3.74945474,1.713927]) - np.array([3.42144011, 0.17309469])))
