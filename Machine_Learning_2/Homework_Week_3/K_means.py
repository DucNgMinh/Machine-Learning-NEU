import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
np.random.seed(10)
class K_means():
    def __init__(self, K):
        self.K = K

    def kmean_init_centers(self):
        # random choose point from original dataset
        return self.X[np.random.choice(self.X.shape[0], self.K, replace= False)]

    def kmean_assign_labels(self, X):
        # calculate distances from points to centers
        D = np.sqrt(np.sum((X[:, None] - self.centers)**2, axis=2))

        # return the group label of the point which is the closest distance
        return np.argmin(D, axis= 1)

    def kmean_update_centers(self):
        # create center point
        centers = np.zeros((self.K, self.X.shape[1]))

        for k in range(self.K):
            # cluster data into group
            Xk = self.X[self.labels == k]

            # calculate new center point
            centers[k] = np.mean(Xk, axis= 0)
        return centers

    def fit_transform(self, X):
        self.X = X
        self.centers = self.kmean_init_centers()

        for i in range(500):
            self.labels = self.kmean_assign_labels(self.X)
            self.centers = self.kmean_update_centers()

    def get_centers(self):
        return self.centers

    def data_and_labels(self):
        return np.hstack((self.X, self.predict(self.X)))
    def inertia_(self):
        sum_distances = []
        for k in range(self.K):
            Xk = self.X[self.labels == k]
            sum_distances.append( np.sum(cdist(Xk, self.centers[k].reshape(1,-1))) )

        return np.sum(sum_distances)

    def predict(self, X):
        return pd.DataFrame(self.kmean_assign_labels(X))




