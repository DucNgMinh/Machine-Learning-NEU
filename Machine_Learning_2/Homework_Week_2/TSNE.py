import numpy as np
import pandas as pd

class TSNE():
    def __init__(self, n_components= 2, PERPLEXITY= 5):
        self.n_components = n_components
        self.learning_rate = 0.01
        self.PERPLEXITY = PERPLEXITY

    def getKey(self, item):
        return item[1]

    def compute_k_neighbours(self, X, X_index, p_or_q= 'p'):
        X_1 = X[X_index]
        list_k_neighbours = []
        for i in range(X.shape[0]):
            if i != X_index:
                X_i = X[i]
                if p_or_q == 'p':
                    distance = np.exp(-np.linalg.norm(X_1 - X_i) ** 2 / (2 * 1 ** 2))
                else:
                    distance = np.exp(1 + np.linalg.norm(X_1 - X_i) ** 2) ** -1
                list_k_neighbours.append([i,distance])
        list_k_neighbours = sorted(list_k_neighbours, key=self.getKey)

        return list_k_neighbours[:self.PERPLEXITY]

    def compute_pij(self, X, X1_index, X2_index):
        x1 = X[X1_index]
        x2 = X[X2_index]
        num = np.exp(-np.linalg.norm(x1 - x2) ** 2) / (2 * 1 ** 2)
        demon = 0
        list_k_neighbours = self.compute_k_neighbours(X, X1_index, 'p')

        for i in list_k_neighbours:
            demon += i[1]

        return num / demon

    def compute_p(self, X):
        table = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if j != i:
                    pij = self.compute_pij(X, i, j)
                    pji = self.compute_pij(X, j, i)
                    table[i,j] = (pij + pji) / (2 * X.shape[0])
        return table

    def compute_qij(self, Y, Y1_index, Y2_index):
        y1 = Y[Y1_index]
        y2 = Y[Y2_index]
        num = (1 + np.linalg.norm(y1 - y2) ** 2) ** (-1)
        demon = 0
        list_k_neighbours = self.compute_k_neighbours(Y, Y1_index, 'q')

        for i in list_k_neighbours:
            demon += i[1]

        return num / demon

    def compute_q(self, Y):
        table = np.zeros((Y.shape[0], Y.shape[0]))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[0]):
                if i != j:
                    qij = self.compute_qij(Y, i, j)
                    table[i, j] = qij
        return table

    def KL_divergence(self, p, q):
        total = 0
        for i in range(p.shape[0]):
            for j in range(q.shape[0]):
                total += p[i,j] * np.log(p[i,j] / q[i,j])
        return total

    def gradient_descent(self, p, q, y):
        for iter in range(1000):
            for i in range(y.shape[0]):
                sum = 0
                for j in range(y.shape[0]):
                    sum += ((y[i] - y[j]) * (p[i,j] - q[i,j]) * (1 + np.linalg.norm(y[i] - y[j] ** 2)) ** -1)
                y[i] -= 4 * sum * self.learning_rate
        return y

    def fit_transform(self, X):
        y = np.random.rand(X.shape[0], self.n_components)
        self.p = self.compute_p(X)
        self.q = self.compute_q(y)

        self.q = self.gradient_descent(self.p, self.q, y)
        return self.q



TSNE = TSNE()
x=np.random.rand(10,3)
y=x.dot(np.random.rand(x.shape[1],2))

print(TSNE.fit_transform(x))

