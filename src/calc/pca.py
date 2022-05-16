import numpy as np


class PrincipalComponentAnalysis:

    @staticmethod
    def feature_normalize(x):
        mu = np.mean(x, 0)
        x_norm = x - mu

        sigma = np.std(x_norm, 0)
        x_norm /= sigma
        return x_norm, mu, sigma

    def __init__(self, x, proceed=True):
        self.x = x
        self.x_norm, self.__mu, self.__sigma = PrincipalComponentAnalysis.feature_normalize(x)
        self.__u = None
        self.__s = None
        self.__v = None
        if proceed:
            self.__run_svd()

    def __run_svd(self):
        x_norm = self.x_norm
        m = len(x_norm)
        cov = 1/m*(x_norm.T.dot(x_norm))
        self.__u, self.__s, self.__v = np.linalg.svd(cov)

    def mean_norm(self):
        return self.__mu

    def eigenvectors(self, k):
        return self.__u[:, :k]

    def eigenvalues(self, k):
        return self.__s.reshape((-1, 1))[:k, :]

    def reduce_dim(self, k):
        return self.x_norm.dot(self.__u[:, :k])

    def recover_from_dim(self, z, k):
        return z.dot(self.__u[:, :k].T)



