import numpy as np
import random


class KMeans:

    @staticmethod
    def find_closest_centroids(x, centroids):
        m, n = x.shape
        idx = np.zeros(m).astype(np.int8)
        for i in range(m):
            distances = np.sum((x[i] - centroids)**2, 1)
            idx[i] = np.argmin(distances)
        return idx

    @staticmethod
    def compute_centroids(x, idx, k):
        m, n = x.shape
        centroids = np.zeros((k, n))
        for i in range(k):
            centroids[i, :] = np.mean(x[idx == i], 0)
        return centroids

    @staticmethod
    def k_means_init_centroids(x, k):
        m = len(x)
        rand_idx = random.sample(range(m), k)
        return x[rand_idx, :]

    @staticmethod
    def distortion(x, idx, centroids):
        '''
        this is cost function
        '''
        k = len(centroids)
        cost = 0
        for i in range(k):
            cluster_group = idx == i
            cost += np.sum((x[cluster_group]-centroids[i])**2)
        return cost

    def __init__(self, data, k, num_iter=1, proceed=True):
        self.data = data
        self.k = k
        self.__num_iter = num_iter
        if proceed:
            self.__k_means()

    def __k_means(self):

        x = self.data
        k = self.k
        min_distortion = None

        for i in range(self.__num_iter):
            initial_centroids = KMeans.k_means_init_centroids(x, k)
            idx = KMeans.find_closest_centroids(x, initial_centroids)

            while True:
                acc_idx = idx.copy()
                centroids = KMeans.compute_centroids(x, idx, k)
                idx = KMeans.find_closest_centroids(x, centroids)
                if np.all(acc_idx == idx):
                    break

            distortion = KMeans.distortion(x, idx, centroids)
            if min_distortion is None or distortion < min_distortion:
                min_distortion = distortion
                idx_min_distortion = idx

        self.cluster_ids = idx_min_distortion
        self.centroids = centroids

        # return idx_min_distortion


