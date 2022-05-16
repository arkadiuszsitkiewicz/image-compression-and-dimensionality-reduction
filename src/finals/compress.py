from src.calc.K_means import KMeans

# class CompressFile(KMeans):
#
#     @staticmethod
#     def convert_into_2d(matrix_3d):
#         m, n, o = matrix_3d.shape
#         return matrix_3d.reshape(m*n, o)
#
#     def __init__(self, data, k):
#         self.__original_shape = data.shape
#         super().__init__(CompressFile.convert_into_2d(data), k)
#         self.compressed = self.__compress_file()
#
#     def __compress_file(self):
#         centroids = self.centroids
#         cluster_ids = self.cluster_ids
#
#         m, n, o = self.__original_shape
#         return centroids[cluster_ids, :].reshape(m, n, o)


class CompressFile:

    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.compressed = self.__compress_file()

    def __compress_file(self):
        data = self.data
        m, n, o = data.shape
        data = data.reshape((m*n, o))
        k_means = KMeans(data, self.k)
        centroids = k_means.centroids
        cluster_ids = k_means.cluster_ids
        return centroids[cluster_ids, :].reshape(m, n, o)




