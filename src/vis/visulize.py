import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    VIS_TYPE = ("CLUSTERED DATA", "IMAGE", "PCA VIS", "PCA EIGENVECTORS", "PLOT FACES", "PLOT 3D")

    @classmethod
    def get_vis_type(cls):
        return cls.VIS_TYPE

    @staticmethod
    def get_line(p1, p2):
        return [p1[0], p2[0]], [p1[1], p2[1]]

    def __init__(self, *args, vistype):

        vistype = vistype.upper().strip()
        if vistype not in Visualization.VIS_TYPE:
            raise ValueError(f"{vistype} is not a valid visualization type")
        elif vistype == "CLUSTERED DATA":
            self.__plot_clustered_data(args)
        # todo
        elif vistype == "IMAGE":
            self.__show_images(args)

        elif vistype == "PCA VIS":
            self.__pca_visualization(args)

        elif vistype == "PCA EIGENVECTORS":
            self.__pca_eigenvectors(args)

        elif vistype == "PLOT FACES":
            self.__plot_faces(args)

        else:
            self.__plot3D(args)

    def __plot_clustered_data(self, args):
        m = len(args)
        fig, axs = plt.subplots(m, 2, figsize=(9, 6))
        fig.suptitle("K means algorithm", size=25)
        for i in range(m):
            data, cluster_ids = args[i]
            axs[2 * i].scatter(data[:, 0], data[:, 1], edgecolor="black")
            axs[2 * i - 1].scatter(data[:, 0], data[:, 1], c=cluster_ids, cmap="rainbow")
            axs[2 * i].grid()
            axs[2 * i - 1].grid()
            axs[2 * i].set_title("Original data")
            axs[2 * i - 1].set_title("Clustered data")
            axs[2 * i].set_aspect("equal", "box")
            axs[2 * i - 1].set_aspect("equal", "box")
        fig.tight_layout()
        plt.show()

    def __show_images(self, imgs):
        m = len(imgs)

        fig, axs = plt.subplots(m // 2, 2, figsize=(6, 6))
        fig.suptitle("Image compression", size=25)

        if m >= 4:
            axs = axs.flatten()
        i = 0
        for img, ax in zip(imgs, axs):
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title("Original")
            if i == 1:
                ax.set_title("Compressed")
            i += 1
            ax.set_aspect("equal", "box")
        fig.tight_layout(h_pad=0.1, w_pad=0.001)
        plt.show()

    def __pca_visualization(self, data):
        if len(data) != 2:
            raise AttributeError(f"PCA visualization takes only 2 arguments")
        x_norm, x_rec = data
        fig = plt.figure(figsize=(6, 6))
        plt.plot(x_norm[:, 0], x_norm[:, 1], "bo", markerfacecolor="none", label="original data (normalized)")
        plt.plot(x_rec[:, 0], x_rec[:, 1], "ro", markerfacecolor="none", label="recover from 1D projected data")
        for i, j in zip(x_norm, x_rec):
            x, y = Visualization.get_line(i, j)
            plt.plot(x, y, "k--")
        plt.legend()
        plt.axis("equal")
        plt.grid()
        plt.show()

    def __pca_eigenvectors(self, args):
        data, mu, s, u = args
        mu = np.tile(mu, (len(s), 1))
        eigen_coor = np.array(mu + 1.5 * s * u.T)

        plt.figure(figsize=(6, 6))
        plt.title("Eigenvectors visualization", fontsize=18)
        plt.scatter(data[:, 0], data[:, 1], edgecolors="black")

        for i, j in zip(mu, eigen_coor):
            x, y = Visualization.get_line(i, j)
            plt.plot(x, y, "r-")
        plt.grid()
        plt.axis("equal")
        plt.show()

    def __plot_faces(self, args):
        data, title = args
        m, n = data.shape

        axis_num = np.floor(np.sqrt(m)).astype(int)
        pic_pixels = np.floor(np.sqrt(n)).astype(int)

        fig = plt.figure()
        fig.suptitle(title)
        plt.axis("off")
        plt.gray()
        for i in range(0, axis_num ** 2):
            sub = fig.add_subplot(axis_num, axis_num, i + 1)
            sub.axis("equal")
            sub.imshow(data[i].reshape((pic_pixels, -1), order="F"))
            sub.axis("off")

        fig.tight_layout()
        plt.show()

    def __plot3D(self, args):
        mat_original, mat_compressed = args
        m, n, l = mat_original.shape
        mat_original = mat_original.reshape((m * n, l))
        plt.suptitle("Original file 3D plot, pixels values and its color representation")
        x, y, z = mat_original[:, 0], mat_original[:, 1], mat_original[:, 2]
        plt.axes(projection='3d').scatter3D(x, y, z, c=mat_original, cmap="rgb")
        plt.show()

        plt.suptitle("Compressed file 3D plot, pixels values and its color representation")
        plt.axes(projection='3d').scatter3D(x, y, z, c=mat_compressed.reshape((m * n, l)), cmap="rgb")
        plt.show()


