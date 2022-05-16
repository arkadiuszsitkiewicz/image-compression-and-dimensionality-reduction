import numpy as np
from src.data.prepare_data import PrepareData
from src.calc.K_means import KMeans
from src.finals.compress import CompressFile
from src.vis.visulize import Visualization
from src.calc.pca import PrincipalComponentAnalysis

# # # todo Kmeans - cluster data
data = PrepareData("data/raw/data.csv", "raw data").converted_data
k = 3
num_iter = 100
cluster_ids = KMeans(data, k, num_iter).cluster_ids
Visualization((data, cluster_ids), vistype="clustered data")

# # # todo Kmeans for image compression
img_bird = PrepareData("data/raw/bird_small.png", "image").converted_data
img_bird_compressed = CompressFile(img_bird, 4).compressed
Visualization(img_bird, img_bird_compressed, vistype="image")

# # # todo 3D plot of a raw and compressed data
Visualization(img_bird, img_bird_compressed, vistype="plot 3d")


# # # todo PCA - calculating and vis of eigenvectors on simple 2D example
data = PrepareData("data/raw/data_pca.csv", "raw data").converted_data
pca = PrincipalComponentAnalysis(data)
mu = pca.mean_norm()
s = pca.eigenvalues(2)
u = pca.eigenvectors(2)
Visualization(data, mu, s, u, vistype="PCA EIGENVECTORS")

# # # todo PCA - dimension reduction from 2D to 1D, and then recover back to 2D
x_norm = pca.x_norm
z = pca.reduce_dim(1)
x_rec = pca.recover_from_dim(z, 1)
Visualization(x_norm, x_rec, vistype="PCA vis")


# # # # # # todo PCA - reduction face photos from 1024px to 100px
# todo uploading face images data and vis random num_ex examples (out of 2.5k)
faces_data = np.load("data/modified/faces_data.npy")


num_ex = 10
rand_idx = np.random.choice(len(faces_data), num_ex)
Visualization(faces_data[rand_idx, :], "Data set visualization for image compression with PCA", vistype="PLOT FACES")

# todo run PCA algorithm
faces_pca = PrincipalComponentAnalysis(faces_data)

# todo visualizing eigenvectors - it looks like a mask that shows you how many details the image maintains
num = 36
eigenvec = faces_pca.eigenvectors(num)
Visualization(eigenvec.T, f"First {num} principal components that describes larges variations", vistype="PLOT FACES")

# todo reduce original dimension (1024px) to k dimension
k = 100
z_faces = faces_pca.reduce_dim(k)
# todo recover back to original dimension for the purpose of visualization results
x_rec_faces = faces_pca.recover_from_dim(z_faces, k)
Visualization(x_rec_faces[rand_idx, :], f"Reconstructed data from {k} eigenfaces", vistype="PLOT FACES")




