# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:35:57 2020

@author: fgm.si
"""

#########################################
# K-means clustering using scikit-learn #
#########################################
"""
1. Randomly pick k centroids from the sample points as initial cluster centers.
2. Assign each sample to the nearest centroid.
3. Move the centroids to the center of the samples that were assigned to it.
4. Repeat steps 2 and 3 until the cluster assignments do not change or a user-defined tolerance or maximum number of iterations is reached.
"""
# Create the data

from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples = 150,
                  n_features = 2,
                  centers = 3,
                  cluster_std = 0.5,
                  shuffle = True,
                  random_state = 0)

import matplotlib.pyplot as plt
plt.scatter(X[:,0],
            X[:,1],
            c = "white",
            marker = "o",
            edgecolor = "black",
            s = 50)
plt.grid()
plt.show()

# Set the algorithm 

from sklearn.cluster import KMeans
km = KMeans(
            """
            - Set the desired clusters to 3.
            - Initialize in a random centroid.
            - Run the k-means clustering algorithms 10 times indepentdently to choose the final model as the one with the lowest sum of squared errors.
            - We set 300 as the maximum number of iterations for each single run.
            - "tol" is the parameter that controls the tolerance with regard to the changes in the withning cluster SSE to declare convergence.
            """
            n_clusters = 3, 
            init = "random",
            n_init = 10, 
            max_iter = 300,
            tol = 0.0001,
            random_state = 0)
y_km = km.fit_predict(X)

# Viz the algorithm

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            c = "lightgreen",
            marker = "s",
            edgecolor = "black",
            s = 50,
            label = "cluster 1")
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            c = "orange",
            marker = "o",
            edgecolor = "black",
            s = 50,
            label = "cluster 2")
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            c = "lightblue",
            marker = "v",
            edgecolor = "black",
            s = 50,
            label = "cluster 3")
plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            c = "red",
            marker = "*",
            edgecolor = "black",
            s = 250,
            label = "centroids")
plt.legend(scatterpoints = 1)
plt.grid()
plt.show()

            