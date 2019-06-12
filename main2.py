import pandas as pd
import math
from node import Node
from cluster import Cluster
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

#TODO:
# run another dataset

# K nearest neighbors
k = 3

# C clusters
c = 2

def euclidean_distance(x1,x2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x1, x2)]))


data = pd.read_csv("iris.csv")

# data = pd.read_csv("jain.csv", header=None, delimiter='\t')

data = data.values
# random.shuffle(data)
# data = data[0:100]
dataLen = len(data)

# adjK = np.zeros((dataLen, dataLen))
# adjAll = np.zeros((dataLen, dataLen))


# for i in range(dataLen):
#     distances = []
#     for j in range(dataLen):
#         if i == j: continue
#         d = euclidean_distance(data[i], data[j])
#         distances.append({
#                  "distance": d,
#                  "id": j
#         })
#         adjAll[i][j] = d

#     distances.sort(key=lambda d : d.get('distance'))
#     for dis in distances[0:k]: adjK[i][dis.get('id')] = dis.get('distance')

# # Create spaning matrix
# span = minimum_spanning_tree(adjAll).toarray()

# # Convert
# adjK = 1. /(adjK + 0.1 )
# adjAll = 1. /(adjAll + 0.1 )


# # Merge
# matMerged = np.zeros((dataLen, dataLen))
# dMat = np.zeros((dataLen, dataLen))

# for i in range(dataLen):
#         for j in range(dataLen):
#                 matMerged[i][j] = adjK[i][j]
#                 m
#                 if matMerged[i][j] == None or matMerged[i][j] == 0:
#                         matMerged[i][j] = span[i][j]
#                         matMerged[j][i] = span[i][j]


# # Vg & dMat
# vg = 0
# for i in range(dataLen):
#         tmp = 0
#         for j in range(dataLen):
#                 vg += matMerged[i][j]
#                 tmp += matMerged[i][j]
        
#         dMat[i][i] = tmp

# # L
# lMat = dMat - matMerged

# # L+
# lPlusMat = np.linalg.pinv(lMat)

# nMat = np.zeros((dataLen, dataLen))
# for i in range(dataLen):
#         for j in range(dataLen):
#                 eVector = np.zeros((dataLen, 1))
#                 eVector[i] = 1
#                 eVector[j] = -1

#                 nMat[i][j] = vg * np.dot(np.dot(eVector.T, lPlusMat), eVector)


# ##########################################################################################

# # c random prototype
# prototypes = np.random.randint(0, dataLen, size=c)
# labels = np.zeros(dataLen)

# # Initialize clusters
# clusters = []
# for i in range(c):
#         clusters.append(Cluster(i ,np.random.randint(0, dataLen)))


# for kk in range(30):

#         # Allocation of the observations
#         for i in range(dataLen):
#                 min = 100000000000
#                 nearestCluster = None
#                 for cluster in clusters:
#                         cluster.members.discard(i)
#                         if nMat[i][cluster.prototype] ** 2 < min:
#                                 min = nMat[i][cluster.prototype] ** 2 
#                                 nearestCluster = cluster
#                 nearestCluster.members.add(i)
#                 labels[i] = nearestCluster.id


#         # Computation of the prototypes
#         for cluster in clusters:
#                 minSum = 99999999999999
#                 minM = 0
#                 for m in cluster.members:
#                         sum = 0
#                         for n in cluster.members:
#                                 if n == m : continue
#                                 sum += nMat[m][n] ** 2
#                         if sum < minSum:
#                                 minSum = sum
#                                 minM = m
               
#                 cluster.prototype = minM

# # print(len(clusters[0].members))
# print(len(clusters[1].members))
# print(len(clusters[2].members))

# print(clusters[0].prototype)
# print(clusters[1].prototype)
# print(clusters[2].prototype)

# plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
# plt.show()


from sklearn.metrics.cluster import normalized_mutual_info_score
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3)
# kmeans.fit_predict(data)
# plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
# print(normalized_mutual_info_score(kmeans.labels_, data[:, 2]))

from sklearn.cluster import AgglomerativeClustering
hierarchical = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
hierarchical.fit_predict(data) 
print(normalized_mutual_info_score(hierarchical.labels_, data[:, 2]))
# plt.scatter(data[:, 0], data[:, 1], c=hierarchical.labels_, s=50, cmap='viridis')

# plt.show()
