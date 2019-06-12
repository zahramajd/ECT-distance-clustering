import pandas as pd
import math
from node import Node
from cluster import Cluster
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import sys
from sklearn.metrics.cluster import normalized_mutual_info_score


# Params
numOfNeighbors = 3
numOfClusters = 3
maxIterations = 100

# Read data
# data = pd.read_csv("jain.csv", header=None, delimiter='\t').values
data = pd.read_csv("iris.csv", header=None).values


# Create Nodes
nodes = []

for i in range(len(data)):
    nodes.append(Node(data[i][:-1], i))

# Add k nearest nodes
for node in nodes:
    for node2 in nodes:
        if node == node2: continue
        node.distances.append((node2.id, node.euclidean_distance(node2)))
    node.distances.sort(key = lambda d : d[1])

    for d in node.distances[0:numOfNeighbors]:
        if( not d[1]==0 ):
                id = d[0]
                distance = d[1]
                node.connect_to(d[0], 1./distance)

# Create a graph
fullyConnectedGraph = np.zeros((len(nodes), len(nodes)))

for node1 in nodes:
    for node2 in nodes:
        fullyConnectedGraph[node1.id][node2.id] = node1.euclidean_distance(node2)

mst = minimum_spanning_tree(fullyConnectedGraph).toarray()

for i in range(len(nodes)):
    for j in range(len(nodes)):
        if mst[i][j] != 0:
            w = 1./mst[i][j]
            nodes[i].connect_to(j, w)
            nodes[j].connect_to(i, w)

# Create Adjacency graph
A = np.zeros((len(nodes), len(nodes)))

for n in nodes:
    for c in n.connections:
        A[n.id][c] = n.connections[c]

# Create D matrix
D = np.zeros((len(nodes), len(nodes)))

for i in range(len(nodes)):
    D[i][i]=nodes[i].total_weight()

# Create L matrix
L = D - A

# L+
lPlus = np.linalg.pinv(L)

# VG
vg = np.sum(A)

# N
N = np.zeros((len(nodes), len(nodes)))

for i in range(len(nodes)):
    for j in range(len(nodes)):
        e = np.zeros((len(nodes), 1))
        e[i] = 1
        e[j] = -1
        N[i][j] = vg * np.matrix.dot(np.matrix.dot(np.matrix.transpose(e), lPlus), e)

# Initialize clusters
clusters = []
prototypes = set()
for i in range(numOfClusters):
    rand = np.random.randint(0, len(nodes) - 1)
    while rand in prototypes:
        rand = np.random.randint(0, len(nodes) - 1)
    prototypes.add(rand)
    clusters.append(Cluster(i, rand))

# Do labeling
labels = np.zeros(len(nodes))
iteration = 0
while True:
        iteration+=1
        if iteration > maxIterations: break
        print('Iteration ', iteration + 1 , '/' , maxIterations)
        changed = False

        # Allocation of the observations
        for i in range(len(nodes)):
            min = math.inf
            nearestCluster = None
            for cluster in clusters:
                cluster.members.discard(i)
            for cluster in clusters:
                if N[i][cluster.prototype] ** 2 < min:
                        min = N[i][cluster.prototype] ** 2 
                        nearestCluster = cluster
            nearestCluster.members.add(i)
            labels[i] = nearestCluster.id

        # Computation of the prototypes
        for cluster in clusters:
            minSum = math.inf
            minM = 0
            for m in cluster.members:
                    sum = 0
                    for n in cluster.members:
                            if n == m : continue
                            sum += N[m][n] ** 2
                    if sum < minSum:
                            minSum = sum
                            minM = m
            if minM != cluster.prototype:
                cluster.prototype = minM
                changed = True

        # if changed:
        #     print('change!')

# print('Clusters:')
# for c in clusters:
#     print(c)



nmi = normalized_mutual_info_score(labels, data[:, 2], average_method='arithmetic')
print(nmi)
# plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')

# plt.title('Spiral')
# plt.show()