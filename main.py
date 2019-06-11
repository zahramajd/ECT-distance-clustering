import pandas as pd
import math
from node import Node
import numpy as np
import random
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


k = 3

def euclidean_distance(x1,x2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x1, x2)]))

data = pd.read_csv("data.csv")
data = data.values
random.shuffle(data)
data = data[0:10]

dataLen = len(data)
adj4 = np.zeros((dataLen, dataLen))
adjAll = np.zeros((dataLen, dataLen))

for i in range(dataLen):
    distances = []
    for j in range(dataLen):
        if i == j: continue
        d = euclidean_distance(data[i], data[j])
        distances.append({
                 "distance": d,
                 "id": j
        })
    distances.sort(key=lambda d : d.get('distance'))
    for d in distances[0:k]: adj4[i][d.get('id')] = d.get('distance')
    for d in distances: adjAll[i][d.get('id')] = d.get('distance')

# Create spaning matrix
span = minimum_spanning_tree(adjAll).toarray()

# Merge
matMerged = np.zeros((dataLen, dataLen))
dMat = np.zeros((dataLen, dataLen))

vg = 0

for i in range(dataLen):
        tmp = 0
        for j in range(dataLen):
                matMerged[i][j] = adj4[i][j]
                if matMerged[i][j] == None:
                        adj4[i][j] = span[i][j]

                vg += matMerged[i][j]
                tmp += matMerged[i][j]
        
        dMat[i][i] = tmp

# L
lMat = dMat - matMerged

# L+
lPlusMat = np.linalg.pinv(lMat)

nMat = np.zeros((dataLen, dataLen))

for i in range(dataLen):
        for j in range(dataLen):
                eVector = np.zeros((dataLen, 1))
                eVector[i] = 1
                eVector[j] = -1

                nMat[i][j] = vg * np.dot(np.dot(eVector.T,lPlusMat), eVector)

# print('Alt + Tab please!')
# plt.matshow(matMerged)
# plt.show()