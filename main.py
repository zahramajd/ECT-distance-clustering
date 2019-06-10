import pandas as pd
import math
from node import Node
import numpy as np
import random
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

#TODO:
# spanning tree
# ETC

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
    for d in distances[0:3]: adj4[i][d.get('id')] = d.get('distance')
    for d in distances: adjAll[i][d.get('id')] = d.get('distance')

# Create spaning matrix
span = minimum_spanning_tree(adjAll).toarray()

# Merge
matMerged = np.zeros((dataLen, dataLen))
for i in range(dataLen):
        for j in range(dataLen):
                matMerged[i][j] = adj4[i][j]
                if matMerged[i][j] == None:
                        adj4[i][j] = span[i][j]

print('Alt + Tab please!')
plt.matshow(matMerged)
plt.show()