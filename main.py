import pandas as pd
import math
from node import Node
# Clustering using a random walk based distance measure

#TODO:
# euclidian distance
# spanning tree
# ETC

k = 3

def euclidean_distance(x1,x2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x1, x2)]))

data = pd.read_csv("data.csv")
data = data.as_matrix()


graph = []
distances = []

for i in range(len(data)):
    graph.append(Node(data[i][0], data[i][1], i))
    for j in range(i+1, len(data)):
        distances.append({'i':i , 'j':j, 'dis':euclidean_distance(data[i], data[j])})


print(len(data))
print(len(distances))