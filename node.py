import math

class Node:

    def __init__(self, data, id):
        self.data = data
        self.id = id
        self.distances = []
        self.connections = {}

    def total_weight(self):
        sum = 0
        for c in self.connections:
            sum+=self.connections[c]
        return sum

    def euclidean_distance(self, node):
        sum = 0
        for i in range(len(self.data)):
            sum += (self.data[i] - node.data[i]) ** 2
        return math.sqrt(sum)

    def connect_to(self, nodeId, weight):
        if nodeId not in self.connections:
            self.connections[nodeId] = weight

    def __str__(self):
        return 'id: ' + str(self.id) + ', X: ' + str(self.x) + ', ' + 'Y: ' + str(self.y)
