import math

class Node:

    def __init__(self, x, y, id):

        self.x = x
        self.y = y

        self.id = id
        self.distances = []
        self.connections = {}

    def total_weight(self):
        sum = 0
        for c in self.connections:
            sum+=self.connections[c]
        return sum

    def euclidean_distance(self, node):
        return math.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)

    def connect_to(self, nodeId, weight):
        if nodeId not in self.connections:
            self.connections[nodeId] = weight

    def __str__(self):
        return 'id: ' + str(self.id) + ', X: ' + str(self.x) + ', ' + 'Y: ' + str(self.y)
