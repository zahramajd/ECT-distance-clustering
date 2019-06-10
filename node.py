

class Node:

    def __init__(self, x, y, id):

        self.x = x
        self.y = y

        self.id = id
        self.neighbors = []

    def show(self):
        print(self.neighbors)

