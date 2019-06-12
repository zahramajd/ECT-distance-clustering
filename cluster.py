
class Cluster:
    def __init__(self, id ,prototype):
        self.prototype = prototype
        self.members = set()
        self.members.add(self.prototype)
        self.id = id
