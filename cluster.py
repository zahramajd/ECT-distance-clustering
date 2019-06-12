
class Cluster:
    
    def __init__(self, id ,prototype):

        self.prototype = prototype
        self.members = set()
        self.members.add(self.prototype)
        self.id = id

    def __str__(self):
        return 'id: ' + str(self.id) + ', members: ' + str(len(self.members)) + ', prototype: ' + str(self.prototype)
