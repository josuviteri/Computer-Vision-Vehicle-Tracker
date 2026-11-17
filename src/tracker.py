

class Track():
    def __init__(self, id:int):
        self.id = id
        self.state = None
        self.hits = 0
        self.misses = 0
        self.history = []