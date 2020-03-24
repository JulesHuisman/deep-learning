def start(config):
    worker = OptimizeWorker(config)

    for _ in range(5):
        print(__name__)

class OptimizeWorker:
    def __init__(self, config):
        self.config = config