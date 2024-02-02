from abc import abstractmethod


class Algorithm:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)

    @abstractmethod
    def main(self, *args, **kwargs):
        pass
