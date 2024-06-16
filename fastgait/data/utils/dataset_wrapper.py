import copy
from .base_dataset import Dataset

class IterLoader:
    """
    Wrapper for repeating dataloaders
    """

    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.loader)

    def new_epoch(self, epoch):
        self.loader.sampler.set_epoch(epoch)
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except Exception:
            self.iter = iter(self.loader)
            return next(self.iter)
