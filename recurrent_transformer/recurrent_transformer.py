import torch
from torch.nn import Module

# helpers

def exists(v):
    return v is not None

# classes

class RecurrentTransformer(Module):
    def __init__(self):
        super().__init__()
