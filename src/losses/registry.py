from torch import nn


def build_loss() -> nn.Module:
    return nn.CrossEntropyLoss()
