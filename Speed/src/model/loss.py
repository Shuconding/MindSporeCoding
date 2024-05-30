import mindspore.nn as nn
import mindspore.numpy as np


class TGCNLoss(nn.Cell):
    """
    Custom T-GCN loss cell
    """

    def construct(self, predictions, targets):
        """
        Calculate loss

        Args:
            predictions(Tensor): predictions from models
            targets(Tensor): ground truth

        Returns:
            loss: loss value
        """
        targets = targets.reshape((-1, targets.shape[2]))
        return np.sum((predictions - targets) ** 2) / 2
