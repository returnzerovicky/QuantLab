"""
QuantLab AI
Loss Functions
"""

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error Loss
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        return torch.sqrt(
            self.mse(prediction, target) + 1e-8
        )


class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss
    """

    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, prediction, target):
        return self.mae(prediction, target)


class HuberLoss(nn.Module):
    """
    Robust loss for noisy financial data.
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta)

    def forward(self, prediction, target):
        return self.loss(prediction, target)


class QuantileLoss(nn.Module):
    """
    Quantile Loss
    Useful for probabilistic forecasting.
    """

    def __init__(self, q=0.5):
        super().__init__()
        self.q = q

    def forward(self, prediction, target):

        error = target - prediction

        return torch.mean(
            torch.max(
                self.q * error,
                (self.q - 1) * error
            )
        )


def get_loss(name="mse"):

    name = name.lower()

    if name == "mse":
        return nn.MSELoss()

    if name == "mae":
        return MAELoss()

    if name == "rmse":
        return RMSELoss()

    if name == "huber":
        return HuberLoss()

    if name == "quantile":
        return QuantileLoss()

    raise ValueError(
        f"Unknown loss function: {name}"
    )
