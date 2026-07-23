"""
QuantLab AI
Model Evaluation Pipeline
"""

import torch
import numpy as np

from .config import config
from .dataset import get_dataloaders
from .model import build_model
from .utils import (
    get_device,
    calculate_metrics,
    inverse_transform,
    plot_predictions,
    load_checkpoint,
)

device = get_device()


@torch.no_grad()
def evaluate(model_name="lstm"):

    model = build_model(model_name).to(device)

    _, test_loader, scaler, _ = get_dataloaders()

    optimizer = torch.optim.Adam(model.parameters())

    model, optimizer, epoch, loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=config.checkpoint_path
    )

    model.eval()

    predictions = []
    actuals = []

    for inputs, targets in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        predictions.extend(
            outputs.squeeze().cpu().numpy()
        )

        actuals.extend(
            targets.squeeze().cpu().numpy()
        )

    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    predictions = inverse_transform(
        scaler,
        predictions
    )

    actuals = inverse_transform(
        scaler,
        actuals
    )

    metrics = calculate_metrics(
        actuals,
        predictions
    )

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    plot_predictions(
        actuals,
        predictions
    )

    return metrics


if __name__ == "__main__":

    evaluate()
