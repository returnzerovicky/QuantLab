"""
QuantLab AI
Utility Functions
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def get_device():

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def inverse_transform(scaler, values):

    return scaler.inverse_transform(values)


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred)
                / (y_true + 1e-8)
            )
        )
        * 100
    )

    return {

        "MAE": float(mae),

        "MSE": float(mse),

        "RMSE": float(rmse),

        "MAPE": float(mape),

        "R2": float(r2),

    }


def save_metrics(metrics, path):

    df = pd.DataFrame([metrics])

    if os.path.exists(path):

        old = pd.read_csv(path)

        df = pd.concat(
            [old, df],
            ignore_index=True
        )

    df.to_csv(path, index=False)


def plot_predictions(
    actual,
    predicted,
    save_path="reports/prediction_plot.png"
):

    plt.figure(figsize=(14, 6))

    plt.plot(
        actual,
        label="Actual"
    )

    plt.plot(
        predicted,
        label="Predicted"
    )

    plt.title(
        "Actual vs Predicted Stock Prices"
    )

    plt.xlabel("Time")

    plt.ylabel("Price")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path,
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "loss": loss,

    }

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    model,
    optimizer,
    path,
):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return (

        model,

        optimizer,

        checkpoint["epoch"],

        checkpoint["loss"]

    )
