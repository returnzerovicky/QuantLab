"""
QuantLab AI
Future Stock Price Prediction
"""

import numpy as np
import torch

from .config import config
from .dataset import DataProcessor
from .model import build_model
from .utils import (
    get_device,
    inverse_transform,
    load_checkpoint,
)

device = get_device()


@torch.no_grad()
def predict_next_day(model_name="lstm"):

    processor = DataProcessor()

    df = processor.download_data()

    scaled = processor.preprocess(df)

    sequence = scaled[-config.window_size:]

    sequence = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    model = build_model(model_name).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    model, optimizer, _, _ = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=config.checkpoint_path
    )

    model.eval()

    prediction = model(sequence)

    prediction = prediction.cpu().numpy()

    prediction = inverse_transform(
        processor.scaler,
        prediction
    )

    print("=" * 60)
    print("Next Day Prediction")
    print("=" * 60)
    print(f"Predicted Close Price: ${prediction[0][0]:.2f}")

    return prediction[0][0]


if __name__ == "__main__":

    predict_next_day()
