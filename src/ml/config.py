"""
QuantLab AI Configuration
Production-grade configuration for training and experimentation.
"""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
EXPERIMENT_DIR = ROOT / "experiments"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    # Data
    ticker: str = "AAPL"
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"

    # Sequence
    window_size: int = 30

    # Split
    train_split: float = 0.8

    # Model
    input_size: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 1
    dropout: float = 0.2

    # Training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Runtime
    random_seed: int = 42
    num_workers: int = 0

    # Early Stopping
    patience: int = 10

    # Checkpoints
    checkpoint_name: str = "lstm_stock_model.pth"

    @property
    def checkpoint_path(self):
        return MODEL_DIR / self.checkpoint_name


@dataclass
class ExperimentConfig:

    experiment_name: str = "LSTM Forecasting"

    save_predictions: bool = True

    save_metrics: bool = True

    metrics_file: str = "results.csv"

    @property
    def metrics_path(self):
        return EXPERIMENT_DIR / self.metrics_file


config = TrainingConfig()
experiment = ExperimentConfig()
