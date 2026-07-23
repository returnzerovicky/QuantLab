"""
QuantLab AI
Training Pipeline
"""

import copy
import time
import torch
import torch.optim as optim
from tqdm import tqdm

from .config import config, experiment
from .dataset import get_dataloaders
from .model import build_model
from .losses import get_loss
from .utils import (
    get_device,
    save_checkpoint,
    save_metrics,
)

device = get_device()


class EarlyStopping:

    def __init__(self, patience=10):

        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.best_weights = None
        self.stop = False

    def __call__(self, loss, model):

        if loss < self.best_loss:

            self.best_loss = loss
            self.counter = 0
            self.best_weights = copy.deepcopy(
                model.state_dict()
            )

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.stop = True


class Trainer:

    def __init__(self, model_name="lstm"):

        self.model = build_model(
            model_name
        ).to(device)

        self.train_loader, self.test_loader, \
        self.scaler, self.df = get_dataloaders()

        self.criterion = get_loss("mse")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        self.early_stopping = EarlyStopping(
            patience=config.patience
        )

        self.history = {
            "train_loss": [],
            "val_loss": []
        }
    def train_epoch(self):

        self.model.train()

        running_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False
        )

        for inputs, targets in progress:

            inputs = inputs.to(device)

            targets = targets.to(device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(
                outputs,
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            self.optimizer.step()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.6f}"
            )

        epoch_loss = (
            running_loss /
            len(self.train_loader)
        )

        return epoch_loss


    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        for inputs, targets in self.test_loader:

            inputs = inputs.to(device)

            targets = targets.to(device)

            outputs = self.model(inputs)

            loss = self.criterion(
                outputs,
                targets
            )

            running_loss += loss.item()

        epoch_loss = (
            running_loss /
            len(self.test_loader)
        )

        return epoch_loss


    def fit(self):

        print("=" * 60)
        print("QuantLab AI Training")
        print("=" * 60)

        start = time.time()

        for epoch in range(config.epochs):

            train_loss = self.train_epoch()

            val_loss = self.validate()

            self.scheduler.step(val_loss)

            self.history["train_loss"].append(
                train_loss
            )

            self.history["val_loss"].append(
                val_loss
            )

            print(
                f"Epoch [{epoch+1}/{config.epochs}] | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f}"
            )

            self.early_stopping(
                val_loss,
                self.model
            )

            if self.early_stopping.stop:

                print(
                    "Early stopping triggered."
                )

                break

        end = time.time()

        print(
            f"Training Time: {(end-start):.2f} sec"
        )


        if self.early_stopping.best_weights is not None:
            self.model.load_state_dict(
                self.early_stopping.best_weights
            )

        save_checkpoint(
    model=self.model,
    optimizer=self.optimizer,
    epoch=len(self.history["train_loss"]),
    loss=self.early_stopping.best_loss,
    path=config.checkpoint_path
)

        metrics = {
            "best_validation_loss":
                self.early_stopping.best_loss,
            "final_training_loss":
                self.history["train_loss"][-1],
            "epochs_completed":
                len(self.history["train_loss"]),
            "learning_rate":
                self.optimizer.param_groups[0]["lr"],
            "model":
                self.model.__class__.__name__
        }

        if experiment.save_metrics:
            save_metrics(
                metrics,
                experiment.metrics_path
            )

        print("=" * 60)
        print("Training Finished Successfully")
        print("=" * 60)

        return self.model


def main():

    trainer = Trainer(
        model_name="lstm"
    )

    trainer.fit()


if __name__ == "__main__":
    main()
